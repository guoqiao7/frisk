"""Factor Copy16 group addresses through checked GF(2) direct-sum translates."""

from dataclasses import dataclass

from access_plan import linear
from attention_tile_mma import THREADS, TILE, shared_offsets_64x64
from valid_domain import GuardedCopyPlan


_ALL_THREADS = (1 << THREADS) - 1
_ROW_BYTES = TILE * 2


@dataclass(frozen=True)
class CopyTranslate:
    source_delta: int
    destination_delta: int
    row_delta: int


@dataclass(frozen=True)
class CopySchedule:
    source_base: tuple[int, ...]
    destination_base: tuple[int, ...]
    rows_base: tuple[int, ...]
    translates: tuple[CopyTranslate, ...]


def _carry_free_delta(base, target):
    if len(base) != len(target) or not base:
        return None
    delta = target[0] - base[0]
    if delta < 0 or any(value & delta or actual != value + delta
                        for value, actual in zip(base, target, strict=True)):
        return None
    return delta


def prove_copy_schedule(plan):
    """Return a factorization only when every native address is preserved."""
    if (not isinstance(plan, GuardedCopyPlan)
            or plan.domain.carrier_shape != (TILE, TILE)
            or plan.domain.valid_shape[1] != TILE
            or not 1 <= plan.domain.valid_shape[0] <= TILE
            or len(plan.groups) != 4):
        return None
    first = plan.groups[0]
    native_offsets = shared_offsets_64x64()
    prefix = plan.domain.valid_shape[0]
    translates = []
    sources, destinations = set(), set()
    for group in plan.groups:
        tables = (group.rows, group.columns, group.source_offsets, group.destination_offsets)
        if any(len(table) != THREADS or any(type(value) is not int for value in table)
               for table in tables):
            return None
        for row, column, source, destination in zip(*tables, strict=True):
            if (not 0 <= row < TILE or not 0 <= column < TILE or column % 8
                    or source != row * _ROW_BYTES + column * 2
                    or source % 16 or destination % 16
                    or destination != native_offsets[row * TILE + column]
                    or source in sources or destination in destinations):
                return None
            sources.add(source)
            destinations.add(destination)
        if group.active_mask != sum(1 << lane for lane, row in enumerate(group.rows) if row < prefix):
            return None
        source_delta = _carry_free_delta(first.source_offsets, group.source_offsets)
        destination_delta = _carry_free_delta(first.destination_offsets, group.destination_offsets)
        row_delta = _carry_free_delta(first.rows, group.rows)
        if (source_delta is None or destination_delta is None or row_delta is None
                or source_delta % 16 or destination_delta % 16
                or group.columns != first.columns):
            return None
        translates.append(CopyTranslate(source_delta, destination_delta, row_delta))
    if sources != set(range(0, TILE * _ROW_BYTES, 16)) or destinations != sources:
        return None
    try:
        for table in (first.source_offsets, first.destination_offsets, first.rows):
            linear(table)
    except ValueError:
        return None
    return CopySchedule(first.source_offsets, first.destination_offsets, first.rows,
                        tuple(translates))


def _address(pointer, displacement):
    return f'[{pointer}+{displacement}]' if displacement else f'[{pointer}]'


def emit_factored_copy(emitter, root, tile_row, shared_offset, *, guard,
                       prefix=TILE, storage_rows=TILE):
    """Emit a checked copy, or return False before emitting any instructions.

    Bases are local to this call. The caller retains its original publication
    barrier and may use its ordinary copy emitter when factorization fails.
    """
    if (type(prefix) is not int or not 1 <= prefix <= TILE
            or type(storage_rows) is not int or storage_rows not in (8, 16, 32, TILE)
            or storage_rows < prefix):
        raise ValueError('copy publication must cover the checked valid prefix')
    plan = emitter.copy_plans.get(prefix)
    schedule = prove_copy_schedule(plan)
    if schedule is None:
        return False
    if (type(guard) is not bool or type(emitter.length) is not int
            or not 64 <= emitter.length <= 1025
            or type(shared_offset) is not int or shared_offset < 0
            or shared_offset % 16
            or shared_offset + storage_rows * _ROW_BYTES > emitter.mma_tiles.shared_bytes):
        return False
    selected = [(group, translate) for group, translate in zip(plan.groups, schedule.translates, strict=True)
                if min(group.rows) < storage_rows]
    if not selected:
        return False
    for group, _ in selected:
        group_guard = guard and not (emitter.simplify_tail_copy and prefix < TILE
                                     and group.active_mask == _ALL_THREADS)
        if not group_guard and not group.active_mask:
            raise ValueError('inactive copy requires checked zero publication')

    tile = emitter._tile_pointer(root, tile_row, _ROW_BYTES)
    source_offset = emitter._value('cvt.u64.u32', emitter.map(schedule.source_base), kind='b64')
    source = emitter._value('add.u64', tile, source_offset, kind='b64')
    shared = emitter._value('add.u32', emitter.smem_base, shared_offset)
    destination = emitter._value('add.u32', shared, emitter.map(schedule.destination_base))
    need_rows = any((guard and group.active_mask
                     and not (emitter.simplify_tail_copy and prefix < TILE
                              and group.active_mask == _ALL_THREADS))
                    or max(group.rows) >= storage_rows for group, _ in selected)
    row_base = emitter.map(schedule.rows_base) if need_rows else None
    absolute_row = emitter._value('add.u32', tile_row, row_base) if need_rows and guard else None
    vector = [emitter.tmp('b32') for _ in range(4)]
    values = '{' + ', '.join(vector) + '}'
    for group, translate in selected:
        group_guard = guard and not (emitter.simplify_tail_copy and prefix < TILE
                                     and group.active_mask == _ALL_THREADS)
        if group_guard:
            for register in vector:
                emitter.emit(f'mov.b32 {register}, 0;')
        if group.active_mask:
            qualifier = ''
            if group_guard:
                predicate = emitter._value('setp.lt.u32', absolute_row,
                                           emitter.length - translate.row_delta, kind='pred')
                qualifier = f'@{predicate} '
            emitter.emit(f'{qualifier}ld.global.v4.b32 {values}, '
                         f'{_address(source, translate.source_delta)};')
        qualifier = ''
        if max(group.rows) >= storage_rows:
            predicate = emitter._value('setp.lt.u32', row_base,
                                       storage_rows - translate.row_delta, kind='pred')
            qualifier = f'@{predicate} '
        emitter.emit(f'{qualifier}st.shared.v4.b32 '
                     f'{_address(destination, translate.destination_delta)}, {values};')
    return True

