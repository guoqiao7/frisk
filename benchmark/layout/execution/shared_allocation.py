"""Static Shared allocation of the bounded scalar GPU adapter.

This plan is available before PTX/native compilation. It consumes execution
contracts and redistribution proofs, never emitter metadata or measured CUDA
attributes. It models declared allocation, not the SSA active-root peak.
"""
from redistribution import validate as validate_redistribution
from async_copy import plan as validate_async_copy
from storage_addresses import RowMajorAddresses, needs_storage_normalization

UINT64_MAX = 2**64 - 1
SUPPORTED_OPERATIONS = frozenset({
    "builtin.module", "frisk.kernel", "frisk.parallel", "frisk.end",
    "frisk.storage_contract", "frisk.layout_view", "memref.alloc", "tensor.empty", "linalg.yield",
    "arith.constant", "frisk.copy", "frisk.tile_load", "frisk.convert_layout", "frisk.mma",
    "frisk.reduce_tensor", "linalg.broadcast", "frisk.tile_store", "arith.addf", "arith.mulf", "arith.subf",
    "arith.divf", "math.exp", "math.rsqrt", "arith.truncf",
})


def _u64(value, field):
    if type(value) is not int or not 0 <= value <= UINT64_MAX:
        raise ValueError("invalid Shared " + field)
    return value


def reserve_shared(used, size, alignment, limit):
    # Preserve the existing checked allocator API and overflow/limit semantics.
    if any(type(x) != int or x < 0 or x > UINT64_MAX for x in (used, size, alignment, limit)) or not alignment or alignment & (alignment - 1):
        raise ValueError("invalid Shared reservation")
    if used > 2**64 - alignment:
        raise ValueError("Shared alignment overflow")
    aligned = (used + alignment - 1) // alignment * alignment
    if aligned > limit or size > limit - aligned:
        raise ValueError("CTA Shared budget exceeded")
    return aligned + size


def _space(storage):
    layout = storage.get("layout")
    if type(layout) is not str:
        raise ValueError("storage layout missing")
    spaces = [space for space in ("Global", "Shared") if "memory_space " + space in layout]
    if len(spaces) != 1:
        raise ValueError("unsupported storage memory space")
    return spaces[0]


def _extent(storage):
    addresses = storage.get("addresses")
    if type(addresses) is RowMajorAddresses:
        return addresses.extent()
    if type(addresses) is not list or not addresses:
        raise ValueError("empty Shared root footprint")
    end = 0
    for address in addresses:
        if type(address) is not dict:
            raise ValueError("invalid Shared address")
        begin = _u64(address.get("begin_bit"), "address begin")
        upper = _u64(address.get("end_bit"), "address end")
        if begin >= upper:
            raise ValueError("invalid Shared address interval")
        end = max(end, upper)
    return (end + 7) // 8, end


def plan_shared_allocation(data, *, redistribution_validator=None, lifetimes=None, epilogue_bytes=0, reuse_after=None):
    """Return a JSON-serializable checked pre-native allocation plan.

    Current lowering emits sequential operations. Copy waits for completion
    before it returns and every Shared redistribution completes with CTA
    barriers. Thus all redistribution operations reuse one maximum-sized scratch
    segment. Pipelined overlap, unrecognized completion contracts or control flow
    are unsupported; they must not silently inherit this reuse rule.
    """
    checked_redistribution = validate_redistribution if redistribution_validator is None else redistribution_validator
    if not callable(checked_redistribution):
        raise TypeError("redistribution validator must be callable")
    if isinstance(data, dict) and (data.get("schema_version") == 2 or needs_storage_normalization(data.get("values"))):
        from execution_support import normalize_packet
        data = normalize_packet(data)
    if type(data) is not dict or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("unsupported execution schema for Shared planning")
    if set(data) - {"schema_version", "values", "operations", "shared_limit_bytes"}:
        raise ValueError("unknown execution fields for Shared planning")
    if type(data.get("values")) is not list or type(data.get("operations")) is not list:
        raise ValueError("execution values and operations required")
    values = {}
    for value in data["values"]:
        if type(value) is not dict:
            raise ValueError("invalid execution value")
        identity = _u64(value.get("id"), "value identity")
        if identity in values:
            raise ValueError("duplicate execution value identity")
        values[identity] = value
    limit = _u64(data.get("shared_limit_bytes", 227328), "limit")
    roots, conversions, copies, parallel_count = [], [], [], 0
    tma_operations = []
    for index, operation in enumerate(data["operations"]):
        if type(operation) is not dict or operation.get("name") not in SUPPORTED_OPERATIONS:
            raise ValueError("unsupported lowering operation " + str(operation.get("name") if isinstance(operation, dict) else operation))
        name = operation["name"]
        operands = operation.get("operands")
        if type(operands) is not list or any(type(x) is not int or x not in values for x in operands):
            raise ValueError("invalid execution operands")
        attrs = operation.get("attrs", {})
        if type(attrs) is not dict:
            raise ValueError("invalid operation attributes")
        if name == "frisk.parallel":
            parallel_count += 1
            if parallel_count > 1 or attrs.get("ranges") != [1] or any(type(x) is not int for x in attrs.get("ranges", [])) or type(attrs.get("threads")) is not int or attrs["threads"] != 128:
                raise ValueError("unsupported overlapping/parallel Shared lifetime")
        elif name == "frisk.storage_contract":
            if len(operands) != 1 or "storage" not in values[operands[0]]:
                raise ValueError("storage contract root missing")
            root = _u64(values[operands[0]]["storage"].get("root"), "root identity")
            if root not in values or "storage" not in values[root]:
                raise ValueError("invalid canonical storage root")
            if _u64(values[root]["storage"].get("root"), "canonical root identity") != root:
                raise ValueError("invalid canonical storage root")
            if root not in roots:
                roots.append(root)
        elif name == "frisk.convert_layout":
            conversions.append((index, checked_redistribution(operation, values)))
        elif name == "frisk.copy":
            contract = attrs.get("frisk.copy_contract")
            if type(contract) is not dict:
                raise ValueError("Copy contract missing")
            path = contract.get("path")
            if path == "vector":
                if (contract.get("completion") != "synchronous" or contract.get("vector_bytes") not in (1, 16) or
                        attrs.get("frisk.writer_policy") != "first_owner"):
                    raise ValueError("unsupported copy contract")
            elif path in ("cp.async", "tma"):
                validate_async_copy(operation, values)
            else:
                raise ValueError("unsupported copy contract")
            copies.append({"operation": index, "path": path, "completion": contract["completion"]})
            if path == "tma":
                tma_operations.append(index)
    # Canonical root contracts establish physical allocation. Alias endpoints do
    # not reserve another array, but their absolute-root intervals must fit it.
    spaces, extents, root_alignment = {}, {}, {}
    for root in roots:
        storage = values[root]["storage"]
        spaces[root] = _space(storage)
        extents[root] = _extent(storage)
        alignment = _u64(storage.get("alignment"), "root alignment")
        if not alignment or alignment & (alignment - 1):
            raise ValueError("invalid Shared root alignment")
        root_alignment[root] = alignment
    for operation in data["operations"]:
        for operand in operation["operands"]:
            value = values[operand]
            if "storage" not in value or operation["name"] == "frisk.storage_contract":
                continue
            storage = value["storage"];root = _u64(storage.get("root"), "endpoint root identity")
            if root not in spaces or _space(storage) != spaces[root]:
                raise ValueError("storage endpoint has no matching declared root")
            _, upper = _extent(storage)
            if upper > extents[root][1]:
                raise ValueError("storage endpoint exceeds root footprint")
    used, alignment, padding, allocations = 0, 1, 0, []
    for root in roots:
        if spaces[root] == "Global":
            continue
        size = extents[root][0];align = root_alignment[root]
        alignment = max(alignment, align)
        if lifetimes is not None:
            continue
        end = reserve_shared(used, size, align, limit);start = end - size
        gap = start - used; padding += gap
        allocations.append(dict(root_id=root, offset=start, bytes=size, alignment=align,
                                padding=gap, lifetime="whole-kernel-declared"))
        used = end
    reuse_barriers = []
    if lifetimes is None and reuse_after is not None:
        raise ValueError('exact Shared reuse requires verified lifetimes')
    if lifetimes is not None:
        from execution_schedule import place_reusable_roots
        if set(lifetimes) != {r for r in roots if spaces[r] == 'Shared'}:
            raise ValueError('Shared lifetime plan does not cover allocated roots')
        allocations, reuse_barriers = place_reusable_roots(extents, root_alignment, lifetimes, limit,
                                                            reuse_after=reuse_after)
        used = max((p['offset'] + p['bytes'] for p in allocations), default=0)
        padding = sum(p['padding'] for p in allocations)
    root_bytes = used
    route_counts = {name: sum(plan["mode"] == name for _, plan in conversions) for name in ("Register", "Shuffle", "Shared")}
    _u64(epilogue_bytes, "epilogue size")
    scratch_max = max(epilogue_bytes, max((plan["scratch_bytes"] for _, plan in conversions), default=0))
    _u64(scratch_max, "scratch size")
    scratch_alignment = 16 if epilogue_bytes else 4
    scratch_offset = reserve_shared(used, 0, scratch_alignment, limit) if scratch_max else used
    static_bytes = reserve_shared(used, scratch_max, scratch_alignment, limit) if scratch_max else used
    alignment = max(alignment, scratch_alignment)
    tma_bytes = _u64(8 * len(tma_operations), "TMA barrier size")
    total = reserve_shared(static_bytes, tma_bytes, 8, limit) if tma_bytes else static_bytes
    barrier_offset = total - tma_bytes if tma_bytes else None
    return dict(schema_version=1, estimation_stage="before-native-compilation",
                semantics=("checked synchronous live intervals" if lifetimes is not None else
                           "declared-static-allocation; not SSA active peak or backend measurement"),
                **({"reuse_barriers": reuse_barriers} if lifetimes is not None else {}),
                declared_root_order=roots, global_roots=sorted(root for root in roots if spaces[root] == "Global"),
                allocations=allocations, shared_root_bytes=root_bytes,
                root_payload_bytes=sum(entry["bytes"] for entry in allocations), root_alignment_padding_bytes=padding,
                scratch_offset=scratch_offset, scratch_max_bytes=scratch_max,
                scratch_alignment_padding_bytes=scratch_offset-root_bytes,
                scratch_reuse="one serial segment, max across completed redistributions, not sum",
                redistribution_count=len(conversions), route_counts=route_counts,
                copy_completion_contracts=copies, static_shared_bytes=static_bytes, main_alignment=alignment,
                tma_barrier_offset=barrier_offset, tma_barrier_bytes=tma_bytes,
                tma_alignment_padding_bytes=(barrier_offset-static_bytes) if tma_bytes else 0,
                tma_barriers=[dict(operation=index,offset=barrier_offset+8*i,bytes=8,alignment=8)
                              for i,index in enumerate(tma_operations)],
                physical_shared_bytes=total, shared_limit_bytes=limit)
