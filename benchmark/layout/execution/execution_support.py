"""Checked scalar-H100 capability queries and compact execution-map adapters.

A positive assessment proves packet consistency and actual TTGIR emission only.
It does not claim native compilation, resource fit, numerical or GPU safety.
No global result cache or hardware-holder proof enumeration is introduced.
"""
from collections.abc import Sequence
import copy
import hashlib
import json
import math
from storage_addresses import normalize_storage_values

SUPPORT_CONTRACT_VERSION = "frisk-h100-scalar-ttgir-v1"
SYMBOLIC_VERSION = "gf2-canonical-v1"
MAX_LOGICAL = 65536


def _int(value):
    return type(value) is int


def _power(value):
    return _int(value) and value > 0 and not value & (value - 1)


class _ImageBasis:
    def __init__(self, columns, bits):
        self.pivots = [None] * bits
        self.rank = 0
        for index, column in enumerate(columns):
            value, owner = column, 1 << index
            for bit in range(bits - 1, -1, -1):
                if not value >> bit & 1:
                    continue
                previous = self.pivots[bit]
                if previous is None:
                    self.pivots[bit] = (value, owner)
                    self.rank += 1
                    break
                value ^= previous[0]
                owner ^= previous[1]

    def solve(self, value):
        owner = 0
        for bit in range(len(self.pivots) - 1, -1, -1):
            if value >> bit & 1:
                pivot = self.pivots[bit]
                if pivot is None:
                    raise ValueError("logical value is outside the source image")
                value ^= pivot[0]
                owner ^= pivot[1]
        return owner


class SymbolicMap:
    """Immutable checked canonical map; logical slots remain row-major."""
    def __init__(self, distribution):
        symbolic = distribution.get("symbolic")
        if type(symbolic) is not dict or set(symbolic) != {
                "version", "hardware_bits", "logical_bits", "hardware_widths",
                "logical_widths", "rows_hex"} or symbolic.get("version") != SYMBOLIC_VERSION:
            raise ValueError("invalid compact execution-map schema/version")
        self.topology = tuple(distribution.get("topology", ()))
        self.shape = tuple(distribution.get("shape", ()))
        if len(self.topology) != 5 or any(not _power(x) for x in self.topology) or self.topology[1] > 32:
            raise ValueError("invalid compact hardware topology")
        if any(not _power(x) for x in self.shape):
            raise ValueError("invalid compact logical shape")
        self.hardware_widths = tuple(x.bit_length()-1 for x in self.topology)
        self.logical_widths = tuple(x.bit_length()-1 for x in self.shape)
        self.hardware_bits = sum(self.hardware_widths)
        self.logical_bits = sum(self.logical_widths)
        if self.hardware_bits > 256 or self.logical_bits > 16:
            raise ValueError("compact algebra/logical domain limit exceeded")
        if (symbolic["hardware_widths"] != list(self.hardware_widths) or
                symbolic["logical_widths"] != list(self.logical_widths) or
                type(symbolic["hardware_bits"]) is not int or symbolic["hardware_bits"] != self.hardware_bits or
                type(symbolic["logical_bits"]) is not int or symbolic["logical_bits"] != self.logical_bits):
            raise ValueError("compact map widths do not match topology/shape")
        rows = symbolic["rows_hex"]
        if type(rows) is not list or len(rows) != self.logical_bits:
            raise ValueError("invalid compact row count")
        self.rows = []
        for row in rows:
            if type(row) is not str or not row or any(c not in "0123456789abcdefABCDEF" for c in row):
                raise ValueError("invalid compact hexadecimal row")
            value = int(row, 16)
            if value.bit_length() > self.hardware_bits:
                raise ValueError("compact row exceeds hardware bit width")
            self.rows.append(value)
        self.rows = tuple(self.rows)
        self.volume = 1 << self.logical_bits
        self.holders = 1 << self.hardware_bits
        self.columns = tuple(sum(((row >> bit) & 1) << r for r, row in enumerate(self.rows))
                             for bit in range(self.hardware_bits))
        self.owners = _ImageBasis(self.columns, self.logical_bits)
        if self.owners.rank != self.logical_bits:
            raise ValueError("compact map does not cover logical domain")
        replication = distribution.get("replication")
        if not _int(replication) or replication <= 0 or replication != self.holders // self.volume:
            raise ValueError("compact replication does not match ownership")
        # Conversion verification can solve a local owner without visiting any
        # other thread, even for a large replicated source register axis.
        self.registers = _ImageBasis(self.columns[:self.hardware_widths[0]], self.logical_bits)
        # Row-major flattening is a bit permutation for power-of-two shapes.
        # Precompute small byte tables without enumerating hardware holders.
        flat_columns = tuple(self.flatten(column) for column in self.columns)
        tables = []
        for start in range(0, self.hardware_bits, 8):
            table = [0]
            for column in flat_columns[start:start+8]:
                table += [value ^ column for value in table]
            tables.append(tuple(table))
        self._flat_byte_tables = tuple(tables)
        self._flat_dependencies = (self.rows, self.shape, self.logical_widths, self.holders)

    def packed(self, hardware):
        if not _int(hardware) or not 0 <= hardware < self.holders:
            raise IndexError("hardware holder outside compact map")
        return sum(((row & hardware).bit_count() & 1) << bit for bit, row in enumerate(self.rows))

    def flatten(self, packed):
        result, offset = 0, 0
        for extent, width in zip(self.shape, self.logical_widths):
            result = result * extent + ((packed >> offset) & ((1 << width)-1))
            offset += width
        return result

    def pack_slot(self, slot):
        if not _int(slot) or not 0 <= slot < self.volume:
            raise IndexError("logical slot outside compact map")
        coordinates = [0] * len(self.shape)
        for axis in range(len(self.shape)-1, -1, -1):
            coordinates[axis] = slot % self.shape[axis]
            slot //= self.shape[axis]
        result, offset = 0, 0
        for coordinate, width in zip(coordinates, self.logical_widths):
            result |= coordinate << offset
            offset += width
        return result

    def logical(self, hardware):
        rows, shape, widths, holders = self._flat_dependencies
        if (type(self) is not SymbolicMap or self.rows is not rows or
                self.shape is not shape or self.logical_widths is not widths or
                self.holders != holders or "packed" in self.__dict__ or "flatten" in self.__dict__):
            return self.flatten(self.packed(hardware))
        if not _int(hardware) or not 0 <= hardware < self.holders:
            raise IndexError("hardware holder outside compact map")
        value = 0
        for byte, table in enumerate(self._flat_byte_tables):
            value ^= table[(hardware >> (8*byte)) & 255]
        return value

    def first_table(self):
        """Expand the existing canonical linear section in O(logical volume)."""
        result = [0]
        for bit in range(self.logical_bits):
            column = self.first(1 << bit)
            result += [value ^ column for value in result]
        return result

    def first(self, slot):
        return self.owners.solve(self.pack_slot(slot))

    def local_owner(self, slot, thread):
        fixed = thread * self.topology[0]
        residual = self.pack_slot(slot) ^ self.packed(fixed)
        return self.registers.solve(residual)


class LogicalTable(Sequence):
    """Read-only table interface; constructing it never expands holder slots."""
    def __init__(self, mapping):
        self.mapping = mapping

    def __len__(self):
        # Python's Sequence protocol needs a Py_ssize_t length. The declared
        # backend profile rejects wider topologies before materialization.
        if self.mapping.holders > 2**63-1:
            raise ValueError("compact map exceeds table-consumer index width")
        return self.mapping.holders

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.mapping.logical(i) for i in range(*index.indices(len(self)))]
        if not _int(index):
            raise TypeError("logical table index must be integer")
        if index < 0:
            index += len(self)
        return self.mapping.logical(index)

    def __eq__(self, other):
        return isinstance(other, Sequence) and len(self) == len(other) and all(a == b for a, b in zip(self, other))


def normalize_distribution(distribution, *, materialize=False):
    if "symbolic" not in distribution:
        return distribution
    if isinstance(distribution.get("logical"), LogicalTable):
        return distribution
    result = copy.deepcopy(distribution)
    mapping = SymbolicMap(result)
    table = LogicalTable(mapping)
    result["logical"] = list(table) if materialize else table
    result["first"] = mapping.first_table()
    return result


def normalize_packet(data, *, materialize=False):
    """Adapt schema2 maps to schema1 consumers; no proof budget is enlarged.

    The caller should retain the original JSON bytes/hash as execution identity.
    The returned holder table is lazy by default, first-owner is logical-sized.
    """
    if type(data) is not dict or type(data.get("schema_version")) is not int or data["schema_version"] not in (1, 2):
        raise ValueError("unsupported execution schema")
    if set(data) - {"schema_version", "values", "operations", "shared_limit_bytes"}:
        raise ValueError("unknown execution packet fields")
    if type(data.get("values")) is not list or type(data.get("operations")) is not list:
        raise ValueError("execution values and operations must be lists")
    values = normalize_storage_values(data["values"], schema_version=data["schema_version"], materialize=materialize)
    if values is data["values"] and data["schema_version"] == 1 and all("symbolic" not in v.get("distributed", {}) or isinstance(v["distributed"].get("logical"), LogicalTable) for v in data["values"] if isinstance(v, dict)):
        return data
    result = dict(data)
    result["schema_version"] = 1
    result["values"] = [dict(v, distributed=normalize_distribution(v["distributed"], materialize=materialize))
                        if isinstance(v, dict) and "distributed" in v else copy.deepcopy(v) for v in values]
    result["operations"] = []
    for operation in data["operations"]:
        op = dict(operation)
        if "execution" in op:
            op["execution"] = normalize_distribution(op["execution"], materialize=materialize)
        result["operations"].append(op)
    return result


def _explicit_distribution(distribution):
    topology, shape = distribution.get("topology"), distribution.get("shape")
    if type(topology) is not list or len(topology)!=5 or any(not _power(x) for x in topology) or topology[1]>32:
        raise ValueError("invalid explicit hardware topology")
    if type(shape) is not list or any(not _power(x) for x in shape):
        raise ValueError("invalid explicit logical shape")
    holders, volume = math.prod(topology), math.prod(shape)
    if holders>65536 or volume>65536:
        raise ValueError("explicit execution table exceeds original bounded domain")
    logical, first = distribution.get("logical"), distribution.get("first")
    if type(logical) is not list or len(logical)!=holders or type(first) is not list or len(first)!=volume:
        raise ValueError("invalid explicit logical/first table size")
    expected, counts = [None]*volume, [0]*volume
    for hardware, slot in enumerate(logical):
        if not _int(slot) or not 0<=slot<volume:
            raise ValueError("explicit logical coordinate out of bounds")
        counts[slot]+=1
        if expected[slot] is None:expected[slot]=hardware
    if first!=expected or any(not _int(x) for x in first) or any(count==0 for count in counts):
        raise ValueError("explicit first-owner/coverage mismatch")
    if "replication" in distribution:
        replication = distribution["replication"]
        if not _int(replication) or replication<=0 or any(count!=replication for count in counts):
            raise ValueError("explicit replication mismatch")


def _envelope(data):
    if type(data) is not dict or type(data.get("schema_version")) is not int or data["schema_version"] not in (1,2):
        raise ValueError("unsupported execution schema")
    if set(data) - {"schema_version","values","operations","shared_limit_bytes"}:
        raise ValueError("unknown execution packet fields")
    if type(data.get("values")) is not list or type(data.get("operations")) is not list:
        raise ValueError("values and operations must be lists")
    ids = set()
    for value in data["values"]:
        if type(value) is not dict or not _int(value.get("id")) or value["id"] < 0 or value["id"] in ids or type(value.get("type")) is not str:
            raise ValueError("invalid or duplicate execution value")
        ids.add(value["id"])
        distribution = value.get("distributed")
        if distribution is not None:
            if type(distribution) is not dict:
                raise ValueError("invalid distribution")
            if "symbolic" in distribution:
                if data["schema_version"] != 2:
                    raise ValueError("compact map requires schema2")
                SymbolicMap(distribution)
            else:
                _explicit_distribution(distribution)
    modules, kernels, parallels = [], [], []
    definitions = set()
    for operation in data["operations"]:
        if type(operation) is not dict or type(operation.get("name")) is not str or type(operation.get("attrs")) is not dict:
            raise ValueError("invalid execution operation")
        if "execution" in operation:
            execution=operation["execution"]
            if type(execution) is not dict:raise ValueError("invalid operation execution map")
            if "symbolic" in execution:
                if data["schema_version"]!=2:raise ValueError("compact map requires schema2")
                SymbolicMap(execution)
            else:_explicit_distribution(execution)
        for field in ("operands","results"):
            if type(operation.get(field)) is not list or any(not _int(i) or i not in ids for i in operation[field]):
                raise ValueError("invalid execution operand/result reference")
        for identity in operation["results"]:
            if identity in definitions:
                raise ValueError("duplicate SSA result definition")
            definitions.add(identity)
        if operation["name"] == "builtin.module":modules.append(operation)
        if operation["name"] == "frisk.kernel":kernels.append(operation)
        if operation["name"] == "frisk.parallel":parallels.append(operation)
    if len(modules)!=1 or len(kernels)!=1 or len(parallels)!=1:
        raise ValueError("requires one module/kernel/parallel execution envelope")
    if modules[0]["attrs"].get("frisk.target")!="sm_90a" or parallels[0]["attrs"].get("threads")!=128 or parallels[0]["attrs"].get("ranges")!=[1]:
        raise ValueError("requires one sm_90a 128-thread single-CTA execution")


def assess(data, *, include_emission=False):
    result = dict(schema_version=1, support_contract_version=SUPPORT_CONTRACT_VERSION,
                  claim="checked-emission-only", supported=False,
                  status="invalid", stage="export-envelope", code="invalid-packet")
    try:
        # Snapshot caller-owned JSON before any validation or emitter query.
        raw = json.dumps(data, sort_keys=True, separators=(",",":"), allow_nan=False)
        snapshot = json.loads(raw)
        result["packet_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
        _envelope(snapshot)
        normalized = normalize_packet(snapshot)
    except (ValueError,TypeError,KeyError,IndexError,OverflowError) as error:
        result["reason"] = str(error)
        return result
    try:
        from backend import emit
        text, info = emit(normalized)
    except ValueError as error:
        result.update(status="unsupported",stage="backend-emission",code="backend-rejection",reason=str(error))
        return result
    except (KeyError,IndexError,TypeError,AssertionError,OverflowError,ZeroDivisionError) as error:
        result.update(status="invalid",stage="backend-emission",code="malformed-emission-input",reason=type(error).__name__+": "+str(error))
        return result
    except Exception as error:
        result.update(status="error",stage="backend-emission",code="infrastructure-error",reason=type(error).__name__+": "+str(error))
        return result
    # The emitter reports total SM90 Shared capacity. Its current inline-PTX
    # allocation is static, so compiler support additionally requires <=48 KiB.
    from native_static_support import check_static_shared, StaticSharedUnsupported
    try:
        result["static_shared_domain"] = check_static_shared(info)
    except StaticSharedUnsupported as error:
        result.update(status="unsupported",stage="backend-static-support",code="static-shared-limit",reason=str(error))
        return result
    except ValueError as error:
        result.update(status="invalid",stage="backend-static-support",code="invalid-emitted-config",reason=str(error))
        return result
    result.update(status="supported",supported=True,stage="checked-emission",code="supported",
                  reason="actual checked emitter accepted this exact packet within the current static Shared domain")
    if include_emission:
        result.update(ttgir=text,execution_config=info)
    return result


def assess_file(path, *, include_emission=False):
    try:
        with open(path) as source:
            data = json.load(source)
    except (OSError,ValueError) as error:
        return dict(schema_version=1,support_contract_version=SUPPORT_CONTRACT_VERSION,
                    claim="checked-emission-only",supported=False,status="invalid",
                    stage="export-input",code="unreadable-packet",reason=str(error))
    return assess(data,include_emission=include_emission)
