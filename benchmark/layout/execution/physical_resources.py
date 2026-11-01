"""Query the loaded CUDA function: actual allocation and theoretical residency.

No context is created and no occupancy value is inferred from logical SSA slots.
The driver API accounts for architecture-specific allocation granularity. Its
occupancy result is a resource ceiling, not measured active warps or throughput.
"""
import ctypes
import math


class ResourceQueryError(RuntimeError):
    pass


FUNCTION_ATTRIBUTES = {
    "max_threads_per_block": 0,
    "shared_bytes": 1,
    "local_bytes_per_thread": 3,
    "registers_per_thread": 4,
    "max_dynamic_shared_bytes": 8,
}
DEVICE_ATTRIBUTES = {
    "max_threads_per_block": 1,
    "warp_size": 10,
    "multiprocessor_count": 16,
    "max_threads_per_sm": 39,
    "max_shared_bytes_per_sm": 81,
    "max_register32_per_sm": 82,
    "max_blocks_per_sm": 106,
    "reserved_shared_bytes_per_block": 111,
}


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum or value > 2**31 - 1:
        raise ValueError(name + " must be a bounded integer >= " + str(minimum))
    return value


def _call(cuda, name, *args):
    function = getattr(cuda, name, None)
    if function is None:
        raise ResourceQueryError("CUDA driver does not export " + name)
    status = function(*args)
    status = status.value if isinstance(status, ctypes.c_int) else status
    if type(status) is not int or status != 0:
        raise ResourceQueryError(name + " failed with CUDA status " + str(status))


def query_physical_resources(cuda, function, threads=128, dynamic_shared_bytes=0):
    """Query an already-loaded CUfunction in the caller's current context.

    `threads` is the block's total thread count. The supplied dynamic allocation
    must match the launch; it is not silently replaced by the kernel's maximum.
    Invalid queries and driver failures raise instead of producing zero data.
    """
    _integer(threads, "threads", 1)
    _integer(dynamic_shared_bytes, "dynamic_shared_bytes")
    if (isinstance(function, bool) or not isinstance(function, (int, ctypes.c_void_p)) or
            (type(function) is int and not 0 < function < 2 ** (8 * ctypes.sizeof(ctypes.c_void_p)))):
        raise ValueError("a non-null loaded CUfunction is required")
    pointer = function if isinstance(function, ctypes.c_void_p) else ctypes.c_void_p(function)
    if not pointer.value:
        raise ValueError("a non-null loaded CUfunction is required")
    device = ctypes.c_int()
    _call(cuda, "cuCtxGetDevice", ctypes.byref(device))
    if device.value < 0:
        raise ResourceQueryError("negative current device ordinal")
    attributes = {}
    for name, attribute in FUNCTION_ATTRIBUTES.items():
        result = ctypes.c_int()
        _call(cuda, "cuFuncGetAttribute", ctypes.byref(result), attribute, pointer)
        if result.value < 0:
            raise ResourceQueryError("negative function attribute " + name)
        attributes[name] = result.value
    limits = {}
    for name, attribute in DEVICE_ATTRIBUTES.items():
        result = ctypes.c_int()
        _call(cuda, "cuDeviceGetAttribute", ctypes.byref(result), attribute, device)
        if result.value < 0 or (result.value == 0 and name != "reserved_shared_bytes_per_block"):
            raise ResourceQueryError("invalid device limit " + name)
        limits[name] = result.value
    if threads > min(attributes["max_threads_per_block"], limits["max_threads_per_block"]):
        raise ValueError("thread count exceeds this function/device launch limit")
    if dynamic_shared_bytes > attributes["max_dynamic_shared_bytes"]:
        raise ValueError("dynamic shared allocation exceeds this function's configured limit")
    blocks = ctypes.c_int()
    _call(cuda, "cuOccupancyMaxActiveBlocksPerMultiprocessor", ctypes.byref(blocks),
          pointer, ctypes.c_int(threads), ctypes.c_size_t(dynamic_shared_bytes))
    warps_per_block = math.ceil(threads / limits["warp_size"])
    max_warps = limits["max_threads_per_sm"] // limits["warp_size"]
    active_warps = blocks.value * warps_per_block
    if not 0 <= blocks.value <= limits["max_blocks_per_sm"] or not 0 <= active_warps <= max_warps:
        raise ResourceQueryError("inconsistent driver occupancy result")
    return {
        "schema_version": 1,
        "source": "CUDA-driver-loaded-function",
        "device_ordinal": device.value,
        "threads_per_block": threads,
        "dynamic_shared_bytes": dynamic_shared_bytes,
        "function_attributes": attributes,
        "device_limits": limits,
        "registers_per_thread": attributes["registers_per_thread"],
        "shared_bytes": attributes["shared_bytes"],
        "local_bytes": attributes["local_bytes_per_thread"],
        "total_shared_bytes_per_block": attributes["shared_bytes"] + dynamic_shared_bytes,
        "residency": {
            "kind": "theoretical-resource-ceiling",
            "api": "cuOccupancyMaxActiveBlocksPerMultiprocessor",
            "active_blocks_per_sm": blocks.value,
            "warps_per_block": warps_per_block,
            "active_warps_per_sm": active_warps,
            "max_warps_per_sm": max_warps,
            "fraction": active_warps / max_warps,
            "launchable_by_resources": blocks.value > 0,
        },
        "measured_occupancy": None,
        "local_memory_is_not_a_spill_measurement": True,
    }


def compare_logical_resources(ir_resources, physical, emitter_info=None):
    """Keep every SSA scope separate and preserve unavailable information.

    Slot/reg differences are comparisons of proxies, not register-allocation
    errors: logical carriers omit address, descriptor and execution temporaries.
    Shared active intervals also need not equal allocated static capacity.
    """
    if type(ir_resources) is not dict or type(ir_resources.get("scopes")) is not list:
        raise ValueError("IR resource diagnostics with scopes are required")
    for key in ("registers_per_thread", "shared_bytes", "local_bytes"):
        _integer(physical.get(key), key)
    rows = []
    for scope in ir_resources["scopes"]:
        if type(scope) is not dict or type(scope.get("scope")) is not str:
            raise ValueError("malformed IR resource scope")
        row = {"scope": scope["scope"], "status": scope.get("status")}
        if scope.get("status") != "available":
            row["unavailable_operations"] = scope.get("unavailable_operations", [])
            rows.append(row)
            continue
        keys = ("logical_live_carrier_slots_per_thread", "logical_live_carrier_bytes_per_thread",
                "logical_live_shared_root_bytes", "declared_shared_root_bytes")
        for key in keys:
            value = scope.get(key)
            if type(value) is not int or value < 0:
                raise ValueError("available scope has invalid " + key)
            row[key] = value
        if type(scope.get("saturated")) is not bool:
            raise ValueError("available scope must declare saturation")
        row["saturated"] = scope["saturated"]
        if not row["saturated"]:
            row["physical_minus_logical_slots"] = (physical["registers_per_thread"] -
                row["logical_live_carrier_slots_per_thread"])
            row["physical_shared_minus_declared_roots"] = (physical["shared_bytes"] -
                row["declared_shared_root_bytes"])
        rows.append(row)
    result = {"schema_version": 1, "scopes": rows,
              "comparison_kind": "logical-proxies-versus-physical-allocation",
              "logical_slots_are_not_physical_registers": True,
              "scope_peaks_are_not_summed": True}
    if emitter_info is not None:
        expected = emitter_info.get("physical_shared_bytes")
        if type(expected) is not int or expected < 0:
            raise ValueError("emitter physical shared allocation is required")
        result["emitter_shared_bytes"] = expected
        result["driver_shared_matches_emitter"] = physical["shared_bytes"] == expected
    return result
