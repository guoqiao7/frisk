"""Capability bound of the current explicit static-PTX Shared emitter.

This is an emitter/toolchain limit, not the device's total Shared capacity or
its opt-in dynamic Shared limit. Keep the full emitted configuration unchanged.
"""

STATIC_SHARED_CONTRACT = "frisk-static-ptx-shared-v1"
STATIC_SHARED_LIMIT_BYTES = 48 * 1024


class StaticSharedUnsupported(ValueError):
    """A valid allocation exceeds the current static-PTX support domain."""

    def __init__(self, actual_bytes):
        self.static_shared_bytes = actual_bytes
        self.static_shared_limit_bytes = STATIC_SHARED_LIMIT_BYTES
        super().__init__(
            "current static PTX backend unsupported: static_shared_bytes="
            + str(actual_bytes) + " exceeds static Shared limit "
            + str(STATIC_SHARED_LIMIT_BYTES)
            + " bytes; device total/opt-in dynamic Shared capacity is a separate limit")


def check_static_shared(info):
    """Validate an emission's static allocation and return its bounded proof.

    Missing or malformed allocation data is invalid, never implicit zero.
    StaticSharedUnsupported distinguishes an otherwise valid out-of-domain
    allocation from malformed configuration. No caller-owned data is changed.
    """
    if type(info) is not dict:
        raise ValueError("emitted configuration must be an object")
    if "static_shared_bytes" not in info:
        raise ValueError("emitted configuration missing static_shared_bytes")
    actual = info["static_shared_bytes"]
    if type(actual) is not int or actual < 0:
        raise ValueError("static_shared_bytes must be a nonnegative integer")
    if actual > STATIC_SHARED_LIMIT_BYTES:
        raise StaticSharedUnsupported(actual)
    return dict(schema_version=1, contract=STATIC_SHARED_CONTRACT,
                allocation_kind="static-ptx", supported=True,
                static_shared_bytes=actual,
                static_shared_limit_bytes=STATIC_SHARED_LIMIT_BYTES)
