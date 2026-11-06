import ctypes
import unittest
from physical_resources import (query_physical_resources, compare_logical_resources,
                                ResourceQueryError, DEVICE_ATTRIBUTES, FUNCTION_ATTRIBUTES)


class Driver:
    def __init__(self):
        self.failure = None
        self.blocks = 8
        self.shared = 6144
        self.functions = {0: 1024, 1: self.shared, 3: 0, 4: 64, 8: 4096}
        self.devices = {1: 1024, 10: 32, 16: 114, 39: 2048, 81: 233472, 82: 65536, 106: 32, 111: 1024}
        self.occupancy_args = None
    def cuCtxGetDevice(self, result):
        result._obj.value = 1
        return 201 if self.failure == "context" else 0
    def cuFuncGetAttribute(self, result, attribute, function):
        result._obj.value = self.functions[attribute]
        return 400 if self.failure == "function" else 0
    def cuDeviceGetAttribute(self, result, attribute, device):
        result._obj.value = self.devices[attribute]
        return 101 if self.failure == "device" else 0
    def cuOccupancyMaxActiveBlocksPerMultiprocessor(self, result, function, threads, dynamic):
        self.occupancy_args = (threads.value, dynamic.value)
        result._obj.value = self.blocks
        return 1 if self.failure == "occupancy" else 0


class ResourcesTest(unittest.TestCase):
    def test_launch_specific_driver_query(self):
        driver = Driver()
        row = query_physical_resources(driver, 123, threads=96, dynamic_shared_bytes=128)
        self.assertEqual(driver.occupancy_args, (96, 128))
        self.assertEqual(row["residency"]["active_warps_per_sm"], 24)
        self.assertEqual(row["residency"]["fraction"], 24 / 64)
        self.assertEqual(row["total_shared_bytes_per_block"], 6272)
        self.assertIsNone(row["measured_occupancy"])
    def test_partial_warp_consumes_a_whole_warp(self):
        row = query_physical_resources(Driver(), ctypes.c_void_p(123), threads=33)
        self.assertEqual(row["residency"]["warps_per_block"], 2)
    def test_zero_residency_is_not_silently_launchable(self):
        driver = Driver(); driver.blocks = 0
        self.assertFalse(query_physical_resources(driver, 123)["residency"]["launchable_by_resources"])
    def test_each_driver_failure_is_fatal(self):
        for failure in ("context", "function", "device", "occupancy"):
            driver = Driver(); driver.failure = failure
            with self.subTest(failure=failure), self.assertRaises(ResourceQueryError):
                query_physical_resources(driver, 123)
    def test_malformed_handles_and_launch_sizes(self):
        for kwargs in ({"function": 0}, {"function": -1}, {"function": 2**128},
                       {"function": True}, {"function": "123"},
                       {"function": 1, "threads": True}, {"function": 1, "threads": 0},
                       {"function": 1, "threads": 2048},
                       {"function": 1, "dynamic_shared_bytes": -1},
                       {"function": 1, "dynamic_shared_bytes": 4097}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                query_physical_resources(Driver(), **kwargs)
    def test_inconsistent_driver_data_rejected(self):
        driver = Driver(); driver.blocks = 33
        with self.assertRaises(ResourceQueryError): query_physical_resources(driver, 123)
        driver = Driver(); driver.devices[10] = 0
        with self.assertRaises(ResourceQueryError): query_physical_resources(driver, 123)
        driver = Driver(); driver.functions[4] = -1
        with self.assertRaises(ResourceQueryError): query_physical_resources(driver, 123)
    def test_logical_scopes_unavailable_and_capacity_are_distinct(self):
        scope = dict(scope="kernel@f", status="available", saturated=False,
                     logical_live_carrier_slots_per_thread=106,
                     logical_live_carrier_bytes_per_thread=424,
                     logical_live_shared_root_bytes=4096, declared_shared_root_bytes=6144)
        row = compare_logical_resources({"scopes": [scope, dict(scope="func@g", status="unavailable")]},
                                       dict(registers_per_thread=64, shared_bytes=6168, local_bytes=0),
                                       dict(physical_shared_bytes=6168))
        self.assertEqual(row["scopes"][0]["physical_minus_logical_slots"], -42)
        self.assertEqual(row["scopes"][0]["physical_shared_minus_declared_roots"], 24)
        self.assertNotIn("physical_minus_logical_slots", row["scopes"][1])
        self.assertTrue(row["driver_shared_matches_emitter"])
    def test_saturated_scopes_do_not_produce_error_numbers(self):
        scope = dict(scope="kernel@f", status="available", saturated=True,
                     logical_live_carrier_slots_per_thread=2**64-1,
                     logical_live_carrier_bytes_per_thread=2**64-1,
                     logical_live_shared_root_bytes=0, declared_shared_root_bytes=0)
        row = compare_logical_resources({"scopes": [scope]},
                                       dict(registers_per_thread=64, shared_bytes=0, local_bytes=0))
        self.assertNotIn("physical_minus_logical_slots", row["scopes"][0])


if __name__ == "__main__": unittest.main()
