import unittest

from attention_execution_audit import check_tma_issue


class AttentionExecutionAuditTest(unittest.TestCase):
    def test_equal_rank_k_and_v_are_bound_by_parameter_and_slot(self):
        ptx = "mov.b64 %rd8, frisk_execution_param_4;\ncvta.param.u64 %rd5, %rd8;\nsetp.eq.u32 p, tid, 0;\n"
        for dest, bar in [(1024,8192),(2048,8200)]:
            ptx += (f"add.u32 d{dest}, smem, {dest};\nadd.u32 b{bar}, smem, {bar};\n"
                    f"@p cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
                    f"[d{dest}], [%rd5, {{0, row, head}}], [b{bar}];\n")
        desc = dict(descriptor_arg=4, abi="value", box=[64,64,1], slots=[1024,2048])
        check_tma_issue(ptx,desc,[8192,8200])
        for invalid in (ptx.replace("%rd5, {0", "%rd6, {0"), ptx.replace("tid, 0;", "tid, 1;"),
                        ptx.replace("[b8200];","[b8192];")):
            with self.assertRaises(ValueError):
                check_tma_issue(invalid,desc,[8192,8200])
