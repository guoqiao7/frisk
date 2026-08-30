import os

import lit.formats

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

config.name = "FRISK"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.frisk_obj_root, "test")
config.excludes = ["CMakeLists.txt", "lit.cfg.py", "lit.site.cfg.py"]

llvm_config.use_default_substitutions()
llvm_config.add_tool_substitutions(
    [ToolSubst("frisk-opt", unresolved="fatal")],
    [config.frisk_tools_dir, config.llvm_tools_dir],
)
