#!/usr/bin/env python3
"""Export audited, enumerated pinned TileLang semantics (no Frisk imports).

The parent validates sources and launches an isolated child. Only the child
imports native reference modules. Canonical output is replaced after success;
audit paths and native-library hashes go to a separate JSON log.
"""
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REVISION = "5e149e31674658f94779c7d0c6039549a1853123"
DEPENDENCIES = {
    "3rdparty/tvm": "907a88c8791ccf33b9874821bc875e7abf624367",
    "3rdparty/cutlass": "b2dd65dc864e09688245b316ac46c4a6cd07e15c",
}
PYTHON_DEPENDENCIES = {
    "apache-tvm-ffi": "0.1.12", "torch": "2.8.0", "torch-c-dlpack-ext": "0.1.5",
    "cloudpickle": "3.1.2", "ml-dtypes": "0.5.3", "numpy": "2.2.6",
    "psutil": "7.2.2", "tqdm": "4.67.1", "typing-extensions": "4.15.0",
    "z3-solver": "4.15.4.0",
}


def verify_python_versions(actual):
    for name, expected in PYTHON_DEPENDENCIES.items():
        if actual.get(name) != expected:
            raise ValueError(f"Python dependency revision mismatch: {name}: {actual.get(name)} != {expected}")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def verify_checkout(root, revision):
    if revision != REVISION:
        raise ValueError(f"unsupported TileLang revision: {revision}")
    actual = git(root, "rev-parse", "HEAD")
    if actual != REVISION:
        raise ValueError(f"TileLang revision mismatch: {actual} != {REVISION}")
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("dirty pinned TileLang checkout")
    dependencies = {}
    for line in git(root, "submodule", "status", "--recursive").splitlines():
        # git() strips the leading blank only on the first line.
        if line[0] in "+-U":
            raise ValueError(f"uninitialized or mismatched dependency: {line}")
        commit, path, *_ = line.split()
        dependencies[path] = commit
        if git(root / path, "status", "--porcelain", "--untracked-files=no"):
            raise ValueError(f"dirty dependency: {path}")
    for path, commit in DEPENDENCIES.items():
        if dependencies.get(path) != commit:
            raise ValueError(f"dependency revision mismatch: {path}")
    return dependencies


def require_inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"module/native library outside pinned environment: {path} (expected {root})")
    return str(path)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest(root):
    digest = hashlib.sha256()
    paths = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"]).split(b"\0")
    for name in sorted(p for p in paths if p):
        path = root / os.fsdecode(name)
        if path.is_file():
            digest.update(name + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def native_audit(root):
    import tilelang
    import tvm
    import tvm_ffi
    if sys.prefix == sys.base_prefix:
        raise ValueError("reference exporter requires an isolated Python venv")
    modules = {
        "tilelang": require_inside(tilelang.__file__, root / "tilelang"),
        "tvm": require_inside(tvm.__file__, root / "3rdparty/tvm/python/tvm"),
        "tvm_ffi": require_inside(tvm_ffi.__file__, Path(sys.prefix)),
    }
    python_versions = {name: importlib.metadata.version(name) for name in PYTHON_DEPENDENCIES}
    verify_python_versions(python_versions)
    libraries = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = line.split()[-1]
        name = Path(path).name
        if path.startswith("/") and ".so" in name and any(s in name for s in ("tilelang", "tvm", "core.cpython")):
            allowed = Path(sys.prefix) if "tvm_ffi" in path else root
            checked = require_inside(path, allowed)
            libraries[checked] = digest_file(checked)
    for needed in ("libtilelang.so", "libtvm_compiler.so", "libtvm_ffi.so"):
        if not any(Path(p).name == needed for p in libraries):
            raise ValueError(f"required native library not loaded: {needed}")
    return {"python": sys.executable, "modules": modules, "native_libraries": libraries,
            "python_dependencies": python_versions, "tracked_checkout_status": "clean",
            "tvm_ffi_version": "0.1.12", "tilelang_commit": REVISION}


def export_cases():
    import tilelang as tl
    import tilelang.language as T
    from tilelang import tvm
    from tilelang.layout import Layout, Fragment, PartialFragment
    from tilelang.layout import (make_linear_layout, make_quarter_bank_swizzled_layout,
                                 make_half_bank_swizzled_layout, make_full_bank_swizzled_layout,
                                 make_fully_replicated_layout_fragment)
    from tilelang.cuda.intrinsics.macro.wgmma_macro_generator import TensorCoreIntrinEmitter

    analyzer = tvm.arith.Analyzer()
    cases, sources = [], {}

    def integer(expr, bindings):
        result = analyzer.simplify(tvm.tirx.stmt_functor.substitute(expr, bindings))
        if not isinstance(result, tvm.tirx.IntImm):
            raise ValueError(f"nonconstant native enumeration: {result}")
        return int(result)

    def base(name, operation, shape, dtype, parameters=None, reject=None, valid_shape=None):
        return {"name": name, "operation": operation, "shape": list(shape),
                "valid_shape": list(valid_shape or shape), "dtype": dtype,
                "carrier_extents": [1, 1, 1, 1, 1], "storage_bytes": 0,
                "value_semantics": "complete", "writer_policy": "none",
                "parameters": parameters or {}, "distributed_points": [], "storage_points": [],
                "writer_owners": [], "frisk_expectation": {
                    "kind": "reject" if reject else "equivalent",
                    "diagnostic": reject[0] if reject else "", "reason": reject[1] if reject else ""}}

    def storage(case, layout):
        variables = list(layout.get_forward_vars())
        expr = layout.get_linearized_forward_index()
        bits = {"f16": 16, "bf16": 16, "f32": 32, "i8": 8}[case["dtype"]]
        for logical in itertools.product(*(range(s) for s in case["valid_shape"])):
            offset = integer(expr, dict(zip(variables, logical))) * bits
            case["storage_points"].append({"logical": list(logical), "byte_offset": offset // 8, "bit_offset": offset % 8})
        case["storage_bytes"] = max(p["byte_offset"] + bits // 8 for p in case["storage_points"])

    def fragment(case, layout):
        if list(map(int, layout.get_input_shape())) != case["shape"]:
            raise ValueError(f"native fragment shape mismatch: {case['name']}")
        replicas = int(layout.replicate_size)
        variables = list(layout.get_forward_vars())
        # Native FragmentNode::GetForwardVars prepends the replication
        # placeholder iff its extent exceeds one (layout.cc:1471).
        extra, variables = (variables[:1], variables[1:]) if replicas > 1 else ([], variables)
        if len(variables) != len(case["shape"]):
            raise ValueError("unexpected native Fragment variable rank")
        thread_expr, register_expr = layout.thread, layout.get_linearized_forward_index()
        partial = isinstance(layout, PartialFragment)
        combine = int(layout.combine_size) if partial else 1
        case["value_semantics"] = "partial" if partial else "complete"
        for logical in itertools.product(*(range(s) for s in case["shape"])):
            for replica in range(replicas):
                bindings = dict(zip(variables, logical))
                bindings.update({v: replica for v in extra})
                try:
                    thread, register = integer(thread_expr, bindings), integer(register_expr, bindings)
                except ValueError as error:
                    raise ValueError(f"{case['name']}: vars={variables}; extra={extra}; thread={thread_expr}; register={register_expr}; bindings={bindings}") from error
                carrier = [register, thread % 32, (thread // 32) % 4, thread // 128, 0]
                case["distributed_points"].append({"carrier": carrier, "logical": list(logical),
                    "copy_group": replica // combine if partial else 0,
                    "addend": replica % combine if partial else -1,
                    "valid": all(i < s for i, s in zip(logical, case["valid_shape"]))})
        case["distributed_points"].sort(key=lambda p: p["carrier"])
        case["carrier_extents"] = [max(p["carrier"][i] for p in case["distributed_points"]) + 1 for i in range(5)]
        if not partial:
            # Policy compatibility, NOT an upstream writer-election claim.
            case["writer_policy"] = "compatible_first_owner"
            owners = {}
            for p in case["distributed_points"]:
                if p["valid"]:
                    key = tuple(p["logical"])
                    if key not in owners or p["carrier"][::-1] < owners[key][::-1]:
                        owners[key] = p["carrier"]
            case["writer_owners"] = sorted(owners.values())

    def finish(case, source):
        sources[case["name"]] = source
        cases.append(case)

    # Production WGMMA emitter: no local transcription of its layout formula.
    for dtype, native_dtype in (("f16", "float16"), ("bf16", "bfloat16")):
        emitter = TensorCoreIntrinEmitter(a_dtype=native_dtype, b_dtype=native_dtype, accum_dtype="float32",
            block_row_warps=4, block_col_warps=1, warp_row_tiles=16, warp_col_tiles=64, chunk=16)
        for operand, shape, value_dtype in ((1, [64, 16], dtype), (0, [64, 64], "f32")):
            name = f"sm90_{dtype}_gemm_64x64x16_{'a' if operand else 'c'}"
            case = base(name, "gemm", shape, value_dtype, {"m": 64, "n": 64, "k": 16, "operand_a": operand, "input_bits": 16})
            buffer = tvm.tirx.decl_buffer(shape, native_dtype if operand else "float32", scope="local.fragment")
            layout = emitter.make_mma_load_layout(buffer) if operand else emitter.make_mma_store_layout(buffer)
            fragment(case, layout)
            finish(case, "TensorCoreIntrinEmitter.make_mma_load_layout" if operand else "TensorCoreIntrinEmitter.make_mma_store_layout")

    for name, swizzle in (("linear", 0), ("transpose", -1), ("32b", 32), ("64b", 64), ("128b", 128)):
        shape = [8, 64]
        buffer = tvm.tirx.decl_buffer(shape, "float16", scope="shared")
        factories = {0: make_linear_layout, 32: make_quarter_bank_swizzled_layout,
                     64: make_half_bank_swizzled_layout, 128: make_full_bank_swizzled_layout}
        layout = Layout(shape, lambda i, j: [j, i]) if swizzle == -1 else factories[swizzle](buffer)
        case = base("shared_" + name, "shared", shape, "f16", {"swizzle": swizzle})
        storage(case, layout)
        finish(case, "Layout transpose" if swizzle == -1 else factories[swizzle].__name__)

    def infer(kernel):
        target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"})
        with target:
            mod = tvm.IRModule({"main": kernel})
            mod = tvm.tirx.transform.BindTarget(target)(mod)
            mod = tl.transform.MaterializeKernelLaunch()(mod)
            mod = tl.transform.LayoutInference()(mod)
        layouts = {}
        def visit(node):
            if isinstance(node, tvm.tirx.SBlock):
                for buffer, layout in node.annotations.get("layout_map", {}).items():
                    layouts[buffer.name] = layout
        tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
        return layouts

    def copy_fragment(width):
        return Fragment([8, 16], forward_thread_fn=lambda i, j: ((i * 16 + j) // width) % 32,
                        forward_index_fn=lambda i, j: ((i * 16 + j) % width) + ((i * 16 + j) // (32 * width)) * width)

    for width in (1, 2, 4):
        pin = copy_fragment(width)
        @T.prim_func
        def copy_kernel(A: T.Tensor((8, 16), "float32"), B: T.Tensor((8, 16), "float32")):
            with T.Kernel(1, threads=32):
                data = T.alloc_fragment((8, 16), "float32")
                T.annotate_layout({data: pin})
                T.copy(A, data, coalesced_width=width)
                T.copy(data, B, coalesced_width=width)
        layout = infer(copy_kernel)["data"]
        case = base(f"copy_vector_{width * 4}b", "copy", [8, 16], "f32", {"vector_bytes": width * 4})
        fragment(case, layout)
        storage(case, make_linear_layout(tvm.tirx.decl_buffer([8, 16], "float32", scope="shared")))
        finish(case, "LayoutInference(T.copy coalesced_width, pinned source Fragment)")

    for axis in (0, 1):
        pin = copy_fragment(1)
        result_size = 16 if axis == 0 else 8
        @T.prim_func
        def reduce_kernel(A: T.Tensor((8, 16), "float32"), B: T.Tensor((result_size,), "float32")):
            with T.Kernel(1, threads=32):
                data = T.alloc_fragment((8, 16), "float32")
                result = T.alloc_fragment((result_size,), "float32")
                T.annotate_layout({data: pin})
                T.copy(A, data)
                T.reduce_sum(data, result, dim=axis)
                T.copy(result, B)
        layouts = infer(reduce_kernel)
        for role, layout in (("input", layouts["data"]), ("result", layouts["result"])):
            shape = [8, 16] if role == "input" else [result_size]
            case = base(f"reduce_dim{axis}_{role}", "reduce", shape, "f32", {"axis": axis, "result": int(role == "result")})
            fragment(case, layout)
            finish(case, "LayoutInference(T.reduce_sum)." + role)

    # Native condense and reshape operations cover the remaining supported or
    # explicitly rejected semantics. No unexecuted inference claim is made.
    for name, shape in (("owner_compatibility", [4]), ("non_power_of_two_broadcast", [3])):
        layout = make_fully_replicated_layout_fragment(tvm.tirx.decl_buffer(shape, "float32", scope="local.fragment"), 32)
        reject = ("non-power-of-two", "general Tensor/broadcast extents require powers of two") if shape == [3] else None
        case = base(name, "broadcast" if reject else "owner_compatibility", shape, "f32", reject=reject)
        fragment(case, layout)
        finish(case, "make_fully_replicated_layout_fragment")

    unused_pin = make_fully_replicated_layout_fragment(tvm.tirx.decl_buffer([4], "float32", scope="local.fragment"), 32)
    @T.prim_func
    def floating_kernel(A: T.Tensor((4,), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=32):
            data = T.alloc_fragment((4,), "float32")
            unused = T.alloc_fragment((4,), "float32")
            T.annotate_layout({unused: unused_pin})
            T.copy(A, data)
            if data[0] >= 0:
                T.copy(data, B)
    floating_layouts = infer(floating_kernel)
    for name, key in (("floating_fragment", "data"), ("unused_fragment_seeded", "unused")):
        case = base(name, "owner_compatibility", [4], "f32")
        fragment(case, floating_layouts[key])
        finish(case, "LayoutInference: unseeded fragment access outside TileOp" if key == "data"
               else "LayoutInference: declared unused fragment with explicit fully replicated seed")

    padded = Fragment([4, 8], forward_thread_fn=lambda i, j: i * 8 + j, forward_index_fn=lambda i, j: 0)
    @T.prim_func
    def guarded_kernel(A: T.Tensor((3, 5), "float32"), B: T.Tensor((3, 5), "float32")):
        with T.Kernel(1, threads=32):
            data = T.alloc_fragment((4, 8), "float32")
            T.annotate_layout({data: padded})
            T.copy(A[0:3, 0:5], data[0:3, 0:5])
            T.copy(data[0:3, 0:5], B[0:3, 0:5])
    guarded = infer(guarded_kernel)["data"]
    case = base("ragged_padding_guard", "guarded_copy", [4, 8], "f32", {"vector_bytes": 1}, valid_shape=[3, 5])
    fragment(case, guarded)
    storage(case, make_linear_layout(tvm.tirx.decl_buffer([3, 5], "float32", scope="shared")))
    finish(case, "LayoutInference(T.copy 3x5 regions in pinned 4x8 Fragment); static valid region 3x5")

    layout = make_linear_layout(tvm.tirx.decl_buffer([4], "float32", scope="shared")).reshape([8], 2, 1)
    case = base("dtype_changing_view", "dtype_view", [8], "f16", {"source_bits": 32},
                reject=("dtype-changing-view", "layout_view preserves exact memref type; byte reinterpretation unsupported"))
    storage(case, layout)
    finish(case, "Layout.reshape([8], rescale_num=2, rescale_den=1)")

    partial = PartialFragment([4], forward_thread_fn=lambda i, rep: rep, forward_index_fn=lambda i: i,
                              replicate=32, combine=8)
    case = base("partial_fragment_addends", "partial_fragment", [4], "f32", {"combine": 8},
                reject=("partial-contribution", "Frisk reductions expose only broadcast_complete results"))
    fragment(case, partial)
    finish(case, "PartialFragment(replicate=32, combine=8): four groups of eight addends, not 32 equal copies")
    return sorted(cases, key=lambda c: c["name"]), sources


def validate_destinations(output, audit):
    if output.resolve() == audit.resolve() or (
            output.exists() and audit.exists() and os.path.samefile(output, audit)):
        raise ValueError("corpus and audit destinations must be distinct files")
    output.parent.mkdir(parents=True, exist_ok=True)
    for destination in (output, audit):
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise ValueError(f"destination must be a regular file: {destination}")
        if not destination.parent.is_dir():
            raise FileNotFoundError(f"destination parent does not exist: {destination.parent}")
        if not os.access(destination.parent, os.W_OK | os.X_OK):
            raise PermissionError(f"destination parent is not writable: {destination.parent}")


def publish_artifacts(output, audit, corpus_bytes, audit_bytes):
    """Stage both files, publish audit first, and commit corpus last.

    A failure may leave a newer audit beside the unchanged old corpus; the
    audit's corpus hash makes that mismatch detectable. Two independent paths
    cannot form one atomic filesystem transaction. Corpus bytes are preserved
    on every failure before the final rename. Cleanup is best effort and cannot
    turn a successful corpus commit into a reported failure.
    """
    validate_destinations(output, audit)
    staged = []
    try:
        for destination, contents in ((output, corpus_bytes), (audit, audit_bytes)):
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".tilelang-", delete=False) as file:
                staged.append(Path(file.name))
                file.write(contents)
                file.flush()
                os.fsync(file.fileno())
        validate_destinations(output, audit)
        os.replace(staged[1], audit)
        os.replace(staged[0], output)  # Last required, fallible publication step.
    finally:
        for path in staged:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass  # Do not report a failed export after corpus commit.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tilelang", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.tilelang.resolve()
    dependencies = verify_checkout(root, args.revision)
    if not args.child:
        audit_target = args.audit or args.output.with_suffix(".audit.json")
        validate_destinations(args.output, audit_target)
        # Isolated interpreter ignores ambient PYTHONPATH, then the child pins
        # its explicit first import roots before any TileLang/TVM import.
        with tempfile.TemporaryDirectory(prefix="frisk-tilelang-export-") as temporary:
            output = Path(temporary) / "corpus.json"
            audit = Path(temporary) / "audit.json"
            command = [sys.executable, "-I", str(Path(__file__).resolve()), "--child", "--tilelang", str(root),
                       "--revision", args.revision, "--output", str(output), "--audit", str(audit)]
            env = dict(os.environ)
            for name in ("PYTHONPATH", "TVM_LIBRARY_PATH", "TILELANG_LIGHT_IMPORT"):
                env.pop(name, None)
            env["TVM_IMPORT_PYTHON_PATH"] = str(root / "3rdparty/tvm/python")
            env["TILELANG_CACHE_DIR"] = str(Path(temporary) / "cache")
            subprocess.run(command, env=env, check=True)
            # Validate native child completed and emitted nonempty schema2.
            result = json.loads(output.read_text())
            if result.get("schema_version") != 2 or not result.get("cases"):
                raise ValueError("incomplete child export")
            corpus_bytes, audit_bytes = output.read_bytes(), audit.read_bytes()
        # Temporary-directory/cache cleanup happens BEFORE publication: even
        # a cleanup error must leave the previous corpus unchanged.
        publish_artifacts(args.output, audit_target, corpus_bytes, audit_bytes)
        return
    sys.path[:0] = [str(root), str(root / "3rdparty/tvm/python")]
    audit = native_audit(root)
    print(json.dumps(audit, sort_keys=True), file=sys.stderr, flush=True)
    cases, sources = export_cases()
    # Reject source changes during a long export as well as before import.
    if dependencies != verify_checkout(root, args.revision):
        raise ValueError("dependency changed during export")
    result = {"schema_version": 2, "tilelang_commit": REVISION, "dependencies": dependencies,
              "exporter_sha256": digest_file(__file__), "source_sha256": source_digest(root), "cases": cases}
    args.output.write_text(canonical_json(result))
    audit.update({"case_sources": sources, "corpus_sha256": digest_file(args.output),
                  "exporter_sha256": result["exporter_sha256"], "source_sha256": result["source_sha256"],
                  "dependencies": dependencies})
    args.audit.write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
