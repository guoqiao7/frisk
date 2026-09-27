"""Assert legacy inference is absent from production sources and link products."""
import argparse
import pathlib
import re
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, required=True)
    parser.add_argument("--build-dir", type=pathlib.Path)
    args = parser.parse_args()
    patterns = [
        re.compile(r"\b(?:ParallelOp|GemmOp|ReduceOp)\s*::\s*inferLayout\s*\("),
        re.compile(r"\binferLayout\s*\([^;{}]*DenseMap\s*<", re.S),
        re.compile(r'"inferLayout"'),
        re.compile(r"\bgetLegacyParallelInferenceCallCount\b"),
        re.compile(r"LegacyLayoutOracle"),
    ]
    violations = []
    for directory in ("include", "lib", "ffi", "tools"):
        for path in sorted((args.source_root / directory).rglob("*")):
            if path.suffix not in (".h", ".cpp", ".td"):
                continue
            text = path.read_text()
            for pattern in patterns:
                if pattern.search(text):
                    violations.append(f"{path}: production legacy inference reference")
                    break
    if args.build_dir:
        products = [args.build_dir / "bin/frisk-opt"]
        products.extend(args.build_dir.glob("frisk_ffi*.so"))
        assert len(products) > 1, "built frisk_ffi product missing"
        for product in products:
            symbols = subprocess.run(
                ["nm", "-C", str(product)], check=True, text=True,
                capture_output=True).stdout
            # Other test adapters predate this migration and may be linked by
            # production libraries. Check the retired APIs/oracle specifically.
            if re.search(
                r"mlir::frisk::(?:test::(?:inferLegacy(?:Parallel|Gemm|Reduce)Layout|"
                r"getLegacyParallelOracleCallCount|computeUsedExtentForDim|"
                r"compressReplicateDimInMap|inferFragmentIndexFromThreadMap)|"
                r"(?:ParallelOp|GemmOp|ReduceOp)::inferLayout)", symbols
            ):
                violations.append(f"{product}: production links legacy oracle/inference")
        for path in (args.build_dir / "include").rglob("FriskOps*.inc"):
            if re.search(r"\binferLayout\s*\(", path.read_text()):
                violations.append(f"{path}: generated legacy inference API")
    assert not violations, "\n".join(violations)
    print("legacy inference production boundary: PASS")


if __name__ == "__main__":
    main()
