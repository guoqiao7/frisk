# Pinned TileLang semantic reference

Task 28 uses TileLang `5e149e31674658f94779c7d0c6039549a1853123`,
TVM `907a88c8791ccf33b9874821bc875e7abf624367`, and CUTLASS
`b2dd65dc864e09688245b316ac46c4a6cd07e15c`. This is a test-only
reference: no TVM, TIRX, CUDA, or TileLang library is linked into Frisk.
Ordinary CI reads the audited static JSON and runs the Python trust-boundary
tests without installing TileLang. The real native run on 2026-10-09 exported
23 cases: **20 equivalent and 3 expected rejections**, independently compared
with Frisk. Task 28's independent review and regression gate passed; the
complete M6 Gate remains pending Task 29. This is not a GPU/performance result.

## Schema 2

The top-level object contains exactly `schema_version`, `tilelang_commit`,
`dependencies`, `exporter_sha256`, `source_sha256`, and `cases`. Dependencies
record every recursive gitlink revision; the exporter verifies initialized,
exact, clean submodules. Source SHA-256 hashes sorted tracked root file names
and bytes (excluding gitlink directories); dependency contents are identified
by their audited clean revisions. Exporter SHA-256 covers the exporter file.

Every case has a stable name, operation, dtype, logical `shape`, `valid_shape`,
five carrier extents, storage capacity in bytes, operation parameters,
expectation, value semantics, distributed points, storage points, and writers.
Carrier order is `(register, lane, warp, warp_group, cta)`; register denotes an
element slot, including packed FP16/BF16 halfwords, not necessarily a physical
32-bit register. Election priority reverses this tuple. Storage points carry
an explicit byte offset and a bit offset in `[0,8)`; the complete element bit
interval must be in bounds and disjoint from other elements.

`valid_shape` is an origin-anchored rectangular domain inside `shape`. All
padded distributed coordinates are enumerated, including invalid tail points;
only valid coordinates acquire storage addresses or writers. This initial
schema does not describe arbitrary predicates or dynamic dimensions.

`value_semantics: complete` means all holders of one logical point are equal
copies; `copy_group` is scoped to that logical point and must agree across its
holders, while `addend` is `-1`. For `partial`, `copy_group` identifies an
independent group and `addend` identifies a contribution within it. Partial
cases are expected rejections, never compared as completed replicas. The
initial partial case has four groups of eight addends, not 32 equal values.

`writer_policy: compatible_first_owner` is explicitly a comparison policy,
not a claim that TileLang exposed or selected that writer. The exporter elects
the minimum legal completed holder; Frisk's actual ownership proof and
enumeration must agree with this legal policy. `none` is used for storage-only
and partial cases. Duplicate case names, duplicate/conflicting carriers,
unexplained copy groups, incomplete coverage, invalid domain flags, overlapping
storage, and invalid/missing/duplicate writers are errors. The reader bounds
file size, case count, and enumeration products and executes no data as code.

## Semantic scope

| Category | Native reference construction | Frisk comparison / boundary |
| --- | --- | --- |
| SM90 f16/bf16 GEMM | WGMMA emitter's actual A/load and C/store layouts | `buildSM90MmaFragment` and instruction proof; enumerate canonical maps |
| Shared linear/transpose/32B/64B/128B | Native Layout, linear and bank-swizzle factories | Verified canonical storage map, complete element-bit addresses |
| Copy 4B/8B/16B | Actual `T.copy` + LayoutInference with pinned Fragment and coalesced width | Canonical vector execution map and vector access proof |
| Reduce dim 0/1 | Actual `T.reduce_sum` + LayoutInference; both source and completed output | `projectReductionEncoding`, reduction dependency proof, ownership enumeration |
| Static guarded Copy | Actual 3x5 Copy regions with pinned padded 4x8 Fragment + LayoutInference | Padded carrier map, unchanged 3x5 storage, actual guarded access proof |
| Owner compatibility | Native fully replicated Fragment | Complete holders and compatible first-owner proof |
| Floating fragment | Actual LayoutInference: unseeded fragment read in a condition outside TileOps | Compare resulting fully replicated map with an explicit verified Frisk map; not a claim of identical inference policy |
| Unused fragment | Actual LayoutInference: declared unused fragment with a fully replicated seed | Compare retained explicit layout; no claim about unseeded unused-buffer policy |
| Non-power-of-two broadcast | Native fully replicated shape-3 Fragment | Expected rejection by general Tensor inference boundary |
| Dtype-changing view | Native `Layout.reshape` with element rescaling | Expected rejection by exact-type `layout_view` verifier |
| PartialFragment | Native PartialFragment with explicit combine width | Expected rejection of partial output policy by existing reduction contract |

The corpus tests layout and static operation-contract semantics, not TileLang
kernel numerical execution, algorithm selection optimality, GPU performance,
or a complete equivalence of the two compilers. Pinning source layouts in Copy
and Reduce makes the shared supported domain explicit. Unsupported categories
are counted separately and do not increase the semantic-equivalence count.

## Re-export and upgrades

Use an independent source clone and Python venv; never import or modify the
user's existing TileLang checkout. The parent exporter launches `python -I`,
inserts the pinned checkout and its TVM Python tree first, verifies module
origins, and checks actually loaded native libraries through `/proc/self/maps`.
`apache-tvm-ffi==0.1.12` must come from the isolated venv. All ten Python runtime
dependencies are checked against `PYTHON_DEPENDENCIES` in the exporter (including
torch 2.8.0 and z3-solver 4.15.4.0). Native paths, binary
hashes, Python path, and per-case native APIs are kept in a separate audit log,
not canonical JSON. Timestamps and temporary absolute paths are not corpus
fields. Failed export does not replace an existing corpus: destinations are
validated before the child and before publication, including same-path,
resolved symlink/directory aliases and existing hardlinks. Both artifacts are
staged and flushed; audit is published first and corpus replacement is the
last required fallible step. Child/cache cleanup precedes publication, and
leftover staging cleanup is best effort. This is not a two-file transaction:
if the final corpus rename fails, the previous corpus remains and a newer audit
may exist; its corpus hash detects that mismatch.

```bash
/tmp/frisk-m6-tilelang-venv/bin/python tools/layout-reference/export_tilelang_layouts.py \
  --tilelang /tmp/frisk-m6-tilelang-reference \
  --revision 5e149e31674658f94779c7d0c6039549a1853123 \
  --output test/Dialect/Frisk/layout/Inputs/tilelang_layout_corpus.json \
  --audit /tmp/frisk-m6-tilelang-export.audit.json
/tmp/frisk-m6-tilelang-venv/bin/python tools/layout-reference/export_tilelang_layouts.py \
  --tilelang /tmp/frisk-m6-tilelang-reference \
  --revision 5e149e31674658f94779c7d0c6039549a1853123 \
  --output /tmp/frisk-m6-tilelang-corpus-second.json \
  --audit /tmp/frisk-m6-tilelang-export-second.audit.json
cmp test/Dialect/Frisk/layout/Inputs/tilelang_layout_corpus.json /tmp/frisk-m6-tilelang-corpus-second.json
cmake --build build --target FriskLayoutUnitTests --parallel 12
build/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_filter='TileLang*'
python test/python/tilelang_exporter_test.py
```

An upgrade requires an explicitly chosen full revision, exact recursive
dependencies, a fresh isolated native build, audited imports, two byte-identical
exports, independent review of each changed semantic case and its Frisk
construction/rejection, and all reader/differential regressions. Do not silently
accept a new revision or overwrite a mismatch with Frisk-generated coordinates.
Old revision corpora, if retained, need separately labeled fixtures and readers.

## Build and run evidence

The independent reference was cloned from the previously audited clean source
at `/tmp/frisk-m4-audit.sKsPEx/tilelang`, detached at the full pinned commit,
and all recursive gitlinks initialized. Network TLS failures fetching CUTLASS
were resolved using a read-only local clone containing the exact pinned object;
the user's checkout was not edited. All eleven dependency revisions are in the
canonical corpus and `/tmp/frisk-m6-tilelang-submodules.log`.

The actual Python environment was created with the project's Python 3.10.18
using `-m venv --system-site-packages /tmp/frisk-m6-tilelang-venv`; only
`apache-tvm-ffi==0.1.12` was installed into that venv. Inherited dependencies
were version-audited, not assumed correct. Full installed package versions are
in `/tmp/frisk-m6-tilelang-python-lock.txt`; native export enforces the relevant
runtime pins. CMake was 4.2.1, Ninja 1.13.0, GNU C++ 11.4.0, CUDA 12.8.61.

Actual successful native build commands (Release, CUDA enabled, LLVM disabled):

```bash
cmake -S /tmp/frisk-m6-tilelang-reference \
  -B /tmp/frisk-m6-tilelang-reference/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE=/tmp/frisk-m6-tilelang-venv/bin/python \
  -DPython3_EXECUTABLE=/tmp/frisk-m6-tilelang-venv/bin/python \
  -DUSE_CUDA=ON -DUSE_LLVM=OFF -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build /tmp/frisk-m6-tilelang-reference/build \
  --target tilelang tilelang_cython_wrapper --parallel 12
```

Configure/build logs: `/tmp/frisk-m6-tilelang-configure.log` and
`/tmp/frisk-m6-tilelang-build.log`. The child redirects TileLang's cache into
its temporary directory; it never requires the user's default cache path.

Actually loaded Python modules were the reference `tilelang/__init__.py`,
reference `3rdparty/tvm/python/tvm/__init__.py`, and venv
`lib/python3.10/site-packages/tvm_ffi/__init__.py`. `libtilelang.so`,
`libtvm_compiler.so`, `libtvm_runtime.so`, and the Cython wrapper came from
reference `build/lib`; FFI libraries came from the venv's `tvm_ffi` directory.
The two native audits (including every loaded reference-library hash and the
exact API used for each case) are `/tmp/frisk-m6-tilelang-export.audit.json`
and `/tmp/frisk-m6-tilelang-export-second.audit.json`.

The two actual export commands above succeeded and `cmp` returned zero.
Canonical data contains 11,680 distributed records, 2,967 storage records,
and 10,934 policy-compatible writers. There are no timestamps or absolute
paths in the canonical corpus. SHA-256 values:

```text
exporter 5ebad2735a4fa6368b4901dd518fee73fcc4aa7ff2f9e204b134f68cac9e7453
source   c52ff105af3ecb286fa32fab1d2a088b6b781d52cb60c3000bfdd996720933d5
corpus   bb4a69f647305e9017ecdbb063e87a1a4cf39ec33cf5ee77ec0dd3936842fd44
audit    ff75b70386a2d2af44ea3dfca2b5d8be58c27ff88061ac45c97442685f801b45
```

Local validation: 9 C++ tests passed (8 strict-reader tests plus the 23-case
differential test), and 12 Python tests passed using `/usr/bin/python3 -I -S`,
which disables site-packages. The registered `FriskTileLangExporterTest` CTest
also passed. Final post-review raw logs:
`/tmp/frisk-m6-task28-i1-unit.{log,xml}`,
`/tmp/frisk-m6-task28-i1-python-green.log`, `/tmp/frisk-m6-task28-i1-ctest.log`,
`/tmp/frisk-m6-task28-i1-export-{first,second}.log`, and
`/tmp/frisk-m6-task28-i1-cmp-sha256.log`. The prior review found publication
ordering/collision issue I1; its failure-preservation regressions were first
observed failing, then passed after the bounded Python-only fix.

Frisk constructions consume operation parameters, never corpus coordinates:
production SM90 fragments, production reduction projection/proofs, independent
canonical bit-linear Copy/replication maps, and verified affine storage maps.
Copy footprint proofs use real parsed `memref.alloc`/`frisk.layout_view` IR and
`analyzeStorageAlias`, not a synthetic proof marker. Rejections invoke actual
Tensor constraint collection, `layout_view` verification, and reduction-contract
verification; the partial-policy rejection also checks its completed-policy
control succeeds. These are static proofs, not execution of GPU kernels.
