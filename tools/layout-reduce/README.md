# Finite layout failure replay

`layout-reduce --round-trip FILE`, `--replay FILE`, and `--reduce FILE` use
the same strict parser as `LayoutFailureReductionTest`. Output is JSON on
stdout; errors use stderr and nonzero exit status. `-` reads standard input.

Schema 1 (`domain: finite-same-require-v1`) supports at most eight distributed
finite variables (zero to four distinct string candidates each) and sixteen
hard `same` / `require` relations. This is a **test fixture target**, not a GPU
layout target or a second layout inference engine. Replay invokes the actual
production solver with zero fixture cost. Finite instruction-table, conversion,
SSA/use, alias, region, storage, and hardware-cost payloads are **unsupported**;
unknown fields at every object level are rejected, never silently dropped.

Required fields are `schema`, `domain`, nonnegative `seed`, `host` (parseable
host MLIR), `options` (`exact`, `beam`, `expanded`), `variables` (`id`,
`candidates`), `relations` (`id`, `kind`, `vars`, `required`, `rule`, `reason`),
and `failure` (`kind`, `variable`, `relation`, `rule`, `reason`). All IDs are
stable strings. No native addresses or operation pointers are serialized.
The host is retained even though the supported finite domain has no borrowed
IR endpoints. See `test/Transforms/Inputs/layout-empty.json` for a complete
example. `--round-trip` validates syntax, host and graph but does not claim
that the recorded failure reproduces; `--replay` and `--reduce` do.

The failure signature compares structured kind, stable variable/relation,
provenance rule and diagnostic/provenance reason. Reduction cannot replace a
failure with a different kind or an unrelated same-kind failure. It tries
relations, unreferenced variables, and candidates in stable order, rebuilding
and validating the graph before every replay. Referenced variable deletion
is not a legal single deletion. It repeats to a fixed point: **1-minimal**
under these valid single deletions, not globally smallest. Serialization sorts
variables/candidates/relations but preserves relation endpoint order.

`FriskLayoutFailureSupport` is linked only by the CLI and unit tests. Production
analysis does not link this target or its finite test cost model. The raw
`reduceFailingLayoutGraph` entry is additionally fail-closed for unsupported
borrowed identities, shapes, payloads and provenance; it is not a file format.
The raw entry preserves each surviving candidate's original `stableOrdinal`
by stable variable/candidate identity, including after deletion. It checks its
predicate on the initial canonical rebuild and final result; an input whose
predicate is lost by canonicalization is rejected, never returned as a reduction.
