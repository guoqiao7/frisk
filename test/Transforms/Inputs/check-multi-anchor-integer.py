"""Check both the certified choices and the actual solver-produced uses."""
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {
    "ss": ([[(1, 2), (2, 1)]], 0),
    "rs": ([[(2, 1)]], 0),
    "chain": ([[(1, 2), (2, 1)], [(2, 1)]], 0),
    "fanout-small": ([[(1, 2)], [(2, 1)], [(2, 1)]], 1),
    "write1": ([[(2, 1)], [(2, 1)]], 1),
}
for name, (eligible, copies) in expected.items():
    ir = (root / (name + ".out")).read_text()
    anchors = re.search(r"anchors = \[\[(.*?)\]\],", ir).group(1).split("], [")
    selected = list(map(int, re.search(r"selected = array<i64: ([\d, ]+)>", ir).group(1).split(", ")))
    assert len(anchors) == len(eligible) == len(selected), name
    for table, allowed, winner in zip(anchors, eligible, selected):
        rows = re.findall(r"eligible = (true|false), grid = array<i64: (\d+), (\d+)>", table)
        assert [(int(m), int(n)) for e, m, n in rows if e == "true"] == allowed, name
        assert rows[winner][0] == "true", name
        if name == "chain":
            assert rows[winner][1:] == ("2", "1")
    assert "optimal = true" in ir
    assert int(re.search(r"enumeration_feasible = (\d+)", ir).group(1)) > 0
    assert f"distinct_conversions = {copies} " in ir
    converts = re.findall(r"(%\w+) = frisk\.convert_layout ", ir)
    assert len(converts) == copies, name
    if name == "fanout-small":
        assert sum(" = frisk.mma " in line and converts[0] + " " in line for line in ir.splitlines()) == 2
    if name == "write1":
        def units(key):
            return int(re.search(r"\b" + key + r" = (\d+)", ir).group(1))
        assert units("access_units") + units("conversion_units") < units("best_without_conversions")
        assert units("f2_completions") > 0
        assert "writer_candidates_selected = 1 " in ir
        assert "frisk.tile_store " + converts[0] + "," in ir
