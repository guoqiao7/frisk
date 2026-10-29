import json, sys
trace = json.load(open(sys.argv[1]))
assert trace["cost_model_id"] == "sm90-layout-lex9-static-trip-count-v1"
assert trace["boundaryFrequency"] == "static-trip-count"
solutions = [e for e in trace["events"] if e["event"] == "solution"]
assert solutions[-1]["cost"]["cost"] == [0, 0, 0, 51200, 28, 0, 0, 1, 0]
edge = solutions[-1]["conversions"][0]
assert edge["bytes"] == 512 and edge["frequencyFactor"] == 100
assert not edge["frequencyUsesProxy"] and not edge["frequencySaturated"]
stats = json.load(open(sys.argv[2]))
assert stats["cost_model_id"] == trace["cost_model_id"]
assert stats["boundaryFrequency"] == "static-trip-count"
traffic = stats["attempts"][-1]["statistics"]
assert traffic["conversionBytes"] == 512
assert traffic["weightedConversionBytes"] == 51200
assert not traffic["weightedConversionBytesUsesProxy"]
assert not traffic["weightedConversionBytesSaturated"]
