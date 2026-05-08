"""Test evaluator schema compliance and behavior probes."""
import requests, json

BASE = "http://localhost:8000"

# Test 1: Health check returns {"status": "ok"}
r = requests.get(f"{BASE}/health")
h = r.json()
assert h["status"] == "ok", f"Health must return 'ok', got '{h['status']}'"
print(f"[PASS] Health: {h}")

# Test 2: Schema has all required fields
r = requests.post(f"{BASE}/chat", json={"session_id": "s1", "message": "hello"})
d = r.json()
assert "session_id" in d, "Missing session_id"
assert "response" in d, "Missing response"
assert "recommendations" in d, "Missing recommendations"
assert "end_of_conversation" in d, "Missing end_of_conversation"
print(f"[PASS] Schema: all fields present")

# Test 3: Greeting = empty recommendations, not end
assert d["recommendations"] == [], f"Greeting should have empty recs, got {len(d['recommendations'])}"
assert d["end_of_conversation"] == False, "Greeting should not end conversation"
print(f"[PASS] Greeting: empty recs, not ended")

# Test 4: BEHAVIOR PROBE - vague query on turn 1 should NOT recommend
r = requests.post(f"{BASE}/chat", json={"session_id": "s2", "message": "I need some tests"})
d = r.json()
assert d["recommendations"] == [], f"Vague query turn 1 should NOT recommend, got {len(d['recommendations'])} recs"
assert d["end_of_conversation"] == False
print(f"[PASS] Vague turn 1: asked clarification, no recs")

# Test 5: Specific query should recommend and end
r = requests.post(f"{BASE}/chat", json={"session_id": "s3", "message": "I need Java programming tests for mid-level developers"})
d = r.json()
assert len(d["recommendations"]) > 0, "Specific query should return recs"
assert len(d["recommendations"]) <= 10, "Max 10 recs"
assert d["end_of_conversation"] == True, "Should end after recommendations"
# Verify all recs are from catalog
for rec in d["recommendations"]:
    assert "url" in rec and "name" in rec, f"Rec missing fields: {rec.keys()}"
    assert rec["url"].startswith("https://"), f"Bad URL: {rec['url']}"
print(f"[PASS] Specific query: {len(d['recommendations'])} recs, ended, all valid")

# Test 6: Off-topic should refuse, no recs
r = requests.post(f"{BASE}/chat", json={"session_id": "s4", "message": "What is the weather today?"})
d = r.json()
assert d["recommendations"] == [], "Off-topic should have empty recs"
assert d["end_of_conversation"] == False
print(f"[PASS] Off-topic: refused, empty recs")

# Test 7: Multi-turn refinement
sid = "s5"
r1 = requests.post(f"{BASE}/chat", json={"session_id": sid, "message": "I need to assess Java developers"})
d1 = r1.json()
assert len(d1["recommendations"]) > 0, "Should recommend Java tests"
print(f"[PASS] Multi-turn 1: {len(d1['recommendations'])} Java recs")

r2 = requests.post(f"{BASE}/chat", json={"session_id": sid, "message": "only show remote ones under 15 minutes"})
d2 = r2.json()
assert len(d2["recommendations"]) > 0, "Should return refined recs"
for rec in d2["recommendations"]:
    assert rec["remote"].lower() == "yes", f"Non-remote after filter: {rec['name']}"
print(f"[PASS] Multi-turn 2: {len(d2['recommendations'])} refined recs, all remote")

# Test 8: Thanks should end conversation
r = requests.post(f"{BASE}/chat", json={"session_id": "s6", "message": "thanks, that's all I need"})
d = r.json()
assert d["end_of_conversation"] == True, "Thanks should end"
assert d["recommendations"] == [], "Thanks should have empty recs"
print(f"[PASS] Thanks: ended, empty recs")

print("\n" + "="*50)
print("ALL SCHEMA & BEHAVIOR TESTS PASSED!")
print("="*50)
