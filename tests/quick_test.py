import requests
r = requests.get('http://localhost:8000/health')
print("Health:", r.json())

r2 = requests.post('http://localhost:8000/chat', json={'session_id': 'test1', 'message': 'Java tests for developers'})
d = r2.json()
print(f"Response length: {len(d['response'])}")
print(f"Cards: {len(d['recommendations'])}")
names = [c['name'] for c in d['recommendations'][:5]]
print(f"Top 5: {names}")

# Test refinement
r3 = requests.post('http://localhost:8000/chat', json={'session_id': 'test1', 'message': 'only remote ones under 15 minutes'})
d3 = r3.json()
print(f"\nRefined cards: {len(d3['recommendations'])}")
for c in d3['recommendations'][:3]:
    print(f"  - {c['name']} | {c['duration']} | remote={c['remote']}")

# Test comparison
r4 = requests.post('http://localhost:8000/chat', json={'session_id': 'test1', 'message': 'compare the top two'})
d4 = r4.json()
print(f"\nComparison response snippet: {d4['response'][:200]}")
