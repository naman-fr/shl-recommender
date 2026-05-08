import requests
r = requests.post('http://localhost:8000/chat', json={'session_id':'debug','message':'What is the weather today?'})
d = r.json()
print(f"Recs: {len(d['recommendations'])}")
print(f"End: {d['end_of_conversation']}")
print(f"Resp: {d['response'][:300]}")
