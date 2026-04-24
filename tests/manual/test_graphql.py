import os, httpx, json
with open('.env') as f:
    for line in f:
        if line.startswith('RUNPOD_API_KEY='):
            key = line.strip().split('=')[1].replace('"', '').replace("'", '')
            break

query = '''
query {
  __schema {
    mutationType {
      fields {
        name
      }
    }
  }
}
'''
resp = httpx.post(f'https://api.runpod.io/graphql?api_key={key}', json={'query': query})
print(json.dumps(resp.json(), indent=2))
