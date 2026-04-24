import os, httpx, json
with open('.env') as f:
    key = [line.strip().split('=')[1].replace('"', '').replace("'", '') for line in f if line.startswith('RUNPOD_API_KEY=')][0]

resp = httpx.post(
    f'https://api.runpod.io/graphql?api_key={key}',
    json={
        'query': '''
            mutation {
                updateEndpoint(endpointId: "8djiexluyybooj", name: "oelala-i2i") {
                    id
                    name
                }
            }
        '''
    },
    timeout=30
)
print(json.dumps(resp.json(), indent=2))
