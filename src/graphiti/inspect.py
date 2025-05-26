import json
with open('../../datasets/2wikimultihop/test.json', 'r') as f:
    data = json.load(f)

print(json.dumps(data[0], indent=4))
