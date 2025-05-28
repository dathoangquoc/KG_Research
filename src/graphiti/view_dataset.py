import json
with open('datasets/2wikimultihopqa.json', 'r') as f:
    data = json.load(f)

print(json.dumps(data[0], indent=4))
