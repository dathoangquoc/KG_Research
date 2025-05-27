import json
with open('../../datasets/2wikimultihopqa.json', 'r') as f:
    data = json.load(f)

print(json.dumps(data[10], indent=4))
