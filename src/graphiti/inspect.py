import json
with open('../../datasets/2wikimultihopqa.json', 'r') as f:
    data = json.load(f)

print(json.dumps(data[2], indent=4))
