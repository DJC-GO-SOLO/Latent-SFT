import json
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from tqdm import tqdm
from eval_utils.grader import check_is_correct
from eval_utils.parser import extract_answer
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data    
def write_jsonl(data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')

data = read_jsonl('results.jsonl')

results = []
for i in tqdm(data):
    pred = extract_answer(i['prediction'])
    results.append(check_is_correct(pred,i['answer']))

print(f'Scores: {sum(results)/len(results)}')

