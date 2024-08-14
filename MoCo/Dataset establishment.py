import copy
import json
import os
import shutil
from tqdm import tqdm
import random

folder_path = '../HUST-OBC/deciphered'
dataset = []
for root, directories, files in tqdm(os.walk(folder_path)):

    for file in files:
        if'ID'in file:
            continue
        data = {}
        file_path = os.path.join(root, file)

        data['label'] = int(file[2:6])
        data['path'] = file_path
        dataset.append(copy.deepcopy(data))

print(len(dataset))
with open('MOCO_train.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False)
