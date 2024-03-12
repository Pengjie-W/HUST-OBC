import copy
import json
import os
import shutil
from tqdm import tqdm
import random

folder_path = './OCR_Dataset'
dataset = []
for root, directories, files in tqdm(os.walk(folder_path)):

    for file in files:
        data = {}
        file_path = os.path.join(root, file)

        data['label'] = int(file.replace('.png',''))
        data['path'] = file_path
        dataset.append(copy.deepcopy(data))

print(len(dataset))
with open('OCR_train.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False)
