import copy
import json
import os
from pathlib import Path
import numpy as np

dataset=[]
X=[]
y=[]
for root, directories, files in os.walk('../HUST-OBS/deciphered/'):
    for file in files:
        data={}
        if 'json'in file:
            continue
        file_path = str(Path(os.path.join(root, file)))
        folders = os.path.split(file_path)[0].split(os.sep)
        folder_name = folders[3]
        y.append(folder_name)
        X.append(file_path)
from sklearn.model_selection import StratifiedShuffleSplit

unique_classes, class_counts = np.unique(y, return_counts=True)
single_sample_classes = unique_classes[class_counts == 1]
# 将只有一个样本的类别放入训练集，其他类别放入测试集
train_indices = [idx for idx, label in enumerate(y) if label in single_sample_classes]
test_indices = [idx for idx in range(len(y)) if idx not in train_indices]

# 找出除只有一个样本的类别外的其他类别
remaining_classes = unique_classes[class_counts > 1]
X_train = [X[idx] for idx in train_indices]
y_train = [y[idx] for idx in train_indices]
# 将剩余类别进行分层采样
X_remaining = [X[idx] for idx in test_indices]
y_remaining = [y[idx] for idx in test_indices]


stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_indices, test_indices in stratified_splitter.split(X_remaining, y_remaining):
    X_train_remaining = [X_remaining[idx] for idx in train_indices]
    y_train_remaining = [y_remaining[idx] for idx in train_indices]
    X_test_remaining = [X_remaining[idx] for idx in test_indices]
    y_test_remaining = [y_remaining[idx] for idx in test_indices]
dataset={}
num=0
for i in unique_classes:
    if i not in dataset:
        dataset[i]=num
        num+=1
train=[]
for i in range(len(X_train)):
    data={}
    data['path']=X_train[i]
    data['label']=dataset[y_train[i]]
    train.append(copy.deepcopy(data))
for i in range(len(X_train_remaining)):
    data={}
    data['path']=X_train_remaining[i]
    data['label']=dataset[y_train_remaining[i]]
    train.append(copy.deepcopy(data))
test=[]
for i in range(len(X_test_remaining)):
    data={}
    data['path']=X_test_remaining[i]
    data['label']=dataset[y_test_remaining[i]]
    test.append(copy.deepcopy(data))
with open('Validation_train.json','w',encoding='utf8') as f:
    json.dump(train, f, ensure_ascii=False)
with open('Validation_label.json','w',encoding='utf8') as f:
    json.dump(dataset, f, ensure_ascii=False)
with open('Validation_test.json','w',encoding='utf8') as f:
    json.dump(test, f, ensure_ascii=False)
print(len(train))
print(len(test))

