import copy
import json
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
# 设置随机种子
seed = 42
np.random.seed(seed)
import random
random.seed(seed)

dataset = []
X = []
y = []
for root, directories, files in os.walk('../HUST-OBC/deciphered/'):
    for file in files:
        data = {}
        if 'json' in file:
            continue
        file_path = str(Path(os.path.join(root, file)))
        folders = os.path.split(file_path)[0].split(os.sep)
        folder_name = folders[3]
        y.append(folder_name)
        X.append(file_path)

# 找出样本数量
unique_classes, class_counts = np.unique(y, return_counts=True)
single_sample_classes = unique_classes[class_counts == 1]
multiple_sample_classes = unique_classes[class_counts > 1]

# 将只有一个样本的类别放入训练集
train_indices = [idx for idx, label in enumerate(y) if label in single_sample_classes]
remaining_indices = [idx for idx in range(len(y)) if idx not in train_indices]

# 剩余样本分割为训练集和验证/测试集 (8:2)
X_train = [X[idx] for idx in train_indices]
y_train = [y[idx] for idx in train_indices]

X_remaining = [X[idx] for idx in remaining_indices]
y_remaining = [y[idx] for idx in remaining_indices]

# 分层抽样分割
stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_val_idx in stratified_splitter.split(X_remaining, y_remaining):
    X_train_remaining = [X_remaining[idx] for idx in train_idx]
    y_train_remaining = [y_remaining[idx] for idx in train_idx]
    X_test_val = [X_remaining[idx] for idx in test_val_idx]
    y_test_val = [y_remaining[idx] for idx in test_val_idx]

X_train.extend(X_train_remaining)
y_train.extend(y_train_remaining)

unique_classes, class_counts = np.unique(y_test_val, return_counts=True)
single_sample_classes = unique_classes[class_counts == 1]
multiple_sample_classes = unique_classes[class_counts > 1]

# 将只有一个样本的类别放入随机放入验证集和测试集
train_indices = [idx for idx, label in enumerate(y_test_val) if label in single_sample_classes]
remaining_indices = [idx for idx in range(len(y_test_val)) if idx not in train_indices]

X_val = []
y_val = []
X_test = []
y_test = []

for idx in train_indices:
    if np.random.rand() < 0.5:
        X_val.append(X_test_val[idx])
        y_val.append(y_test_val[idx])
    else:
        X_test.append(X_test_val[idx])
        y_test.append(y_test_val[idx])

# 将其余样本进行分层抽样分割验证集和测试集 (1:1)

X_test_val_remaining = [X_test_val[idx] for idx in remaining_indices]
y_test_val_remaining = [y_test_val[idx] for idx in remaining_indices]

stratified_splitter_test_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for val_idx, test_idx in stratified_splitter_test_val.split(X_test_val_remaining, y_test_val_remaining):
    X_val.extend([X_test_val_remaining[idx] for idx in val_idx])
    y_val.extend([y_test_val_remaining[idx] for idx in val_idx])
    X_test.extend([X_test_val_remaining[idx] for idx in test_idx])
    y_test.extend([y_test_val_remaining[idx] for idx in test_idx])

unique_classes, class_counts = np.unique(y, return_counts=True)
# 创建标签字典
dataset = {label: idx for idx, label in enumerate(unique_classes)}

# 保存训练集、验证集和测试集
train_data = [{'path': path, 'label': dataset[label]} for path, label in zip(X_train, y_train)]
val_data = [{'path': path, 'label': dataset[label]} for path, label in zip(X_val, y_val)]
test_data = [{'path': path, 'label': dataset[label]} for path, label in zip(X_test, y_test)]

with open('Validation_train.json', 'w', encoding='utf8') as f:
    json.dump(train_data, f, ensure_ascii=False)
with open('Validation_label.json', 'w', encoding='utf8') as f:
    json.dump(dataset, f, ensure_ascii=False)
with open('Validation_val.json', 'w', encoding='utf8') as f:
    json.dump(val_data, f, ensure_ascii=False)
with open('Validation_test.json', 'w', encoding='utf8') as f:
    json.dump(test_data, f, ensure_ascii=False)

print(f'Training set size: {len(train_data)}')
print(f'Validation set size: {len(val_data)}')
print(f'Test set size: {len(test_data)}')
