# HDDFeaturesX

**HDDFeaturesX** is a Python library designed for **feature selection** in high-dimensional datasets. It supports both **binary** and **multi-class** problems and is compatible with various **machine learning** and **deep learning** models.

This study proposes a feature selection method motivated by **rough set theory**, inspired by the sample and feature selection approach introduced by Yang in 2022 (*Yang et al., 2022*). For more details, refer to:  
**Yang, Y., Chen, D., Zhang, X., Ji, Z., & Zhang, Y. (2022)**. *Incremental feature selection by sample selection and feature-based accelerator*. Applied Soft Computing, 121. [https://doi.org/10.1016/j.asoc.2022.108800](https://doi.org/10.1016/j.asoc.2022.108800).

In this study, we propose an enhanced version of Induced Partitioning for Incremental Feature Selection, combining Rough Set Theory and the Long-Tail Position Grey Wolf Optimizer. This method has been accepted for publication in **Acta Informatica Pragensia**.


## Objective
The library facilitates feature selection for high-dimensional datasets, supporting:
- Binary and multi-class classification problems.
- Seamless integration with machine learning and deep learning models.

## Installation
Install the library using pip:
```bash
pip install HDDFeaturesXS


## Usage
Here is an example of how to use the library:

from HDDFeaturesXS import *

import numpy as np
import time
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Load dataset (example using a .mat file)
data_path = 'PCMAC.mat'
data_loaded = loadmat(data_path)
data_key = next(key for key in data_loaded.keys() if not key.startswith('__'))
data = data_loaded[data_key]

# Partition the dataset into two parts
parts = partition_fold(data, 2)
ori_data = parts[0]
A = parts[1]

# Further partition part A into 5 parts
U = partition_fold(A, 5)

# Perform initial feature selection
fea_slt, ds_vector, fea_redun, sam_delete, ori_time = fast_feature_selection_ds(ori_data)

# Incrementally update feature selection as new data arrives
add_data = np.array([])
in_fea_slt_fs = []
unuse_ds = []
nrf_fs = []
nrs_fs = []
time_in_ds_fs = np.array([])

for i in range(5):
    add_data = np.vstack([add_data, U[i]]) if add_data.size else U[i]
    result = incre_fea_slt_ds_filter_sam(ori_data, add_data, ds_vector, fea_slt)
    in_fea_slt_fs.append(result[0])
    unuse_ds.append(result[1])
    nrf_fs.append(result[2])
    nrs_fs.append(result[3])
    time_in_ds_fs = np.append(time_in_ds_fs, result[4])

# Calculate total execution time
end_time = time.time()
total_time = end_time - start_time

# Use the selected features for training and testing
X_selected = ori_data[:, fea_slt]
y = ori_data[:, -1]
y = y.astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train a KNN model
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# Predict classes for the test data
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save incremental feature selection execution times
results_path = 'timeInDSfs.txt'
np.savetxt(results_path, time_in_ds_fs, delimiter=',')

# Print total execution time
print(f"Total execution time: {total_time} seconds")
