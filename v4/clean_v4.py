'''
Clean.py
File which conatins functions:
    clean: cleans correctly dataset from kaggle
    save_data_local: saves dataset into local path
    save_data_minio: saves dataset (type of dataset: numpy.array into Minio)
    load_data_minio: loads dataset (type of dataset: numpy.array into Minio)
'''
import torch
from torch.utils.data import TensorDataset
import infos_v4
import kaggle_v4
import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import io

# Cleaning data JSON
## get local file path
## return numpy arrays vith dataset cleaned
def clean_data(filepath):
    print('{}: clean function'.format(os.path.basename(__file__)))
    dataset = pd.read_json(open(filepath,'r'))

    x_set = dataset["data"]
    y_set = dataset["labels"]
    locations = dataset["locations"]
    scene_ids = dataset["scene_ids"]

    # Convert into lists
    X=[]
    Y=[]
    loc = []
    sc_id = []

    for k in x_set.keys():
        X.append(x_set[k])
        Y.append(y_set[k])
        loc.append(locations[k])
        sc_id.append(scene_ids[k])

    # Convert into numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    loc = np.array(loc)
    sc_id = np.array(sc_id)

    # Reshape X into correct
    X = X.reshape(-1,3,80,80)

    # Order corrctly X : (index, RGB, row, column)
    X = X.transpose(0,1,2,3)

    # Reshape labels, loc, scene_id
    Y = Y.reshape(-1)
    loc = loc.reshape(-1)
    sc_id = sc_id.reshape(-1)
    
    # Shuffling
    indexes = np.arange(4000)
    np.random.shuffle(indexes)
    x_train = X[indexes]
    y_train = Y[indexes]
    loc_train = loc[indexes]
    sc_id_train = sc_id[indexes]
    
    # Dvide x_train by 255
    X_train = x_train / 255
    
    return [X_train, y_train, loc_train, sc_id_train]
    
# Save data into local file
## get panda.Dataframe to save and file_name
def save_data_local(dataset,file_name):
    print('{}: save_data_local function'.format(os.path.basename(__file__)))
    try:
        dataset.to_pickle(file_name)
        print('save {0} succeed'.format(file_name))
    except:
        print('Error: save {0} not succeed'.format(file_name))

# Save data into Minio
def save_data_minio(client, dataset, bucket_name, filepath):
    print('{}: save_data_minio function'.format(os.path.basename(__file__)))
    
    # Convert numpy into bytes with shapes and types
    buf = pickle.dumps(dataset)
    
    # Save dataset into Minio as BytesIO object
    client.put_object(bucket_name, filepath, data=io.BytesIO(buf), length=len(buf))

# Load dataset from Minio
## return panda.DataFrame
def load_data_minio(client, bucket_name, filepath):
    print('{}: load_data_minio function'.format(os.path.basename(__file__)))
    try:
        response = client.get_object(bucket_name, filepath)
        data = response.read() # Get bytes dataset
        dataset = pickle.loads(data) # convert into numpy array
    finally:
        response.close()
        response.release_conn()
    return dataset
