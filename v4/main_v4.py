'''
Main.py
Main file which using clean_v*, infos_v*, kaggle_v*, cnn_v* files
Download dataset from Kaggle
Clean dataset
Save dataset into Minio
Load dataset fom Minio
Check if datasets are equal
Make CNN
Train model
Display some samples
'''
import clean_v4 as clean
import cnn_v4 as cnn
import infos_v4 as infos
import kaggle_v4 as kaggle
import minio
import minioClient_v4 as minioClient
import os
import pickle
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import class CNN
import random
import torch

# Part 1: Load data from Kaggle, clean, put in Minio delete localfiles

'''
# Delete all directories
kaggle.delete(infos.kaggle_local_path)
kaggle.delete(infos.clean_local_path)
'''
# get dataset from kaggle and save into local dir
if not(os.path.isdir(infos.kaggle_local_path)):
    print('{}: kaggle files already downloaded'.format(os.path.basename(__file__)))
    kaggle.get_data(infos.kaggle_url,infos.kaggle_local_path)

# init minio
minioCl = minioClient.init_minio(infos.ip_address,infos.access_key, infos.secret_key)

# clean data_set and get numpy arrays [x_tain, y_train, localisations, scenes]
dataset = clean.clean_data('{}shipsnet.json'.format(infos.kaggle_local_path))
#print('type data[0] {}'.format(type(dataset[0])))

# save clean_data into minio
clean.save_data_minio(minioCl,dataset[0],infos.dir_json, infos.images)
clean.save_data_minio(minioCl,dataset[1],infos.dir_json, infos.labels)
#clean.save_data_minio(minioCl,data[2],infos.dir_json, 'loc.pkl')
#clean.save_data_minio(minioCl,data[3],infos.dir_json, 'scenes.pkl')

# Part 2: Load data from minio
images = clean.load_data_minio(minioCl,infos.dir_json, infos.images) # images.flags['WRITEABLE'] = False can't be used in PyTorch

labels = clean.load_data_minio(minioCl,infos.dir_json, infos.labels)

# Copy numpy arrays to be readable
data = [images, labels]


print('Images array is equal ? : {}'.format(np.array_equal(dataset[0], data[0])))
print('Labes array is equal ? : {}'.format(np.array_equal(dataset[1], data[1])))

# Transform data into PyTorch dataset
## Tensors
tensor_x = torch.Tensor(data[0])
tensor_y = torch.Tensor(data[1])

## Convert labels float type into long type (labels need to be type long)
tensor_y = tensor_y.long()

## Create TensorDataset
tensorDataset = TensorDataset(tensor_x, tensor_y)

'''
print('type data {}'.format(type(tensorDataset)))
print('data : \n {}'.format(tensorDataset))
'''

# Function for prediction
## return number of good predictions
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

# Initiating model
network = cnn.CNN()
train_loader = DataLoader(tensorDataset,batch_size=100) #100 samples / batch

optimizer = optim.Adam(network.parameters(), lr=0.01)

# Launch epoches
for epoch in range(10):
    total_loss = 0
    total_correct = 0
    
    for batch in train_loader: # Get batch
        images, labels = batch
        preds = network(images) # Pass Batch
        loss = F.cross_entropy(preds, labels) # Calculate Loss

        # Update hyperparameters
        optimizer.zero_grad()
        loss.backward() # Calculate Gradients
        optimizer.step() # Update Weights
    
        # Save loss and number of good prediction / batch
        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)
    print(
        "Epoch:",epoch,
        "Total_correct:", total_correct,
        "Loss:", total_loss
        )

# Get predictions for All samples
## return all predictions of the loader by the model
def get_all_preds(model,loader):
    all_preds=torch.tensor([])
    for batch in loader:
        images, labels = batch
        preds = model(images)
        all_preds = torch.cat((all_preds,preds),dim=0)
    return all_preds

# Calculate accurancy
with torch.no_grad():
    train_preds = get_all_preds(network,train_loader)
all_labels = tensor_y
preds_correct = get_num_correct(train_preds, all_labels)
print('Total correct:{0}'.format(preds_correct))
print('Accuracy: {0} %'.format(preds_correct*100/len(data[0])))

# Check some samples
print("--------------------- Samples ------------------------")

# Remove samples before
os.system('rm -r {}'.format(infos.displays))
os.system('mkdir {}'.format(infos.displays))
# Draw picture:
indexes = random.sample(range(0,4000),30)

# Function which return the prediction instead of probilities for each class
def get_res_pred(preds):
    return preds.argmax(dim=1)

# Get all predictions
res_preds = get_res_pred(train_preds)

for index in indexes:
    print('Picture num: {0} \t pred: {1} \t labels {2}'.format(index,res_preds[index],data[1][index]))

    # Save spectrums of each sample chose
    pic = data[0][index]
    red_spectrum = pic[0]
    green_spectrum = pic[1]
    blue_spectrum = pic[2]

    plt.figure(index, figsize = (5*3, 5*1))
    plt.set_cmap('jet')
    #show each channel
    plt.subplot(1, 3, 1)
    plt.imshow(red_spectrum)
    plt.subplot(1, 3, 2)
    plt.imshow(green_spectrum)
    plt.subplot(1, 3, 3)
    plt.imshow(blue_spectrum)
    plt.savefig("./{0}/{1}.png".format(infos.displays,index))
    plt.close()
print("--------------------- Samples ------------------------")






