'''

'''



import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os.path
import os
#import class CNN
from cnn_v3 import CNN
import random


# Read JSON File

dataset=pd.read_json(open("datas/shipsnet.json","r"))
# ---------------------------------------------------
## Analyze dataset
print(dataset.head())

# Get datas, labels, locations and scene_ids 
x_set=dataset["data"]
y_set=dataset["labels"]
locations = dataset["locations"]
scene_ids = dataset["scene_ids"]

# convert data into numpy array
X=[]
Y=[]

for k in x_set.keys():
    X.append(x_set[k])
    Y.append(y_set[k])
X=np.array(X)
Y=np.array(Y)

# Reshape X into correct image

X=X.reshape(-1,3,80,80)


"""
# Display a sample
if not(os.path.isfile("./displays/ship_draw.png")):
    print("drawing ship_spectrums")
    # Draw picture:
    pic = X[3]
    red_spectrum = pic[0]
    green_spectrum = pic[1]
    blue_spectrum = pic[2]

    plt.figure(2, figsize = (5*3, 5*1))
    plt.set_cmap('jet')
    #show each channel
    plt.subplot(1, 3, 1)
    plt.imshow(red_spectrum)
    plt.subplot(1, 3, 2)
    plt.imshow(green_spectrum)
    plt.subplot(1, 3, 3)
    plt.imshow(blue_spectrum)
    plt.savefig("./displays/ship_draw.png")
else:
    print("ship_draw.png already exists")
"""
# Order correctly X : (index,RGB, row,column)
X=X.transpose(0,1,2,3)
#print(X.shape)

# Reshape labels
Y = Y.reshape(-1)

"""
# Labels
# 0 no ships 1 ships
class_names= ["no-ships","ships"]
class_name_labels = {class_name:i for i,class_name in enumerate(class_names)}
len(y_set==1)
"""

# Preparing Data:
#Shufling
indexes = np.arange(4000)
np.random.shuffle(indexes)

#Pick X_train, Y_train
x_train = X[indexes]
y_train = Y[indexes]

# Normalization
X_train = x_train / 255 # Images are type uint8 with value [0, 255] and we want values vetween [0,1]

## Transform Data into Pytorch Dataset
# Tensor
tensor_x = torch.Tensor(X_train)
tensor_y = torch.Tensor(y_train)
# convert labels type float into type long
tensor_y = tensor_y.long()

# DataLoader for PyTorch
train_set = TensorDataset(tensor_x,tensor_y)
print('Train_set: {0}'.format(train_set))
print('Train_set type: {0}'.format(type(train_set)))

# Function
## return number of good prediction
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

# Initiating model
network = CNN()
train_loader = DataLoader(train_set,batch_size=100) #100 samples / batch

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
print('Accuracy: {0} %'.format(preds_correct*100/len(train_set)))

# Check some samples
print("--------------------- Samples ------------------------")

# Remove samples before
os.system('rm displays/*.png')
# Draw picture:
indexes = random.sample(range(0,4000),30)
def get_res_pred(preds,labels):
    return preds.argmax(dim=1)
res_preds = get_res_pred(train_preds,tensor_y)
for index in indexes:
    print('Picture num: {0} \t pred: {1} \t labels {2}'.format(index,res_preds[index],y_train[index]))
    


    # Save spectrums of each sample chose
    pic = x_train[index]
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
    plt.savefig("./displays/{0}.png".format(index))
    plt.close()
print("--------------------- Samples ------------------------")






