import ex_net_v6 as ex_net
#import clean_v6 as clean
import cnn_v6 as cnn
import infos_v6 as infos
import kaggle_v6 as kaggle
import read_images_v6 as Images
import minio
import minioClient_v6 as minioClient
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
from sklearn.model_selection import train_test_split
import torch
import sys
from cv2 import cv2
import io
from termcolor import colored
from tqdm import tqdm
# init minio
minioCl = minioClient.init_minio(infos.ip_address,infos.access_key, infos.secret_key)
# Create bucket
# get dataset from kaggle and save into local dir
if not(os.path.isdir(infos.kaggle_local_path)):
    kaggle.get_data(infos.kaggle_url,infos.kaggle_local_path)
else:
    print('{}: kaggle files have already downloaded'.format(os.path.basename(__file__)))


# -----------------------------------------------------------------  Part 0: Save datasets into Minio with dataframes ----------------------------------------------------------------- 
print(colored('\n{}: ----------------- Part 0 Save datasets into Minio with dataframes'.format(os.path.basename(__file__)),'red'))
if minioCl.bucket_exists(infos.bucket_df):
    print('{}: kaggle bucket with dataframes already exists'.format(os.path.basename(__file__)))
else:
    minioCl.make_bucket(infos.bucket_df)
    # Run read_dataset.py
    print("Launch read_images_v6.py")
    os.system("python3 read_images_v6.py")

# -----------------------------------------------------------------  Part 1: Load datasets ----------------------------------------------------------------- 
print(colored('\n{}: ----------------- Part 1 Load datasets'.format(os.path.basename(__file__)),'red'))

# Load datasets with Pickle
# Get classic images
df_images = pickle.loads(
        minioCl.get_object(
            bucket_name= infos.bucket_df,
            object_name = infos.obj_names[0]).read())

# Get rotated images
df_rot = pickle.loads(
        minioCl.get_object(
            bucket_name=infos.bucket_df,
            object_name = infos.obj_names[1]).read())

# Get images where brightness are increased
df_bright = pickle.loads(
        minioCl.get_object(
            bucket_name=infos.bucket_df,
            object_name = infos.obj_names[2]).read())
# -----------------------------------------------------------------  Part 2: Prepare datasets ----------------------------------------------------------------- 
print(colored('\n{}: ----------------- Part 2 Prepare datasets'.format(os.path.basename(__file__)),'red'))

# Get ships only
df_ships_img = df_images[df_images['lab']==1]
df_ships_rot = df_rot[df_rot['lab']==1]
df_ships_br = df_bright[df_bright['lab']==1]

# Get the rest only
df_no_img = df_images[df_images['lab']==0]
df_no_rot = df_rot[df_rot['lab']==0]
df_no_br = df_bright[df_bright['lab']==0]

del df_images
del df_rot
del df_bright

## SHIPS PART
# Get lab for each dataset which are a ship
lab_ships_img = df_ships_img['lab'].to_numpy()
lab_ships_img = lab_ships_img.reshape(-1)

lab_ships_rot = df_ships_rot['lab'].to_numpy()
lab_ships_rot = lab_ships_rot.reshape(-1)

lab_ships_br = df_ships_br['lab'].to_numpy()
lab_ships_br = lab_ships_br.reshape(-1)

# Get all images corresponding
img_ships_img = df_ships_img['datas']
img_ships_img = np.array([np.array(xi) for xi in img_ships_img])
img_ships_img = img_ships_img/255

img_ships_rot = df_ships_rot['datas']
img_ships_rot = np.array([np.array(xi) for xi in img_ships_rot])
img_ships_rot = img_ships_rot/255

img_ships_br = df_ships_br['datas']
img_ships_br = np.array([np.array(xi) for xi in img_ships_br])
img_ships_br = img_ships_br/255

#ALloc memory
del df_ships_img
del df_ships_rot
del df_ships_br

## NO SHIPS PART
# Get lab for each dataset which are not a ship
lab_no_img = df_no_img['lab'].to_numpy()
lab_no_img = lab_no_img.reshape(-1)

lab_no_rot = df_no_rot['lab'].to_numpy()
lab_no_rot = lab_no_rot.reshape(-1)

lab_no_br = df_no_br['lab'].to_numpy()
lab_no_br = lab_no_br.reshape(-1)

# Get all images corresponding
img_no_img = df_no_img['datas']
img_no_img = np.array([np.array(xi) for xi in img_no_img])
img_no_img = img_no_img/255

img_no_rot = df_no_rot['datas']
img_no_rot = np.array([np.array(xi) for xi in img_no_rot])
img_no_rot = img_no_rot/255

img_no_br = df_no_br['datas']
img_no_br = np.array([np.array(xi) for xi in img_no_br])
img_no_br = img_no_br/255

#Alloc memory
del df_no_img
del df_no_rot
del df_no_br



print("img_ships_img shape = {0}".format(img_ships_img.shape))
print("lab_ships_img shape = {0}".format(lab_ships_img.shape))
print()
print("img_no_img shape = {0}".format(img_no_img.shape))
print("lab_no_img shape = {0}".format(lab_no_img.shape))

print()
print("img_ships_rot shape = {0}".format(img_ships_rot.shape))
print("lab_ships_rot shape = {0}".format(lab_ships_rot.shape))
print()

print("img_no_rot shape = {0}".format(img_no_rot.shape))
print("lab_no_rot shape = {0}".format(lab_no_rot.shape))
print()

print("img_ships_br shape = {0}".format(img_ships_br.shape))
print("lab_ships_br shape = {0}".format(lab_ships_br.shape))
print()

print("img_no_br shape = {0}".format(img_no_br.shape))
print("lab_no_br shape = {0}".format(lab_no_br.shape))

print()

# With img and rot pictures
img_ships = np.concatenate((img_ships_img,img_ships_rot,img_ships_br),axis=0)
lab_ships = np.concatenate((lab_ships_img,lab_ships_rot, lab_ships_br),axis=0)


img_no = np.concatenate((img_no_img,img_no_rot, img_no_br),axis=0)
lab_no = np.concatenate((lab_no_img,lab_no_rot, lab_no_br),axis=0)

'''
# With img and rot pictures
img_ships = np.concatenate((img_ships_img,img_ships_rot),axis=0)
lab_ships = np.concatenate((lab_ships_img,lab_ships_rot),axis=0)


img_no = np.concatenate((img_no_img,img_no_rot),axis=0)
lab_no = np.concatenate((lab_no_img,lab_no_rot),axis=0)
'''
'''
# Test with pictures at the begining
img_ships = img_ships_img.copy()
lab_ships = lab_ships_img.copy()
img_no = img_no_img.copy()
lab_no = lab_no_img.copy()

'''

print("img_ships shape = {0}".format(img_ships.shape))
print("lab_ships shape = {0}".format(lab_ships.shape))

print("img_no shape = {0}".format(img_no.shape))
print("lab_no shape = {0}".format(lab_no.shape))


del img_ships_img
del img_ships_rot
del img_ships_br

del lab_ships_img
del lab_ships_rot
del lab_ships_br

del img_no_img
del img_no_rot
del img_no_br

del lab_no_img
del lab_no_rot
del lab_no_br

# -----------------------------------------------------------------  Part 3: Run models ----------------------------------------------------------------- 
print(colored('\n{}: ----------------- Part 3 run models'.format(os.path.basename(__file__)),'red'))


# Create DataFrame to stock results
df = pd.DataFrame({'Epoches': pd.Series([], dtype='int'),
                    'Batchs': pd.Series([], dtype='int'),
                    'Accuracies': pd.Series([], dtype='float'),
                    'Test_size' : pd.Series([], dtype='float'),
                    'Adam_coef': pd.Series([], dtype='float'),
                    'Loss': pd.Series([], dtype = 'float'),
                    'Timer':pd.Series([], dtype ='float'),
                    'Mem_current': pd.Series([], dtype = 'int'),
                    'Mem_peak': pd.Series([], dtype = 'int'),
                    'Mem_diff':pd.Series([], dtype = 'int')})


# Initiating parameters for each model

# Batchs' list
batchs = [200,300,600]
#batchs = [500]
# Epoches' list
epoches = [2,5,7,10]
#epoches = [2]
# Parameters' list for optimizer
optims = [0.001, 0.01, 0.1]
#optims = [0.001]

# Size of test dataset
test_sizes = [0.20, 0.30, 0.40]
#test_sizes = [0.20,0.30]

# Run all networks:
acc_final = 0

# Save all informations:
orig_stdout = sys.stdout 
f = open(infos.results +"/main_v6.txt","w")
sys.stdout = f
for t in tqdm(test_sizes):
    # Run with exactly the same training set and test set
    X_train_ships, X_test_ships, y_train_ships, y_test_ships = train_test_split(img_ships,lab_ships,test_size=t, random_state=11)
    X_train_no, X_test_no, y_train_no, y_test_no = train_test_split(img_no,lab_no,test_size=t, random_state=11)
    #print(type(X_train_ships.shape[0])) 
    #print(X_train_ships.shape[0] + X_train_no.shape[0])
    #print(y_test_ships.shape[0] + y_test_no.shape[0])
    
    # Get size of train dataset and test dataset
    index_train = np.arange(X_train_ships.shape[0] + X_train_no.shape[0])
    index_test = np.arange(y_test_ships.shape[0] + y_test_no.shape[0])
    print("index_train = {0}".format(index_train.shape))
    print("index_test = {0}".format(index_test.shape))
    # Shuffle dataset
    np.random.shuffle(index_train)
    np.random.shuffle(index_test)

    #Merger ships and no ships together for each dataset
    #X_train = np.concatenate((X_train_ships,X_train_no),axis=0)[index_train]
    X_train = np.concatenate((X_train_ships,X_train_no),axis=0)
    del X_train_ships
    del X_train_no

    #X_test = np.concatenate((X_test_ships,X_test_no),axis=0)[index_test]
    X_test = np.concatenate((X_test_ships,X_test_no),axis=0)
    del X_test_ships
    del X_test_no

    #y_train = np.concatenate((y_train_ships,y_train_no),axis=0)[index_train]
    y_train = np.concatenate((y_train_ships,y_train_no),axis=0)
    del y_train_ships
    del y_train_no

    #y_test = np.concatenate((y_test_ships,y_test_no),axis=0)[index_test]
    y_test = np.concatenate((y_test_ships,y_test_no),axis=0)
    del y_test_ships
    del y_test_no

    print("X_train shape = {0}".format(X_train.shape))
    print("y_train shape = {0}".format(y_train.shape))
    print()
    print()
    print("X_test shape = {0}".format(X_test.shape))
    print("y_test shape = {0}".format(y_test.shape))
    """
    print()
    print(y_train)
    print()
    print()
    print(y_test)
    print()
    """
    X_train = X_train[index_train]
    y_train = y_train[index_train]
    X_test = X_test[index_test]
    y_test = y_test[index_test]
    
    for e in epoches:
        for b in batchs:
            for o in optims:
                
                # Launch Model and get accuracy, model, optimizer and results into dict
                acc, model, optimizer  ,res = ex_net.launch_model_v2(X_train, X_test, y_train, y_test, o, t, b, e)
               
                # Save best model
                if acc > acc_final:
                
                    # Save model
                    name = '{0}_{1}_{2}_{3}'.format(res.get('Epoches'), res.get('Batchs'), int(res.get('Test_size')*100), str(res.get('Adam_coef')).replace('.',''))
                    model_name = 'Model_' + name + '.pt'
                    final_model_name = model_name
                    final_model = model
                    
                    # Save optimizer
                    final_optim_name = 'Optim_' + name + '.pt'
                    final_optimizer = optimizer

                    # Save accuracy
                    print('***********************************\nAcc = {0} % > {1} %\t NEW MODEL SAVED: {2}\n***********************************'.format(acc,acc_final, model_name))
                    acc_final = acc

                # Append results into DF    
                df = df.append(res, ignore_index = True)

    del X_train
    del X_test
    del y_train
    del y_test

# -----------------------------------------------------------------  Part 4: Display results ----------------------------------------------------------------- 
print(colored('\n{}: ----------------- Part 4 Display results'.format(os.path.basename(__file__)),'red'))

# Display DF
print(df)
print('\n')


# Check if results directory exists
if os.path.isdir(infos.results):
    # Remove results before
    print()
    #os.system('rm '+ infos.results +'/*')
else:
    # Create the directory
    os.system('mkdir '+infos.results)


# Save results put protocol = 2 to be able to load it on Jupyter Notebook
df.to_pickle('{0}'.format(infos.results_df), protocol = 2)

# Display best model parameters
print("Best Accuracy:")
print(df.iloc[df['Accuracies'].argmax()])

print('\n')

# Display name of model and optimizer saved
print(final_model_name)
print(final_optim_name)

print('\n')

# Display parameters of model
print("Model's state_dict:")
for param_tensor in final_model.state_dict():
        print(param_tensor, "\t", final_model.state_dict()[param_tensor].size())

print('\n')
'''
# Display parameters of optimizer
print("Optimizer's state_dict:")
for var_name in final_optimizer.state_dict():
    print(var_name,"\t", final_optimizer.state_dict()[var_name])
'''
# Save model
torch.save(final_model.state_dict(),'{0}/{1}'.format(infos.results,final_model_name))
torch.save(final_optimizer.state_dict(), '{0}/{1}'.format(infos.results,final_optim_name))
# Save model in root directory
torch.save(final_model.state_dict(),'{0}'.format(infos.model_path))
torch.save(final_optimizer.state_dict(), '{0}'.format(infos.optimizer_path))


'''
## Checking if model and optimizer are saved correctly
# Load model saved
model = cnn.CNN()
model.load_state_dict(torch.load('{0}{1}'.format(infos.results,final_model_name)))
print(model.eval())

# Load optim saved
optimizer = optim.Adam(model.parameters())
optimizer.load_state_dict(torch.load('{0}{1}'.format(infos.results,final_optim_name)))
'''

#sys.stdout.close()
sys.stdout = orig_stdout
f.close()










