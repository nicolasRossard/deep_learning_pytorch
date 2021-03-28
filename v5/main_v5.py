'''
Main.py
Main file which using clean_v*, infos_v*, kaggle_v*, cnn_v*, ex_net_v* files
Download dataset from Kaggle
Clean dataset
Save dataset into Minio
Load dataset fom Minio
Check if datasets are equal
Create several models
Train models
Save all models performances into a panda Dataframe (accuracy, execution time, memory loss etc.)
Save DF into Minio
Save best model and best optimizer into Minio
'''
import ex_net_v5 as ex_net
import clean_v5 as clean
import cnn_v5 as cnn
import infos_v5 as infos
import kaggle_v5 as kaggle
import minio
import minioClient_v5 as minioClient
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
# init minio

minioCl = minioClient.init_minio(infos.ip_address,infos.access_key, infos.secret_key)
'''
#----------------------------------------------------------------- Part 1: Load data from Kaggle, clean, put in Minio delete localfiles ----------------------------------------------------------------- 
# Delete all directories already made
if os.path.isdir(infos.kaggle_local_path):
    kaggle.delete(infos.kaggle_local_path)
else:
    print('{}: kaggle files have not been downloaded yet'.format(os.path.basename(__file__)))

# get dataset from kaggle and save into local dir
if not(os.path.isdir(infos.kaggle_local_path)):
    kaggle.get_data(infos.kaggle_url,infos.kaggle_local_path)
else:
    print('{}: kaggle files have already downloaded'.format(os.path.basename(__file__)))

# clean data_set and get numpy arrays [x_tain, y_train, localisations, scenes] but not shuffle
dataset = clean.clean_data('{}shipsnet.json'.format(infos.kaggle_local_path), False)

# Save clean_data into minio
clean.save_data_minio(minioCl,dataset[0],infos.dir_json, infos.images)
clean.save_data_minio(minioCl,dataset[1],infos.dir_json, infos.labels)
clean.save_data_minio(minioCl,dataset[2],infos.dir_json, infos.locations)
clean.save_data_minio(minioCl,dataset[3],infos.dir_json, infos.scenes)

# Delete all local files
kaggle.delete(infos.kaggle_local_path)
'''
# ----------------------------------------------------------------- Part 2: Load data from minio ----------------------------------------------------------------- 
images = clean.load_data_minio(minioCl,infos.dir_json, infos.images) # images.flags['WRITEABLE'] = False can't be used in PyTorch
labels = clean.load_data_minio(minioCl,infos.dir_json, infos.labels)
locations = clean.load_data_minio(minioCl,infos.dir_json, infos.locations)
scenes = clean.load_data_minio(minioCl,infos.dir_json, infos.scenes)

'''
# Check if datasets are equals
print('Images array is equal ? : {}'.format(np.array_equal(dataset[0], images)))
print('Labels array is equal ? : {}'.format(np.array_equal(dataset[1], labels)))
print('Locations array is equal ? : {}'.format(np.array_equal(dataset[2], locations)))
print('Scenes array is equal ? : {}'.format(np.array_equal(dataset[3], scenes)))
'''

# -----------------------------------------------------------------  Part 3: Launch models ----------------------------------------------------------------- 
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
batchs = [50,100,150,200]
# Epoches' list
epoches = [2,5,10,12]
# Parameters' list for optimizer
optims = [0.001, 0.01, 0.1]

# Size of test dataset
test_sizes = [0.20, 0.30, 0.40]

# Run all networks:
acc_final = 0
for t in test_sizes:
    # Run with exactly the same training set and test set
    X_train, X_test, y_train, y_test = train_test_split(images,labels,test_size=t, random_state=11)
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

print('\n')

# -----------------------------------------------------------------  Part 4: Display results ----------------------------------------------------------------- 
# Display DF
print(df)
print('\n')

# Save results put protocol = 2 to be able to load it on Jupyter Notebook
df.to_pickle('{0}'.format(infos.results_df), protocol = 2)

# Display best model parameters
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
torch.save(final_model.state_dict(),'{0}{1}'.format(infos.results,final_model_name))
torch.save(final_optimizer.state_dict(), '{0}{1}'.format(infos.results,final_optim_name))

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
