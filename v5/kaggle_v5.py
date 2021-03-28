'''
Kaggle.py

Download dataset from Kaggle
Delete dataset in local
'''
import infos_v5
import kaggle
import os
import pandas as pd

# Download dataset from Kaggle
def get_data(url,local_path):
    print('{}: get_data function'.format(os.path.basename(__file__)))
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(url, path=local_path, unzip = True)
    print('Ok: dataset from {0} downloaded to {1}'.format(url,local_path)) 

# Delete kaggle dataset or clean dataset in local
def delete(path):
    print('{}: delete function'.format(os.path.basename(__file__)))
    if os.path.exists(path):
        os.system('rm -r {0}'.format(path))
        print('OK: data in path {0} deleted'.format(path))
    else:
        print('ERROR: path {0} does not exist'.format(path))
