'''
Infos.py
File which has all informations about:
    Minio server
    Local path
    Bucket names
    Kaggle dataset

Here you can change directories' names files' names...
'''

# informations about Minio
ip_address = '192.168.78.1:9000'
access_key = 'AKIAIOSFODNN7EXAMPLE'
secret_key = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'

# Informations about Kaggle data

kaggle_url = 'rhammell/ships-in-satellite-imagery'
json_file = 'shipsnet.json'

# local_path dataset
kaggle_local_path = '../dataset/'
displays = './displays_v6' # directory where examples are saved
results = './results_v6'
results_df = './results_v6/results_df_v6.pkl'

model_path = '../model.pt' # save the best model
optimizer_path = '../optimizer.pt' # save optimizer parameters

# Informations about buckets
dir_json = 'ships/'
dir_scenes = 'scenes/'
dir_images = 'shipsnet/'

bucket_df = 'dataframes'
obj_names = ['df_images','df_rot','df_bright','df_all']

# Execute model part
res_path = 'results/'
res_path_img ='results/images/'
res_path_df = 'results/df/'
scenes_path = '../dataset/scenes/scenes/'
res_path_final = 'results/clean_images/'
