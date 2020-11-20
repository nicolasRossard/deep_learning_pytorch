'''
Infos.py
File which has all informations about:
    Minio server
    Local path
    Bucket names
    Kaggle dataset

Here you can change directories' names files' names...
'''
# Number of label if ships or not ships
nb_ships = 1000
nb_no_ships = 3000

# informations about Minio
ip_address = '192.168.78.1:9000'
access_key = 'AKIAIOSFODNN7EXAMPLE'
secret_key = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'

# Informations about Kaggle data

kaggle_url = 'rhammell/ships-in-satellite-imagery'
json_file = 'shipsnet.json'

# local_path dataset
kaggle_local_path = '/home/minio/dataset/'
clean_local_path = 'cleanData/'
displays = 'displays_v5' # directory where examples are saved
results = './results_v5/'
results_df = './results_v5/results_df_v5.pkl'
# pandas format
clean_data = 'cleanData.pkl'
# Informations about buckets
dir_json = 'ships'
dir_scenes = 'scenes'
dir_images = 'shipsnet'
clean_obj_name ='cleanShips'
images = 'images_v5' # Bucket object where images are saved in Minio after cleaning
labels = 'labels_v5' # Bucket object where labels are saved in Minio after cleaning
scenes = 'scenes_v5'
locations= 'locations_v5'


