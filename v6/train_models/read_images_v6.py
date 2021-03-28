import kaggle_v6 as kaggle
import minio
import minioClient_v6 as minioClient
import ex_net_v6 as ex_net
import clean_v6 as clean
import cnn_v6 as cnn
import infos_v6 as infos
import os
import pandas as pd
from tqdm import tqdm
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

import pickle
import io

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def get_dataset(dataset_path):    
    '''
    Load all files from path parameter
    Get labels, scenes ID and locations of each picture
    Convert pictures PNG into RGB
    return numpy array of : labels (1D), scenes_id (1D), locations (2d), images numpy array( [nb_picutres,RGB,witdh,length])
    '''
    
    print('{}: get_dataset function'.format(os.path.basename(__file__)))
    
    #Get all files into the path
    dirfiles = os.listdir(dataset_path)
    
    # Init lists
    # labels
    labs = []

    # Scenes ID
    ids = []

    # Location
    locs = []
    
    # Pictures in RGB
    imgs = []

    for file in tqdm(os.listdir(dataset_path)):
            
        # Get all information with name of file
        label,scene_id, coord = file.split('__')

        # Split longitude, latitude & delete '.png'
        longitude, latitude = coord[:-4].split('_')
        
        # Save into list
        labs.append(label)
        ids.append(scene_id)
        locs.append([longitude, latitude])
        
        # Get image path
        img_path = os.path.join(dataset_path, file)
        
        # Convert image
        img = cv2.imread(img_path)
        
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # -------------------------------------------------
        ## 1. Save normal picture
        # Get RGB
        red, green, blue = cv2.split(img)
        
        # Save picture
        imgs.append([red,green,blue])

        # -------------------------------------------------

    # Convert list to Numpy
    '''
    labels = np.array(labs,dtype = np.int)
    scenes_id = np.array(ids)
    locations = np.array(locs)
    images = np.array(imgs,dtype=np.int)
    '''
    # Convert to panda.Series
    scene_series = pd.Series(ids)
    loc_series = pd.Series(locs)
    lab_series = pd.Series(labs, dtype = np.int)
    datas_series = pd.Series(imgs)

    # Create DataFrame
    frame = { 'id': scene_series, 'loc': loc_series, 'lab': lab_series, 'datas':datas_series } 
    df = pd.DataFrame(frame)

    return df

def rotate_dataset(dataframe, rotation):
    '''
    arguments:
        df with cols =  ['id', 'loc', 'lab', 'datas']
        rotation = cv2.ROTATION_TYPE
    rotate the picture
    return df with same columns and image rotated in 'data'
    '''
    print('{}: rotate_dataset function'.format(os.path.basename(__file__)))
    
    rots = []
    for img in dataframe['datas']:
        red = img[0]
        blue = img[1]
        green = img[2]
        
        # convert picture
        image = cv2.merge((red, blue, green))
        
        # Rotate
        rot = cv2.rotate(image, rotation) 
        
        # Get RGB
        r_rot, g_rot, b_rot = cv2.split(rot)
        
        #Save
        rots.append([r_rot, g_rot, b_rot])

    # Convert all into pandas Series
    rot_series = pd.Series(rots)
    scene_series = pd.Series(dataframe['id'])
    loc_series = pd.Series(dataframe['loc'])
    lab_series = pd.Series(dataframe['lab'], dtype = np.int)

    # Create DataFrame
    frame = { 'id': scene_series, 'loc': loc_series, 'lab': lab_series, 'datas':rot_series } 
    df = pd.DataFrame(frame)

    return df

def up_brightness_dataset(dataframe, val):
    ''''
    arguments:
        df with cols =  ['id', 'loc', 'lab', 'datas']
        val = brightness value (integer)
    uncreease the brightness of the the picture
    return df with same columns and image with its brighness increased in 'data'
    '''
    
    print('{}: up_brightness_dataset function'.format(os.path.basename(__file__)))
    
    datas = []
    for img in dataframe['datas']:
        red = img[0]
        blue = img[1]
        green = img[2]
        
        # convert picture
        image = cv2.merge((red, blue, green))
        img_hsv = increase_brightness(image, value = val)
        # Split againt
        r_hsv, g_hsv, b_hsv = cv2.split(img_hsv)

        # Save
        datas.append([r_hsv, g_hsv, b_hsv])
 
    # Convert all into pandas Series
    datas_series = pd.Series(datas)
    scene_series = pd.Series(dataframe['id'])
    loc_series = pd.Series(dataframe['loc'])
    lab_series = pd.Series(dataframe['lab'], dtype = np.int)

    # Create DataFrame
    frame = { 'id': scene_series, 'loc': loc_series, 'lab': lab_series, 'datas':datas_series } 
    df = pd.DataFrame(frame)

    return df

# Function not used
def load_dataset(dataset_path):    
    '''
    Load all files from path parameter
    Get labels, scenes ID and locations of each picture
    Convert pictures PNG into RGB
    Save image, rotate image and image with birghtness increased
    return numpy array of : labels (1D), scenes_id (1D), locations (2d), images numpy array( [nb_picutres,RGB,witdh,length]), turned image (rotations)  numpy array( [nb_picutres,RGB,witdh,length]),
    images with more brightness (hsv_array) numpy array( [nb_picutres,RGB,witdh,length])

    '''
    
    print('{}: load_dataset function'.format(os.path.basename(__file__)))
    
    #Get all files into the path
    dirfiles = os.listdir(dataset_path)
    
    # Init lists
    # labels
    labs = []

    # Scenes ID
    ids = []

    # Location
    locs = []
    
    # Pictures in RGB
    imgs = []

    # Rotate pictures in RGB
    rots = []
    
    # Values of HSV modify
    hsvs = []
    first = True
    for file in tqdm(os.listdir(dataset_path)):
            
        # Get all information with name of file
        label,scene_id, coord = file.split('__')

        # Split longitude, latitude & delete '.png'
        longitude, latitude = coord[:-4].split('_')
        
        # Save into list
        labs.append(label)
        ids.append(scene_id)
        locs.append([longitude, latitude])
        
        # Get image path
        img_path = os.path.join(dataset_path, file)
        
        # Convert image
        img = cv2.imread(img_path)
        
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # -------------------------------------------------
        ## 1. Save normal picture
        # Get RGB
        red, green, blue = cv2.split(img)
        
        # Save picture
        imgs.append([red,green,blue])

        # -------------------------------------------------
        ## 2. Modify picture with HSV value
        # Increase brightness
        img_hsv = increase_brightness(img, value = 20)
        
        # Split
        r_hsv, g_hsv, b_hsv = cv2.split(img_hsv)

        # Save
        hsvs.append([r_hsv, g_hsv, b_hsv])

        # -------------------------------------------------
        ## 3. Rotate picture
        # Rotation
        rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # Get RGB
        r_rot, g_rot, b_rot = cv2.split(rot)
        
        #Save
        rots.append([r_rot, g_rot, b_rot])
        
    # Convert list to Series
    labels = np.array(labs,dtype = np.int)
    scenes_id = np.array(ids)
    locations = np.array(locs)
    images = np.array(imgs,dtype=np.int)
    rotations = np.array(rots,dtype=np.int)
    hsv_array = np.array(hsvs, dtype = np.int) 

    return labels, scenes_id, locations, images, rotations, hsv_array




if __name__ == '__main__':
    print("----{}: Execute file".format(os.path.basename(__file__))) 
    
    # path where pictures are saved
    ships_path = infos.kaggle_local_path + infos.dir_images + infos.dir_images
   
    # load datasets + add 4000 pictures which are rotated and 4000 pictures where brightness are incresed
    labels, scenes_id, locations, images, rotations, hsvs_array = load_dataset(ships_path)
    
    # Load dataset with normal picture ==> DF
    df = get_dataset(ships_path)
 
    print(df.shape)
    print(df.head())
    print()
    print()
   
    # Rotate image
    df_rot = rotate_dataset(df, cv2.ROTATE_90_CLOCKWISE)
    print(df_rot.shape)
    print(df_rot.head())
    print()
    print()
    

    df_bright = up_brightness_dataset(df, 20)
    print(df_bright.shape)
    print(df_bright.head())
    print()
    print()

    # ------------------------------------------------------------------------------------------------- #
    # Save dataframes into Minio with pickle
    minioCl = minioClient.init_minio(infos.ip_address,infos.access_key, infos.secret_key)
   
    # Do not forget to delete objects if using this code several times
    # Save DF
    bytes_file = pickle.dumps(df)
    minioCl.put_object(
            bucket_name = infos.bucket_df,
            object_name = infos.obj_names[0],
            data = io.BytesIO(bytes_file),
            length = len(bytes_file)
        )
    
    # Save DF rotations
    bytes_file = pickle.dumps(df_rot)
    minioCl.put_object(
            bucket_name = infos.bucket_df,
            object_name = infos.obj_names[1],
            data = io.BytesIO(bytes_file),
            length = len(bytes_file)
        )

    # Save DF bright
    bytes_file = pickle.dumps(df_bright)
    minioCl.put_object(
            bucket_name = infos.bucket_df,
            object_name = infos.obj_names[2],
            data = io.BytesIO(bytes_file),
            length = len(bytes_file)
        )

    # Load dataframes from Minio
    """ 
    df_ = pickle.loads(minioCl.get_object(bucket_name=infos.bucket_df, object_name = infos.obj_names[0]).read())
    df_rot_ = pickle.loads(minioCl.get_object(bucket_name=infos.bucket_df, object_name = infos.obj_names[1]).read())
    df_bright_ = pickle.loads(minioCl.get_object(bucket_name=infos.bucket_df, object_name = infos.obj_names[2]).read())
    

    print("df")
    print(df_.head())
    print()
    
    print("df_rot")
    print(df_rot_.head())
    print()

    print("df_bright")
    print(df_bright_.head())

    """
