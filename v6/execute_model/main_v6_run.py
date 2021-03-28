import os
from cv2 import cv2
import sys
import torch
from termcolor import colored
from tqdm import tqdm
import pandas as pd
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../train_models')
import cnn_v6 as cnn
#path = 'images/lb_3.png'
import infos_v6 as infos
# -----------------------------------------------------------------  Part 1: Check satellite images are saved and create directories ----------------------------------------------------------------- 
print(colored('\n{}: ----------------- Part 1 Check satellite images are saved and create directories'.format(os.path.basename(__file__)),'red'))
# Dowload dataset on Kaggle
if not(os.path.isdir(infos.kaggle_local_path)):
    kaggle.get_data(infos.kaggle_url,infos.kaggle_local_path)
else:
    print('{}: kaggle files have already downloaded'.format(os.path.basename(__file__)))

# Satellite images path
path = infos.scenes_path

# Create Directories
# Path where dataframe final will be saved
res_path = infos.res_path

# Path where images will be saved
res_path_img = infos.res_path_img

# path where dataframes will be saved
res_path_df = infos.res_path_df
if not os.path.exists(res_path):
    os.makedirs(res_path)
    os.makedirs(res_path_img)
    os.makedirs(res_path_df)
else:
    # Remove all files before
    os.system('rm -r ' + res_path)

    # Create directories
    os.makedirs(res_path_img)
    os.makedirs(res_path_df)

# Path where best model is saved
path_model = infos.model_path
# -----------------------------------------------------------------  Part 1: Check satellite images are saved and create directories ----------------------------------------------------------------- 
print(colored('\n{}: ----------------- Part 2 Launch model on satellite images'.format(os.path.basename(__file__)),'red'))
# Stride chose
strides = [10,20,40,80]
#strides = [80]

# List of images
dirfiles = os.listdir(path)

# Size of cropped images (put on model 80 x 80)
kernel_size=80

# Dataframe final which contains informations for each image: the strides used and the number of ships found
dataframe = pd.DataFrame({'Images': pd.Series([], dtype='string'),
                            'Strides': pd.Series([], dtype='int'),
                            'Ships': pd.Series([], dtype='int')})
# Load best model
model = cnn.CNN()
model.load_state_dict(torch.load(path_model))

# Satellites images loop
for file in tqdm(dirfiles):
    print('*************************************************************************************************************************')
    print(colored("Image = {0}".format(file),'blue'))    
    
    # strides
    for stride in strides:

        df = pd.DataFrame({'Images': pd.Series([], dtype='string'),
                        'Start_col': pd.Series([], dtype='int'),
                        'Start_row': pd.Series([], dtype='int'),
                        'End_col' : pd.Series([], dtype='int'),
                        'End_row': pd.Series([], dtype='int')})

        # Get satellite image path
        img_path = os.path.join(path,file)
        
        # Load image
        img = cv2.imread(img_path)

        #x,y,d = img.shape
        #mod_x = x % stride
        #mod_y = y % stride
        #print('mod_x={0}'.format(mod_x))
        #print('mod_y={0}'.format(mod_y))

        # Rognage de l'image pour avoir un carré divisible par 80
        #img = img[0:x-mod_x, 0:y-mod_y]
        

        img2 = img.copy()
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
        
        # Count index column 
        x = 0
        # Columns
        stop_col = False
        boats = 0
        for i in range(0,img2.shape[0], stride):
            #print('\n')
            #If we arrive at the end of the picture
            if (i+kernel_size) > img2.shape[0]:
                stop_col = True
                col = img2.shape[0] - kernel_size
            else:
                col = i
            # Count index row
            y = 0
            # rows
            stop_row = False
            for j in range(0,img2.shape[1], stride):
                if(j+kernel_size) > img2.shape[1]:
                        stop_row = True
                        row = img2.shape[1] - kernel_size
                else:
                    row = j
                # Save cropped image 80 x 80
                image = img2[col:col+kernel_size,row:row+kernel_size]
                # save image and reshape to [channel, witdh, height]
                image= image.transpose(2,0,1)
             
                # Prepare data to use model
                tensor_img = torch.Tensor(image/255)
                patches_ = tensor_img.unfold(1,kernel_size,stride).unfold(2,kernel_size,stride)
                patches = patches_.contiguous().view(patches_.size(0),-1,kernel_size,kernel_size)
                batch = torch.transpose(patches,0,1)
                
                # Launch model
                output = model(batch)

                # Get the highest probability
                output = torch.argmax(output,1)

                #print('[{0},{1}]\t'.format(i,j),end='')

                # Check if a ship i found
                if output.item() == 1:
                    #print('{0},{1}={2}\t'.format(x,y,output.item()),end='')
                    img = cv2.rectangle(img,(row,col),(row+kernel_size,col+kernel_size),(0,0,255),2)
                    #print(colored('{0},{1}'.format(col,row),'red'))
                    res = {'Images': file, 'Start_col':col,'Start_row':row,'End_col':col+kernel_size,'End_row':row+kernel_size}
                    df = df.append(res, ignore_index = True)
                    boats +=1
                #else:
                    #img = cv2.rectangle(img,(row,col),(row+kernel_size,col+kernel_size),(255,0,0),2)
                    #print(colored('{0},{1}'.format(col,row),'blue'))
                
                y +=1
                # End of the row
                if stop_row:
                    break
            x +=1
            # End of the column
            if stop_col:
                break
        # Save datas information of model use Image stride and number boats found
        datas = {'Images':file, 'Strides':stride, 'Ships':boats}
        # Save informations
        dataframe = dataframe.append(datas,ignore_index = True)
        
        # Get name of Image without extension
        file_name = os.path.splitext(file)[0]
        #print(res_path+file_name+'_'+str(stride)+'_'+model_name+'_res.png')
        
        # Save image into res_path_img
        cv2.imwrite(res_path_img+file_name+'__'+str(stride)+'__res.png',img)
        print(colored('NB of boats = {0},\t stride = {1}'.format(boats, stride),'green')) 
        
        # Save DF wich contains all information about each ships into res_path_df
        df.to_pickle(res_path_df+file_name+'__'+str(stride)+'__ships.pkl', protocol = 2)
            

print(dataframe)
dataframe.to_pickle(res_path+'Donnees.pkl', protocol = 2)
