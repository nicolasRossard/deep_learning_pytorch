import os
from cv2 import cv2
import sys
from termcolor import colored
import pandas as pd
sys.path.insert(1, '../train_models')
import infos_v6 as infos
# COULD BE MODIFIER
# Size to consider that 2 square represent the same ship
epsilons = [40, 80, 160]


# Path contents original images ../dataset/scenes/scenes
image_path =infos.scenes_path 

# Path where dataframe final will be saved 
res_path = infos.res_path

# Path where images will be saved ./results/images
res_path_img = infos.res_path_img

# path where dataframes will be saved ./resulta/df
res_path_df = infos.res_path_df

# path where final images will be saved ./results/clean_images
res_path_final = infos.res_path_final

if not os.path.exists(res_path_final):
    os.makedirs(res_path_final)
else:
    # Remove all files before
    os.system('rm -r ' + res_path_final)

    # Create directories
    os.makedirs(res_path_final)
 
# Size of the image
kernel_size = 80 

# Save info program
orig_stdout = sys.stdout 
f = open(infos.res_path+"delete_duplicates_run.txt","w")
sys.stdout = f

# Loop for each epsilon
for epsilon in epsilons:

    # List of df
    dirfiles = os.listdir(res_path_df)

    # loop for each dataframe 
    for file in dirfiles:
        # Get all information with name of file
        name_img,stride,ext = file.split('__')
        print('name = {0}, stride = {1} epsilon = {2}'.format(name_img,stride, epsilon))
        
        # Get satellite image name
        img_name = name_img+'.png'

        # Get df path
        path_pkl = os.path.join(res_path_df,file)
        # Get df
        file_df = pd.read_pickle(path_pkl)

        # Get ships for the corresponding satellite image
        dataframe = file_df[file_df['Images']== img_name]

        print('NB of ships  = {}'.format(len(dataframe.index)))

        # Sort values by start_col then start row
        df = dataframe.sort_values(by = ['Start_col','Start_row'], ascending = [True,True])
        
        i = 0

        # Loop all ships found and detect duplicate starting with columns index
        while (i+1) < len(df.index):

            # Look if 2 ships are closed with columns index
            if abs(df.iloc[i]['Start_col'] - df.iloc[i+1]['Start_col']) < epsilon:
            
                # Check if these 2 ships are closed also with  rows index
                if abs(df.iloc[i]['Start_row'] - df.iloc[i+1]['Start_row']) < epsilon:
                    
                    # if both conditions are True it means that the 2 ships found are really closed (< epsilon). We suppose it is the same ship
                    # Save only one ship (average)
                    s_col = round((df.iloc[i]['Start_col'] + df.iloc[i+1]['Start_col'])/2)
                    s_row = round((df.iloc[i]['Start_row'] + df.iloc[i+1]['Start_row'])/2) 
                    e_col = s_col + kernel_size
                    e_row = s_row + kernel_size

                    # delete rows of old squares
                    df = df.drop(df.iloc[i+1].name)
                    df = df.drop(df.iloc[i].name)
                    
                    dictionary = {'Images':img_name, 'Start_col':s_col, 'Start_row':s_row, 'End_col':e_col, 'End_row':e_row}
                    # add average of 2 squares
                    df = df.append(dictionary, ignore_index=True)
                    # Sort dataframe
                    df = df.sort_values(by = ['Start_col','Start_row'], ascending = [True,True])
                   
                    # restart
                    i = 0
                else:
                    i+=1
            else:
                i+=1


        # Sort values by start_row then start_col
        df = df.sort_values(by = ['Start_row','Start_col'], ascending = [True,True])

        i = 0
        # Loop all ships found and detect duplicate starting with rows index
        while (i+1) < len(df.index):

            # Look if 2 ships are closed with rows index
            if abs(df.iloc[i]['Start_row'] - df.iloc[i+1]['Start_row']) < epsilon:
            
                # Check if these 2 ships are closed also with  columnss index
                if abs(df.iloc[i]['Start_col'] - df.iloc[i+1]['Start_col']) < epsilon:
                    
                    # if both conditions are True it means that the 2 ships found are really closed (< epsilon). We suppose it is the same ship
                    # Save only one ship (average)
                    s_row = round((df.iloc[i]['Start_row'] + df.iloc[i+1]['Start_row'])/2)
                    s_col = round((df.iloc[i]['Start_col'] + df.iloc[i+1]['Start_col'])/2) 
                    e_col = s_col + kernel_size
                    e_row = s_row + kernel_size
                    
                    dictionary = {'Images':img_name, 'Start_col':s_col, 'Start_row':s_row, 'End_col':e_col, 'End_row':e_row}
                    
                    # delete rows of old squares
                    df = df.drop(df.iloc[i+1].name)
                    df = df.drop(df.iloc[i].name)
                   
                   # add average of 2 squares
                    df = df.append(dictionary, ignore_index=True)
                    
                    # Sort dataframe
                    df = df.sort_values(by = ['Start_row','Start_col'], ascending = [True,True])
                    
                    # restart
                    i = 0
                else:
                    i+=1
            else:
                i+=1

        # Sort correctly final results
        df = df.sort_values(by = ['Start_col','Start_row'], ascending = [True,True])

        # Load satellite image
        img_path = os.path.join(image_path,img_name)
        img = cv2.imread(img_path)
        
        # Display ships
        ships = 0
        for index, row in df.iterrows():
            s_row = row['Start_row']
            e_row = row['End_row']
            s_col = row['Start_col']
            e_col = row['End_col']
            # Display square
            img = cv2.rectangle(img,(s_row,s_col),(e_row,e_col),(0,0,255),2)
            ships+=1
        print('NB of ships after filter = {}'.format(ships))
        img_name = os.path.splitext(img_name)[0]
        final_name = img_name+'__'+stride+'__'+str(epsilon)+'__cleaned.png'
        final_path = os.path.join(res_path_final,final_name)
        # Save satellite image with ships
        cv2.imwrite(final_path,img)
        print('----------------------------------------------\n')
sys.stdout = orig_stdout
f.close()


