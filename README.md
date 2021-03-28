# Deep learning with Pytorch

Create Convolutional Neuronal Network and run it with PyTorch. 
Use dataset from Kaggle : https://www.kaggle.com/rhammell/ships-in-satellite-imagery

Each version made are saved
## Version 3:
Run a CNN on dataset, print some predictions

To execute:

Download kaggle shipsnet.json and save into datas/ ==> datas/shipsnet.json

$ python3 ship_v3.py

## Version 4:
Download dataset from Kaggle

Clean dataset

Save dataset into Minio

Load dataset from Minio

Check if datasets are equals

Create CNN

Train model

Display some samples

To execute:

$ python3 main_v4.py

## Version 5:
Execute several CNN with differents parameters, save performances into a pandas DataFrame, keep the best Model and the best Optimizer. Save all files in local

To execute:

$ python3 main_v5.py

## Version 6:
Load dataset in format PNG (cropped images). Add new data by rotation or by increasing brightness of images

Execute several test with differents parameters for CNN model, save performances into a pandas DF, save the best model and the best optimizer (not included here)

Launch the model on satellite images from Kaggle (main_v6_run.py) with different parameters return satellite images with all ships found (example in v6/execute_model/results/images)
Some other informations are saved too

To eliminate duplicates of ships an other function is used with different parameters (delete_duplicates.py) return satellite images updated (example in v6/execute_model/results/clean_images)


To execute (follow order):

$ python3 train_models/main_v6.py

$ python3 execute_model/main_v6_run.py

$ python3 execute_model/delete_duplicates_v6.py
