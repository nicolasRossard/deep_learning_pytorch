'''
Ex_net.py
Functions which permit to execute network
'''
import cnn_v5 as cnn
import infos_v5 as infos

import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import time
import tracemalloc
# Function for prediction
## return number of good predictions
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

# Function get predictions for All samples from a DataLoader
## return all predictions of the loader by the model
def get_all_preds(model,loader):
    all_preds=torch.tensor([])
    for batch in loader:
        images, labels = batch
        preds = model(images)
        all_preds = torch.cat((all_preds,preds),dim=0)
    return all_preds
'''
Function launching model
# Arguments:
    - images: dataset
    - labels: labels of the dataset in correct order
    - adam_coef: coefficient of Adam
    -  t_size: size of Test dataset
    -  batch_s: size of batch
    -  nb_epoches: number of epoches for the model
# return:
    - model, dict of results and parameters (epoch, batch, accuracy, size of test, coefficient of Adam, time to execute the model, memories)
'''
def launch_model(images, labels, adam_coef=0.01, t_size=0.2, batch_s=100, nb_epoches=10):    
    print('------------------------------------------------------------------------------------------------\n')
    print('{0}\nAdam coef: {4}\tTrain size: {5}%\tPercentage of test data:{1}\tBatch size: {2}\tNb epoches:{3}'.format(os.path.basename(__file__),t_size,batch_s,nb_epoches,adam_coef, t_size))
    # Split dataset =  % training dataset 20% test_dataset
    X_train, X_test, y_train, y_test = train_test_split(images,labels,test_size=t_size, random_state=11)

    # Transform training and test datasets into PyTorch dataset
    ## Tensors
    tensor_x_train = torch.Tensor(X_train)
    tensor_y_train = torch.Tensor(y_train)
    tensor_x_test = torch.Tensor(X_test)
    tensor_y_test = torch.Tensor(y_test)


    ## Convert labels float type into long type (labels need to be type long)
    tensor_y_train = tensor_y_train.long()
    tensor_y_test = tensor_y_test.long()

    ## Create TensorDataset
    tensorDataset_train = TensorDataset(tensor_x_train, tensor_y_train)
    tensorDataset_test = TensorDataset(tensor_x_test, tensor_y_test)

    ## Create dataloaders
    train_loader = DataLoader(tensorDataset_train,batch_size=batch_s) # batch_s samples / batch
    test_loader = DataLoader(tensorDataset_test,batch_size=batch_s)
    
    # Start timer and save memory capacity
    start = time.time()
    tracemalloc.start()

    # Init model
    network = cnn.CNN()
    optimizer = optim.Adam(network.parameters(), lr=adam_coef)

    # Launch epoches
    for epoch in range(nb_epoches):
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

    # Calculate accurancy for test dataset
    with torch.no_grad():
        test_preds = get_all_preds(network,test_loader)
    all_labels = tensor_y_test
    preds_correct = get_num_correct(test_preds, all_labels)
    print('Total correct:{0}/{1}'.format(preds_correct,len(y_test)))
    accuracy = preds_correct*100/len(y_test)
    timer = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    diff = peak - current
    print('Accuracy: {0} %'.format(accuracy))
    print('------------------------------------------------------------------------------------------------\n')
    return network , {'Epoches': nb_epoches,'Batchs': batch_s,'Accuracies': float("{:.2f}".format(accuracy)), 
            'Test_size':t_size, 'Adam_coef':adam_coef, 'Timer':float("{:.4f}".format(timer)),
            'Mem_current': current, 'Mem_peak':peak, 'Mem_diff':diff}

'''
Function launching model V2 more complete
# Arguments:
    - X_train: training dataset
    - X_test: hold out dataset
    - y_train: training labels
    - y_test: hold out labels
    - adam_coef: coefficient of Adam
    -  t_size: size of Test dataset
    -  batch_s: size of batch
    -  nb_epoches: number of epoches for the model
# return:
    - accuracy, model, optimizer, dict of results and parameters (epoch, batch, accuracy, size of test, coefficient of Adam, loss, time to execute the model, memories)
'''

def launch_model_v2(X_train, X_test, y_train, y_test, adam_coef=0.01, t_size=0.2, batch_s=100, nb_epoches=10):    
    print('------------------------------------------------------------------------------------------------\n')
    print('{0}\nAdam coef: {4}\tTrain size: {5}%\tPercentage of test data:{1}\tBatch size: {2}\tNb epoches:{3}\n'.format(os.path.basename(__file__),t_size,batch_s,nb_epoches,adam_coef, t_size))

    # Transform training and test datasets into PyTorch dataset
    ## Tensors
    tensor_x_train = torch.Tensor(X_train)
    tensor_y_train = torch.Tensor(y_train)
    tensor_x_test = torch.Tensor(X_test)
    tensor_y_test = torch.Tensor(y_test)


    ## Convert labels float type into long type (labels need to be type long)
    tensor_y_train = tensor_y_train.long()
    tensor_y_test = tensor_y_test.long()

    ## Create TensorDataset
    tensorDataset_train = TensorDataset(tensor_x_train, tensor_y_train)
    tensorDataset_test = TensorDataset(tensor_x_test, tensor_y_test)

    ## Create dataloaders
    train_loader = DataLoader(tensorDataset_train,batch_size=batch_s) # batch_s samples / batch
    test_loader = DataLoader(tensorDataset_test,batch_size=batch_s)
    
    # Start timer and save memory capacity
    start = time.time()
    tracemalloc.start()

    # Init model
    network = cnn.CNN()
    optimizer = optim.Adam(network.parameters(), lr=adam_coef)

    # Launch epoches
    for epoch in range(nb_epoches):
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

    # Calculate accurancy for test dataset
    with torch.no_grad():
        test_preds = get_all_preds(network,test_loader)
    all_labels = tensor_y_test
    preds_correct = get_num_correct(test_preds, all_labels)
    print("\nResults on test dataset:")
    print('\tTotal correct:{0}/{1}'.format(preds_correct,len(y_test)))
    accuracy = preds_correct*100/len(y_test)
    timer = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    diff = peak - current
    print('\tAccuracy: {0} %'.format(accuracy))
    print('------------------------------------------------------------------------------------------------\n')
    return accuracy, network, optimizer , {'Epoches': nb_epoches,'Batchs': batch_s,'Accuracies': float("{:.2f}".format(accuracy)), 
            'Test_size':t_size, 'Adam_coef':adam_coef,'Loss': float("{:.2f}".format(total_loss)), 'Timer':float("{:.4f}".format(timer)),
            'Mem_current': current, 'Mem_peak':peak, 'Mem_diff':diff}
