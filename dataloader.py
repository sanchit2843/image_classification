import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
from tqdm.autonotebook import tqdm
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from util import class_plot
#data loader

def data_loader(train_data,encoder,test_data = None,valid_data = None , valid_size = None,test_size = None , batch_size = 32,inv_normalize = None):
    class_plot(train_data,encoder,inv_normalize)
    if(test_data == None and valid_size == None and valid_data == None and test_size == None):
        train_loader =  DataLoader(train_data, batch_size = batch_size , shuffle = True)
        dataloaders = {'train':train_loader}
        return dataloaders
    if(test_data == None and valid_size == None and valid_data != None and test_size == None):
        train_loader =  DataLoader(train_data, batch_size = batch_size , shuffle = True)
        valid_loader = DataLoader(valid_data,batch_size = batch_size,shuffle = True)
        dataloaders = {'train':train_loader,'val':valid_loader}
        return dataloaders

    if(test_data !=None and valid_size==None and valid_data == None):
        test_loader = DataLoader(test_data, batch_size= batch_size,shuffle = True)
        train_loader =  DataLoader(train_data, batch_size = batch_size , shuffle = True)

        dataloaders = {'train':train_loader,'test':test_loader}

    if(test_data == None and valid_size!=None and valid_data == None):
        if(test_size==None):
            data_len = len(train_data)
            indices = list(range(data_len))
            np.random.shuffle(indices)
            split1 = int(np.floor(valid_size * data_len))
            valid_idx , train_idx = indices[:split1], indices[split1:]
            valid_sampler = SubsetRandomSampler(valid_idx)
            train_sampler = SubsetRandomSampler(train_idx)
            valid_loader = DataLoader(train_data, batch_size= batch_size, sampler=valid_sampler)
            train_loader =  DataLoader(train_data, batch_size = batch_size , sampler=valid_sampler)
            dataloaders = {'train':train_loader,'val':valid_loader}
            return dataloaders
        if(test_size !=None):
            data_len = len(train_data)
            indices = list(range(data_len))
            np.random.shuffle(indices)
            split1 = int(np.floor(valid_size * data_len))
            split2 = int(np.floor(test_size * data_len))
            valid_idx , test_idx,train_idx = indices[:split1], indices[split1:split1+split2],indices[split1+split2:]
            valid_sampler = SubsetRandomSampler(valid_idx)
            test_sampler = SubsetRandomSampler(test_idx)
            train_sampler = SubsetRandomSampler(train_idx)
            valid_loader = DataLoader(test_data, batch_size= batch_size, sampler=valid_sampler)
            test_loader = DataLoader(test_data, batch_size= batch_size, sampler=test_sampler)
            train_loader =  DataLoader(train_data, batch_size = batch_size , sampler=valid_sampler)
            dataloaders = {'train':train_loader,'val':valid_loader,'test':test_loader}
            return dataloaders
    if(test_data != None and valid_size!=None):
        data_len = len(test_data)
        indices = list(range(data_len))
        np.random.shuffle(indices)
        split1 = int(np.floor(valid_size * data_len))
        valid_idx , test_idx = indices[:split1], indices[split1:]
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        valid_loader = DataLoader(test_data, batch_size= batch_size, sampler=valid_sampler)
        test_loader = DataLoader(test_data, batch_size= batch_size, sampler=test_sampler)
        train_loader =  DataLoader(train_data, batch_size = batch_size , shuffle = True)

        dataloaders = {'train':train_loader,'val':valid_loader,'test':test_loader}
        return dataloaders
    if(test_data!=None and valid_data !=None):
        valid_loader = DataLoader(valid_data, batch_size= batch_size,shuffle  = True)
        test_loader = DataLoader(test_data, batch_size= batch_size,shuffle = True)
        train_loader =  DataLoader(train_data, batch_size = batch_size , shuffle = True)

        dataloaders = {'train':train_loader,'val':valid_loader,'test':test_loader}
        return dataloaders
#To calculate mean and standard deviation of the dataset
#uncomment to calculate mean of dataset and replace in normalize transform
def normalization_parameter(dataloader):
    mean = 0.
    std = 0.
    nb_samples = len(dataloader.dataset)
    for data,_ in tqdm(dataloader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
    mean /= nb_samples
    std /= nb_samples
    return mean,std
