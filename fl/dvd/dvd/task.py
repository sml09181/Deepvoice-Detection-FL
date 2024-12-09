"""dvd: A Flower / PyTorch app."""

from collections import OrderedDict
import numpy as np
import pandas as pd
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, Partitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

import numpy as np
from datasets import Dataset 
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast

# DF 1d
class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()
        kernel_size = 8
        channels = [1, 32, 64, 128, 256] # == filter_num in tf
        conv_stride = 1
        pool_stride = 4
        pool_size = 8
        
        self.conv1 = nn.Conv1d(channels[0], channels[1], kernel_size, stride = conv_stride)
        self.conv1_1 = nn.Conv1d(channels[1], channels[1], kernel_size, stride = conv_stride)
        
        self.conv2 = nn.Conv1d(channels[1], channels[2], kernel_size, stride = conv_stride)
        self.conv2_2 = nn.Conv1d(channels[2], channels[2], kernel_size, stride = conv_stride)
       
        self.conv3 = nn.Conv1d(channels[2], channels[3], kernel_size, stride = conv_stride)
        self.conv3_3 = nn.Conv1d(channels[3], channels[3], kernel_size, stride = conv_stride)
       
        self.conv4 = nn.Conv1d(channels[3], channels[4], kernel_size, stride = conv_stride)
        self.conv4_4 = nn.Conv1d(channels[4], channels[4], kernel_size, stride = conv_stride)
       
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.batch_norm4 = nn.BatchNorm1d(256)
        self.batch_norm5 = nn.BatchNorm1d(512)
        
        self.max_pool_1 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_3 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_4 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.7) # fc1_dropout
        self.dropout3 = nn.Dropout(p=0.5) # fc2_dropout
        self.fc1 = nn.Linear(int(in_dim/500*512), 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, out_dim)
        self.softmax = nn.Softmax(dim=-1)

    def weight_init(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d): # glorot_uniform 
#                 m.weight.data.xavier_uniform_()
                # print (n)
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.zero_()
            
    def forward(self, inp):
        x = inp
        # ==== first block ====
        x = F.pad(x, (3,4))
        x = F.elu(self.batch_norm1(self.conv1(x)))
        x = F.pad(x, (3,4))
        x = F.elu(self.batch_norm1(self.conv1_1(x)))
#         x = F.elu(self.conv1_1(x))
        x = F.pad(x, (3, 4))
        x = self.max_pool_1(x)
        x = self.dropout1(x)
        
        # ==== second block ====
        x = F.pad(x, (3,4))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = F.pad(x, (3,4))
        x = F.relu(self.batch_norm2(self.conv2_2(x)))
#         x = F.relu(self.conv2_2(x))
        x = F.pad(x, (3,4))
        x = self.max_pool_2(x)
        x = self.dropout1(x)
        
        # ==== third block ====
        x = F.pad(x, (3,4))
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = F.pad(x, (3,4))
        x = F.relu(self.batch_norm3(self.conv3_3(x)))
#         x = F.relu(self.conv3_3(x))
        x = F.pad(x, (3,4))
        x = self.max_pool_3(x)
        x = self.dropout1(x)
        
        # ==== fourth block ====
        x = F.pad(x, (3,4))
        x = F.relu(self.batch_norm4(self.conv4(x)))
        x = F.pad(x, (3,4))
        x = F.relu(self.batch_norm4(self.conv4_4(x)))
#         x = F.relu(self.conv4_4(x))
        x = F.pad(x, (3,4))
        x = self.max_pool_4(x)
        x = self.dropout1(x)
        
        # ==== fc layers ====
        x = x.view(x.size(0), -1) # flatten
        x = self.fc1(x)
        x = F.relu(self.batch_norm5(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(self.batch_norm5(x))
        x = self.dropout3(x)
        x = self.fc3(x)
        #x = self.softmax(x)
        return x    

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def load_data(partition_id: int, num_partitions: int, batch_size: int):
    data_type = "raw"
    data_path = {
        "raw": "/scratch4/dvd/DETECT/raw5000_50000.npz", # (257865, 80000)
        "w2v": "/scratch4/dvd/DETECT/w2v_emb5000_50000.npz"
    }
    data = np.load(data_path.get(data_type))
    X = data['data']
    y = data['labels']
    dataset = Dataset.from_dict({"data": X, "label": y}).with_format("numpy")
    del data, X, y
    gc.collect()

    partitioner = IidPartitioner(num_partitions=num_partitions)
    partitioner.dataset = dataset
    partition = partitioner.load_partition(partition_id=0)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader

def train(model, trainloader, valloader, num_epochs, learning_rate, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    loss_func = nn.CrossEntropyLoss()
    
    ### TRAINING
    best_epoch = 0
    best_val_acc = -np.inf
    model.train()
    for epoch in range(num_epochs):
        train_loss = []
        train_corrects = 0
        for batch in trainloader:
            b_x = batch["data"].view(batch["data"].size(0), 1, batch["data"].size(1))
            b_x = b_x.float().to(device)
            b_y = batch["label"].to(device)
            logit = model(b_x.float())
            output = torch.softmax(logit, dim=1)
            loss = loss_func(logit, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)
            train_loss.append(loss.item())
            train_corrects += pred.eq(b_y.view_as(pred)).float().sum().item()
        avg_train_loss = sum(train_loss) / len(train_loss)
        avg_train_accuracy = 100.0 * train_corrects / len(trainloader.dataset)
        
        ## VALIDATION
        with torch.no_grad():
            valid_corrects = 0
            avg_valid_loss = 0
            for batch in valloader:
                b_x = batch["data"].view(batch["data"].size(0), 1, batch["data"].size(1)).float().to(device)
                b_y = batch["label"].to(device)
                logit = model(b_x)
                output = torch.softmax(logit, dim=1)
                pred = output.argmax(dim=1, keepdim=True)
                valid_corrects += pred.eq(b_y.view_as(pred)).float().sum().item()
                loss = loss_func(logit, b_y)
                avg_valid_loss += loss.item()
            avg_valid_loss /= len(valloader)
            avg_valid_accuracy = 100.0 * valid_corrects / len(valloader.dataset)
            if avg_valid_accuracy > best_val_acc:
                best_epoch = epoch
                best_val_acc = avg_valid_accuracy
        print('Train - loss: {:.6f} acc: {:3.4f}%'.format(avg_train_loss, avg_train_accuracy))
        print('Valid - loss: {:.6f} acc: {:3.4f}%'.format(avg_valid_loss, avg_valid_accuracy))
        gc.collect()
        
    results = {
        "val_loss": avg_valid_loss,
        "val_accuracy": avg_valid_accuracy,
    }
    return results

def test(model, test_loader, device):
    model.to(device)
    with torch.no_grad():
        corrects = 0
        avg_loss = 0
        loss_func = nn.CrossEntropyLoss()
        for batch in test_loader:
            b_x = batch["data"].view(batch["data"].size(0), 1, batch["data"].size(1)).float().to(device)
            b_y = batch["label"].to(device)
            logit = model(b_x)
            loss = loss_func(logit, b_y)
            output = torch.softmax(logit, dim=1)
            pred = output.argmax(dim=1)
            corrects += pred.eq(b_y.view_as(pred)).float().sum().item()
            avg_loss += loss.item()
    avg_loss /= len(test_loader)
    accuracy = 100.0 * corrects / len(test_loader.dataset)
    print("Test - loss: {:.6f} acc: {:3.4f}%".format(avg_loss, accuracy))
    return avg_loss, accuracy
