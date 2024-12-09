import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast

class DF1d(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DF1d, self).__init__()
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
        if in_dim==80000: self.fc1 = nn.Linear(80128, 512) 
        elif in_dim==40000: self.fc1 = nn.Linear(int(in_dim/500*512), 512)
        elif in_dim==50000: self.fc1 = nn.Linear(50176, 512)
        elif in_dim==20000: self.fc1 = nn.Linear(int(in_dim/500*512), 512)
        elif in_dim==10000: self.fc1 = nn.Linear(10240, 512)
        elif in_dim==5000: self.fc1 = nn.Linear(int(in_dim/500*512), 512)
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

def prune_module(pruned, model, prune_amount):
    if pruned:
        model, pruned = remove_module(pruned, model)
        return model, pruned
    else:
        pruned = True
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=prune_amount)
                prune.l1_unstructured(module, name='bias', amount=prune_amount)
    return model, pruned

def remove_module(pruned, model):
    if pruned == False:
        return model, pruned
    else:
        pruned = False
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                prune.remove(module, 'weight')
                prune.remove(module, 'bias')
    return model, pruned