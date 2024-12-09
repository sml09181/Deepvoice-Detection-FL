#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import time
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
import gc
import logging
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

from model import *

def create_log(log_path, job_name):
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    
    file_handler = logging.FileHandler(os.path.join(log_path, f"log_{job_name}.txt"), 'w')
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    return log

def set_device(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id;
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu", 0)
    return device

def load_data(logger, data_type, trunc, start_idx):
    data_path = {
        "raw": "/DETECT/raw80000.npz", # (257865, 80000)
        "w2v": "/DETECT/w2v_emb80000.npz"
    }
    data = np.load(data_path.get(data_type))
    X = data['data']
    y = data['labels']
    
    if trunc != len(X[0]):
        X = np.array([x[start_idx:start_idx+trunc] for x in X])
    logger.info(f"X shape: {X.shape}")
    
    # train:valid:test=6:3:1 (3 for client train)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=22, stratify=y)
    # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.4, random_state=22, stratify=y_train)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=22, stratify=y)
    X_valid= X_test
    y_valid= y_test

    num_classes = len(np.unique(y_train))
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Train data shapes: {X_train.shape}, {y_train.shape}")
    logger.info(f"Valid data shapes: {X_valid.shape}, {y_valid.shape}")
    logger.info(f"Test data shapes: {X_test.shape}, {y_test.shape}")
    del data, X, y
    gc.collect()
    return torch.from_numpy(X_train), torch.from_numpy(y_train), \
        torch.from_numpy(X_valid), torch.from_numpy(y_valid), torch.from_numpy(X_test), torch.from_numpy(y_test), num_classes

def create_dataloader(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size):
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader, valid_loader, test_loader

def eval(model, test_loader, logger, device):
    #model.eval()
    # for child in model.children():
    #     if type(child)==nn.BatchNorm1d:
    #         child.track_running_stats = False
    with torch.no_grad():
        corrects = 0
        avg_loss = 0
        loss_func = nn.CrossEntropyLoss()
        for _, (b_x, b_y) in enumerate(test_loader):
            b_x = b_x.view(b_x.size(0), 1, b_x.size(1))
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            logit = model(b_x.float())
            loss = loss_func(logit, b_y)
            output = torch.softmax(logit, dim=1)
            pred = output.argmax(dim=1)
            corrects += pred.eq(b_y.view_as(pred)).float().sum().item()
            avg_loss += loss.item()
    avg_loss /= len(test_loader)
    accuracy = 100.0 * corrects / len(test_loader.dataset)
    print(corrects)
    logger.info("Test - loss: {:.6f} acc: {:3.4f}%".format(avg_loss, accuracy))
    
def main(
    data_type: str,
    trunc: int,
    start_idx: int,
    num_epochs: int,
    gpu_id: str,
    prune_amount: float,
    apply_pruning: bool,
    batch_size: int,
    learning_rate: float,
    temperature: float,
    fp16_precision: bool,
    ):
    ### CUDA SETTING
    device = set_device(gpu_id)
    
    ### MODEL PATH
    model_path = os.path.join("/DETECT/result/non_fl/saved_models/df", f"{data_type}_{trunc}_{start_idx}")
    if not os.path.isdir(model_path): os.makedirs(model_path)
    print("model_path:", model_path)
    
    ### SET LOGGER & SUMMARY WRITER
    job_name = time.strftime('%Y.%m.%d-%H:%M:%S')
    log_path = os.path.join("/DETECT/result/non_fl/log/df", model_path.split("/")[-1])
    if not os.path.isdir(log_path): os.makedirs(log_path)
    logger = create_log(log_path, job_name)
    writer = SummaryWriter(log_path)
    
    ### Data
    X_train, y_train, X_valid, y_valid, X_test, y_test, num_classes = load_data(logger, data_type, trunc, start_idx)
    train_loader, valid_loader, test_loader = create_dataloader(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size)
    
    
    ### SET MODEL
    model = DF1d(in_dim=trunc, out_dim=num_classes).to(device)  # Projection Head
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # OPTIMIZER = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
    loss_func = nn.CrossEntropyLoss()
    
    ### TRAINING
    pruned = False
    best_epoch = 0
    best_val_acc = -np.inf
    logger.info("Start Training")
    model.train()
    for epoch in range(num_epochs):
        if apply_pruning:
            model, pruned = prune_module(pruned, model, prune_amount)
        train_loss = []
        train_corrects = 0
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.view(b_x.size(0), 1, b_x.size(1))
            b_x = b_x.float().to(device)
            b_y = b_y.to(device)
            logit = model(b_x.float())
            output = torch.softmax(logit, dim=1)
            #output = logit
            loss = loss_func(logit, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)
            train_loss.append(loss.item())
            train_corrects += pred.eq(b_y.view_as(pred)).float().sum().item()
        avg_train_loss = sum(train_loss) / len(train_loss)
        avg_train_accuracy = 100.0 * train_corrects / len(train_loader.dataset)
        
        ## VALIDATION
        with torch.no_grad():
            valid_corrects = 0
            avg_valid_loss = 0
            for _, (b_x, b_y) in enumerate(valid_loader):
                b_x = b_x.view(b_x.size(0), 1, b_x.size(1))
                b_x = b_x.float().to(device)
                b_y = b_y.to(device)
                logit = model(b_x.float())
                #output = logit
                output = torch.softmax(logit, dim=1)
                pred = output.argmax(dim=1, keepdim=True)
                valid_corrects += pred.eq(b_y.view_as(pred)).float().sum().item()
                loss = loss_func(logit, b_y)
                avg_valid_loss += loss.item()
                # valid_corrects += (torch.max(logit, 1)
                #             [1].view(b_y.size()).data == b_y.data).sum()
            avg_valid_loss /= len(valid_loader)
            avg_valid_accuracy = 100.0 * valid_corrects / len(valid_loader.dataset)
            if avg_valid_accuracy > best_val_acc:
                best_epoch = epoch
                best_val_acc = avg_valid_accuracy
        
        logger.info('[Epoch {:2d}/{}]'.format(epoch, num_epochs))
        logger.info('Train - loss: {:.6f} acc: {:3.4f}%'.format(avg_train_loss, avg_train_accuracy))
        logger.info('Evaluation - loss: {:.6f} acc: {:3.4f}%'.format(avg_valid_loss, avg_valid_accuracy))
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/valid", avg_valid_loss, epoch)
        writer.add_scalar("Accuracy/train", avg_train_accuracy, epoch)
        writer.add_scalar("Accuracy/valid", avg_valid_accuracy, epoch)
        torch.save(model.state_dict(), f'{model_path}/epoch{epoch}.pth.tar')
        gc.collect()
    logger.info(f"Best Valid Accuracy: {best_val_acc} at Epoch {best_epoch}")
    del model
    
    ### EVALUATION
    num_classes = 2 # binary
    logger.info(f"Epoch for test: {best_epoch}")
    model = DF1d(in_dim=trunc, out_dim=num_classes).to(device)
    checkpoint = torch.load(f'{model_path}/epoch{best_epoch}.pth.tar')
    model.load_state_dict(checkpoint, strict=False)
    eval(model, test_loader, logger, device)
    logger.info("All Done")

if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("-d", '--data_type', type=str, default = "raw") # raw, w2v
    PARSER.add_argument("-t", '--trunc', type=int, default = 80000) # fixed length 
    PARSER.add_argument("-s", '--start_idx', type=int, default = 0) # start index of truncation
    
    PARSER.add_argument("--apply_pruning", action="store_true") # default: False
    PARSER.add_argument("-p", '--prune_amount', type=float, default = 0.2)
    PARSER.add_argument("-e", '--num_epochs', type=int, default = 50) # max_size: -1 입력
    PARSER.add_argument("-g", '--gpu_id', type=str, default = "6")

    ## Parameters
    PARSER.add_argument("--batch_size", type=int, default=128)
    PARSER.add_argument("--learning_rate", type=float, default=0.002)
    PARSER.add_argument("--temperature", type=float, default=0.5)
    PARSER.add_argument("--fp16_precision", action="store_false") # default: True
    main(**vars(PARSER.parse_args()))
