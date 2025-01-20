from torchvision import models
from torch import nn
import torch
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import numpy as np
from shapely import wkt, box
import rasterio
from PIL import Image
from collections import Counter
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from torch.distributed import init_process_group

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_outputs(model, batch, architecture):
    if architecture == "PO":
        imgs, labels = batch
        imgs = imgs.cuda()
        labels = labels.cuda()
        outputs = model(imgs)
        return outputs, labels
    else:
        pre_imgs, post_imgs, labels = batch
        pre_imgs = pre_imgs.cuda()
        post_imgs = post_imgs.cuda()
        labels = labels.cuda()
        outputs = model(pre_imgs, post_imgs)
        return outputs, labels

def save_model(model, min_loss, epoch, architecture):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'min_loss': min_loss
    }, f"{architecture}_model_best.pth")

def ddp_setup(rank: int, world_size: int):
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"
   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size)

def train_epoch(rank, epoch, model, optimizer, data_loader, architecture):
    losses = AverageMeter()
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.train()    
    iterator = tqdm(data_loader)
    for batch in iterator:
        with autocast(device_type=device):
            outputs, labels = get_outputs(model, batch, architecture)
            loss = criterion(outputs, labels)
        losses.update(loss.item(), labels.size(0))
        iterator.set_description(f"GPU: {rank} Epoch: {epoch+1} Loss: {losses.avg:.4f}")

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

def validate_epoch(rank, epoch, model, data_loader, architecture, min_loss):
    losses = AverageMeter()
    criterion = nn.CrossEntropyLoss()    
    model.eval()    
    iterator = tqdm(data_loader)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in iterator:
            outputs, labels = get_outputs(model, batch, architecture)
            loss = criterion(outputs, labels)
            losses.update(loss.item(), labels.size(0))

            preds = torch.argmax(torch.softmax(outputs, 1), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")
    
    print(f"GPU: {rank} | Validation Loss: {losses.avg:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
    if min_loss is None or losses.avg < min_loss:
        min_loss = losses.avg
        save_model(model, min_loss, epoch, architecture)

def evaluate(rank, model, data_loader, architecture):
    losses = AverageMeter()
    criterion = nn.CrossEntropyLoss()  
    model.eval()    
    iterator = tqdm(data_loader)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in iterator:
            outputs, labels = get_outputs(model, batch, architecture)
            loss = criterion(outputs, labels)
            losses.update(loss.item(), labels.size(0))

            preds = torch.argmax(torch.softmax(outputs, 1), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")
    report = classification_report(all_labels, all_preds)
    
    print(f"GPU: {rank} | Test Loss: {losses.avg:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
    print(report)