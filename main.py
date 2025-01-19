from helper import ddp_setup, train_epoch, validate_epoch, evaluate, install_package
install_package("rasterio")
from torchvision import models
from torch import nn
import torch
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import argparse
import numpy as np
from shapely import wkt, box
from PIL import Image
from collections import Counter
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from dataset import DatasetPost, DatasetPrePost
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from models import DamageClassifierPO, DamageClassifierCC, DamageClassifierTTC, DamageClassifierTTS

def main(rank, world_size, args):
    ddp_setup(rank, world_size)
    train_dataset_root_paths = args.train_dataset_root_paths
    val_dataset_root_paths = args.val_dataset_root_paths
    test_dataset_root_paths = args.test_dataset_root_paths
    architecture = args.architecture
    batch_size = args.batch_size
    imgsz = args.imgsz
    workers = args.workers
    epochs = args.epochs

    classes = ['no_damage', 'minor_damage', 'major_damage', 'destroyed']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if architecture == "PO":
        train_dataset = DatasetPost(train_dataset_root_paths, classes, transform=transform, imgsz=imgsz)
        val_dataset = DatasetPost(val_dataset_root_paths, classes, transform=transform, imgsz=imgsz)
        test_dataset = DatasetPost(test_dataset_root_paths, classes, transform=transform, imgsz=imgsz)
    else:
        train_dataset = DatasetPrePost(train_dataset_root_paths, classes, transform=transform, imgsz=imgsz)
        val_dataset = DatasetPrePost(val_dataset_root_paths, classes, transform=transform, imgsz=imgsz)
        test_dataset = DatasetPrePost(test_dataset_root_paths, classes, transform=transform, imgsz=imgsz)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last=True, sampler=DistributedSampler(train_dataset))
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, sampler=DistributedSampler(val_dataset))
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, sampler=DistributedSampler(test_dataset))

    if architecture == "PO":
        model = DamageClassifierPO().cuda()
    elif architecture == "CC":
        model = DamageClassifierCC().cuda()
    elif architecture == "TTC":
        model = DamageClassifierTTC().cuda()
    elif architecture == "TTS":
        model = DamageClassifierTTS().cuda()
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    params = model.parameters()
    optimizer = optim.SGD(params, lr=0.001, momentum=0.9)

    min_loss = None
    for epoch in range(epochs):
        train_epoch(epoch, model, optimizer, train_data_loader, architecture)
        validate_epoch(rank, epoch, model, val_data_loader, architecture, min_loss)
    evaluate(rank, model, test_data_loader, architecture)
    destroy_process_group()

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Train a model for damage classification")
    parser.add_argument("--train_dataset_root_paths", type=str, nargs='+', required=True)
    parser.add_argument("--val_dataset_root_paths", type=str, nargs='+', required=True)
    parser.add_argument("--test_dataset_root_paths", type=str, nargs='+', required=True)
    parser.add_argument("--architecture", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
