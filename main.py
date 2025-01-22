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
import pandas as pd
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
from helper import ddp_setup, train_epoch, validate_epoch, evaluate, predict

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
    combine_minor_major = args.combine_minor_major

    classes = args.classes
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if architecture == "PO":
        train_dataset = DatasetPost(train_dataset_root_paths, classes, combine_minor_major, transform=transform, imgsz=imgsz)
        val_dataset = DatasetPost(val_dataset_root_paths, classes, combine_minor_major, transform=transform, imgsz=imgsz)
        test_dataset = DatasetPost(test_dataset_root_paths, classes, combine_minor_major, transform=transform, imgsz=imgsz)
    else:
        train_dataset = DatasetPrePost(train_dataset_root_paths, classes, combine_minor_major, transform=transform, imgsz=imgsz)
        val_dataset = DatasetPrePost(val_dataset_root_paths, classes, combine_minor_major, transform=transform, imgsz=imgsz)
        test_dataset = DatasetPrePost(test_dataset_root_paths, classes, combine_minor_major, transform=transform, imgsz=imgsz)
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
        train_epoch(rank, epoch, model, optimizer, train_data_loader, architecture)
        validate_epoch(rank, epoch, model, val_data_loader, architecture, min_loss)
    evaluate(rank, model, test_data_loader, architecture)
    destroy_process_group()


# def experiment(rank, world_size, args):
#     ddp_setup(rank, world_size)
#     classes = args.classes
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])

#     checkpoint_path = args.checkpoint_path
#     architecture = args.architecture
#     loc_dir = args.loc_dir
#     original_file_dir = args.original_file_dir
#     batch_size = args.batch_size
#     workers= args.workers
#     imgsz = args.imgsz

#     checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
#     items = [f"architecture: {architecture}"] + [f'{k}: {v}' for k,v in checkpoint.items() if k != "model_state_dict"]
#     print(' '.join(items))

#     if architecture == "PO":
#         model = DamageClassifierPO().cuda()
#     elif architecture == "CC":
#         model = DamageClassifierCC().cuda()
#     elif architecture == "TTC":
#         model = DamageClassifierTTC().cuda()
#     elif architecture == "TTS":
#         model = DamageClassifierTTS().cuda()
#     model = model.to(rank)
#     model = DDP(model, device_ids=[rank]) 

#     loaded_dict = checkpoint['model_state_dict']
#     sd = model.state_dict()
#     for k in model.state_dict():
#         if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
#             sd[k] = loaded_dict[k]
#     loaded_dict = sd
#     model.load_state_dict(loaded_dict)
#     predict(model, loc_dir, original_file_dir, architecture, transform, imgsz)
#     destroy_process_group()

def create_submission(rank, world_size, args):
    ddp_setup(rank, world_size)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    main_model_checkpoint_path = args.main_model_checkpoint_path
    sub_model_checkpoint_path = args.sub_model_checkpoint_path
    architecture = args.architecture
    loc_df = pd.read_csv(args.loc_file_path)
    original_file_dir = args.original_file_dir
    batch_size = args.batch_size
    workers= args.workers
    imgsz = args.imgsz

    main_checkpoint = torch.load(main_model_checkpoint_path, weights_only=True, map_location="cpu")
    sub_checkpoint = torch.load(sub_model_checkpoint_path, weights_only=True, map_location="cpu")

    if architecture == "PO":
        model_main = DamageClassifierPO().cuda()
        model_sub = DamageClassifierPO().cuda()
    elif architecture == "CC":
        model_main = DamageClassifierCC().cuda()
        model_sub = DamageClassifierCC().cuda()
    elif architecture == "TTC":
        model_main = DamageClassifierTTC().cuda()
        model_sub = DamageClassifierTTC().cuda()
    elif architecture == "TTS":
        model_main = DamageClassifierTTS().cuda()
        model_sub = DamageClassifierTTS().cuda()
    model_main = model_main.to(rank)
    model_main = DDP(model_main, device_ids=[rank])
    model_sub = model_sub.to(rank)
    model_sub = DDP(model_sub, device_ids=[rank])


    loaded_dict_main = main_checkpoint['model_state_dict']
    loaded_dict_sub = sub_checkpoint['model_state_dict']
    sd_main = model_main.state_dict()
    sd_sub = model_sub.state_dict()
    for k in model_main.state_dict():
        if k in loaded_dict_main and sd_main[k].size() == loaded_dict_main[k].size():
            sd_main[k] = loaded_dict_main[k]
    loaded_dict_main = sd_main
    model_main.load_state_dict(loaded_dict_main)

    for k in model_sub.state_dict():
        if k in loaded_dict_sub and sd_sub[k].size() == loaded_dict_sub[k].size():
            sd_sub[k] = loaded_dict_sub[k]
    loaded_dict_sub = sd_sub
    model_sub.load_state_dict(loaded_dict_sub)
    predict(model_main, model_sub, loc_df, original_file_dir, architecture, transform, imgsz)
    destroy_process_group()

if __name__ == "__main__":    
    # parser = argparse.ArgumentParser(description="Train a model for damage classification")
    # parser.add_argument("--train_dataset_root_paths", type=str, nargs='+', required=True)
    # parser.add_argument("--val_dataset_root_paths", type=str, nargs='+', required=True)
    # parser.add_argument("--test_dataset_root_paths", type=str, nargs='+', required=True)
    # parser.add_argument("--architecture", type=str, required=True)
    # parser.add_argument("--batch_size", type=int, default=128)
    # parser.add_argument("--imgsz", type=int, default=128)
    # parser.add_argument("--workers", type=int, default=4)
    # parser.add_argument("--epochs", type=int, default=10)
    # parser.add_argument("--classes", type=str, nargs='+', default=['no_damage', 'minor_damage', 'major_damage', 'destroyed'])
    # parser.add_argument("--combine_minor_major", action="store_true", help="Combine minor and major categories (default: False)")
    # args = parser.parse_args()
    # world_size = torch.cuda.device_count()
    # mp.spawn(main, args=(world_size, args), nprocs=world_size)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--checkpoint_path", type=str, required=True)
    # parser.add_argument("--architecture", type=str, required=True)
    # parser.add_argument("--loc_dir", type=str, required=True)
    # parser.add_argument("--original_file_dir", type=str, required=True)
    # parser.add_argument("--batch_size", type=int, default=128)
    # parser.add_argument("--imgsz", type=int, default=128)
    # parser.add_argument("--workers", type=int, default=4)
    # args = parser.parse_args()
    # # experiment(args)
    # world_size = torch.cuda.device_count()
    # mp.spawn(experiment, args=(world_size, args), nprocs=world_size)

    parser = argparse.ArgumentParser()
    parser.add_argument("--main_model_checkpoint_path", type=str, required=True)
    parser.add_argument("--sub_model_checkpoint_path", type=str, required=True)
    parser.add_argument("--architecture", type=str, required=True)
    parser.add_argument("--loc_file_path", type=str, required=True)
    parser.add_argument("--original_file_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(create_submission, args=(world_size, args), nprocs=world_size)
