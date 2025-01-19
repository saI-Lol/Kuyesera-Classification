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
from sklearn.metrics import precision_score, recall_score, f1_score

class DatasetPost(Dataset):
    def __init__(self, dataset_root_paths, classes, transform=None, imgsz=128):
        data = []
        damage_types = []
        for dataset_root in dataset_root_paths:
            dataset_root = Path(dataset_root)
            image_ids = list(set(['_'.join(filename.split('_')[:-2]) for filename in os.listdir(dataset_root / "images")]))
            for image_id in image_ids:
                label_path = dataset_root / "labels" / f"{image_id}_post_disaster.json"
                with open(label_path, "r") as f:
                    json_data = json.load(f)
                    
                for i, feature in enumerate(json_data['features']['xy']):
                    subtype = feature['properties']['subtype'].replace("-", "_")
                    if subtype == "un_classified":
                        continue
                    damage_types.append(subtype)
                    polygon = wkt.loads(feature['wkt'])
                    polygon = self.clip_polygon_to_image(polygon)
                    if polygon:
                        xmin, ymin, xmax, ymax = polygon.bounds
                        data.append({
                            'image_path':dataset_root / "images" / f"{image_id}_post_disaster.tif",
                            'damage_type':subtype,
                            'bbox':list(map(int, [xmin, ymin, xmax, ymax]))
                        })
        self.data = data
        self.damage_type_to_id = {class_:idx for idx, class_ in enumerate(classes)}
        self.transform = transform
        self.imgsz = imgsz
        # print(self.damage_type_to_id, Counter(damage_types))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        with rasterio.open(item['image_path']) as f:
            img_arr = f.read()
        img_arr = np.transpose(img_arr, (1, 2, 0)).astype(np.uint8)        
        xmin, ymin, xmax, ymax = item['bbox']
        img_patch = img_arr[ymin:ymax, xmin:xmax, :]
        img = Image.fromarray(img_patch)
        img = img.resize((self.imgsz, self.imgsz), resample=Image.Resampling.LANCZOS)

        if self.transform is not None:
            img = self.transform(img)
        return img, self.damage_type_to_id[item['damage_type']]
        

    def clip_polygon_to_image(self, polygon):
        image_box = box(0, 0, 1024, 1024)
        return polygon.intersection(image_box)


class DatasetPrePost(Dataset):
    def __init__(self, dataset_root_paths, classes, transform=None, imgsz=128):
        data = []
        damage_types = []
        for dataset_root in dataset_root_paths:
            dataset_root = Path(dataset_root)
            image_ids = list(set(['_'.join(filename.split('_')[:-2]) for filename in os.listdir(dataset_root / "images")]))
            for image_id in image_ids:
                label_path = dataset_root / "labels" / f"{image_id}_post_disaster.json"
                with open(label_path, "r") as f:
                    json_data = json.load(f)
                    
                for i, feature in enumerate(json_data['features']['xy']):
                    subtype = feature['properties']['subtype'].replace("-", "_")
                    if subtype == "un_classified":
                        continue
                    damage_types.append(subtype)
                    polygon = wkt.loads(feature['wkt'])
                    polygon = self.clip_polygon_to_image(polygon)
                    if polygon:
                        xmin, ymin, xmax, ymax = polygon.bounds
                        data.append({
                            'pre_image_path':dataset_root / "images" / f"{image_id}_pre_disaster.tif",
                            'post_image_path':dataset_root / "images" / f"{image_id}_post_disaster.tif",
                            'damage_type':subtype,
                            'bbox':list(map(int, [xmin, ymin, xmax, ymax]))
                        })
        self.data = data
        self.damage_type_to_id = {class_:idx for idx, class_ in enumerate(classes)}
        self.transform = transform
        self.imgsz = imgsz
        # print(self.damage_type_to_id, Counter(damage_types))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        with rasterio.open(item['pre_image_path']) as f:
            pre_image = f.read()
        with rasterio.open(item['post_image_path']) as f:
            post_image = f.read()
        pre_image = np.transpose(pre_image, (1, 2, 0)).astype(np.uint8)  
        post_image = np.transpose(post_image, (1, 2, 0)).astype(np.uint8)        
        xmin, ymin, xmax, ymax = item['bbox']
        pre_image_patch = pre_image[ymin:ymax, xmin:xmax, :]
        post_image_patch = post_image[ymin:ymax, xmin:xmax, :]
        pre_image = Image.fromarray(pre_image_patch)
        post_image = Image.fromarray(post_image_patch)
        pre_image = pre_image.resize((self.imgsz, self.imgsz), resample=Image.Resampling.LANCZOS)
        post_image = post_image.resize((self.imgsz, self.imgsz), resample=Image.Resampling.LANCZOS)

        if self.transform is not None:
            pre_image = self.transform(pre_image)
            post_image = self.transform(post_image)
        return pre_image, post_image, self.damage_type_to_id[item['damage_type']]
        

    def clip_polygon_to_image(self, polygon):
        image_box = box(0, 0, 1024, 1024)
        return polygon.intersection(image_box)