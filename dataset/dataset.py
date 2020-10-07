from torch.utils.data import Dataset, DataLoader
import torch
import sys
from utils.ssw import ssw, feature_mapping
import xml.etree.ElementTree as ET
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import math
from torch.utils.data import DataLoader

class VOCDataset(Dataset):
    def __init__(self, root, transform, classes_num, input_size):
        self.root = root
        self.transform = transform
        self.classes_num = classes_num
        self.input_size = input_size
        self.img_path = os.path.join(root, "JPEGImages")
        self.anno_path = os.path.join(root, "Annotations")
        imgs = os.listdir(self.img_path)
        self.images = [img for img in imgs]
        self.class_id = {
        "aeroplane": 0,
        "bicycle": 1,
        "bird": 2,
        "boat": 3,
        "bottle": 4,
        "bus": 5,
        "car": 6,
        "cat": 7,
        "chair": 8,
        "cow": 9,
        "diningtable": 10,
        "dog": 11,
        "horse": 12,
        "motorbike": 13,
        "person": 14,
        "pottedplant": 15,
        "sheep": 16,
        "sofa": 17,
        "train": 18,
        "tvmonitor": 19,
        }

    def __getitem__(self, index):
        img_path = os.path.join(self.root, "JPEGImages", self.images[index])
        anno_path = self.root + "/Annotations/" + self.images[index].split(".")[0] + ".xml"
        img_data = Image.open(img_path)
        original_size = img_data.size
        # print(original_size)
        xml = ET.parse(anno_path)
        labels = torch.zeros(self.classes_num)
        for obj in xml.findall("object"):
            labels[self.class_id[obj.find("name").text]]=1
        labels = torch.Tensor(labels)   # 必须要为tensor
        """
        ssw_bound = ssw(img_data)          # 在原图上得到ssw后的框
        # ssw_bbox = list(ssw_bound)
        ssw_bbox = feature_mapping(ssw_bound)      # 对ssw后的框下取样16
        ssw_b = ssw_bbox.copy()
        # print(len(ssw_b))
        if len(ssw_b)==0:
            ssw_b = [[1, 1, 1, 1]]                  # 问题就是出现在这里！！！！！！！！
        if ssw_b != [[1, 1, 1, 1]]:
            for i in range(len(ssw_b)):
                ssw_b[i][0] = ssw_bbox[i][0]
                ssw_b[i][1] = ssw_bbox[i][1]
                ssw_b[i][2] = ssw_bbox[i][0] + ssw_bbox[i][2]
                ssw_b[i][3] = ssw_bbox[i][1] + ssw_bbox[i][3]
        """
        ssw_b = self.ssw_process(img_data, original_size)
        ssw_b = torch.Tensor(ssw_b)   # 必须要转为tensor
        img_data = self.transform(img_data)
        # print(img_data.shape)
        return img_data, labels, ssw_b

    def __len__(self):
        return len(self.images)

    
    def get_resized_ssw(self, ssw_bbox, original_size): 
        ssw_bbox[-1][0] = math.floor(self.input_size[0]/original_size[0])*ssw_bbox[-1][0]
        ssw_bbox[-1][2] = math.floor(self.input_size[0]/original_size[0])*ssw_bbox[-1][2]
        ssw_bbox[-1][1] = math.floor(self.input_size[1]/original_size[1])*ssw_bbox[-1][1]
        ssw_bbox[-1][3] = math.floor(self.input_size[1]/original_size[1])*ssw_bbox[-1][3]
        return ssw_bbox
        

    def ssw_process(self, img_data, original_size):
        ssw_bounds = ssw(img_data)          # 在原图上得到ssw后的框
        ssw_bbox = []
        for ele in ssw_bounds:
            ssw_bbox.append([ele[0], ele[1], ele[0]+ele[2], ele[1]+ele[3]])
        if len(ssw_bbox)==0:
            ssw_bbox = [[1, 1, 1, 1]]                  # 问题就是出现在这里！！！！！！！！
        ssw_resized = self.get_resized_ssw(ssw_bbox, original_size)
        return ssw_resized
        
        

"""
if __name__ == "__main__":
    Transform = transforms.Compose([
    transforms.Resize([480, 480]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std  = [ 0.229, 0.224, 0.225 ]),
    ])
    train = VOCDataset("/home/zhenhua/WSDDN_pytorch/data/train", Transform, 20, [512, 512])
    test = VOCDataset("/home/zhenhua/WSDDN_pytorch/data/test", Transform, 20, [512, 512])
    trainloader = DataLoader(train, batch_size=8, shuffle=True, num_workers=4)
    print("train_data_nums:", len(train))
    print("test_data_nums:", len(test))
    dataiter = iter(trainloader)
    img, labels, ssw_bbox= next(dataiter)
    print(labels.shape)"""