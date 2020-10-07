import torch
from model.wsddn import WSDDN
from config import DefaultConfig
import cv2
from PIL import Image, ImageDraw
from utils.ssw import ssw
import numpy as np
import math
import torchvision.transforms as transforms

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
opt = DefaultConfig()
Transform = transforms.Compose([
    transforms.Resize(opt.input_size),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = opt.mean,
                         std  = opt.std),
    ])


def get_resized_ssw(ssw_bbox, original_size): 
    ssw_bbox[-1][0] = math.floor(opt.input_size[0]/original_size[0])*ssw_bbox[-1][0]
    ssw_bbox[-1][2] = math.floor(opt.input_size[0]/original_size[0])*ssw_bbox[-1][2]
    ssw_bbox[-1][1] = math.floor(opt.input_size[1]/original_size[1])*ssw_bbox[-1][1]
    ssw_bbox[-1][3] = math.floor(opt.input_size[1]/original_size[1])*ssw_bbox[-1][3]
    return ssw_bbox

class_id = {
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

if __name__ == "__main__":
    img = Image.open("./data/000085.jpg")
    candidates = ssw(img)
    ssw_bbox = []
    for ele in candidates:
        ssw_bbox.append([ele[0], ele[1], ele[0]+ele[2], ele[1]+ele[3]])
    resized_ssw_bbox = get_resized_ssw(ssw_bbox, img.size)
    transform_img = Transform(img)

    wsddn = WSDDN(opt.backbone, pretrained=False, classes_num=opt.classes_num)
    print("loading checkpoint...")
    wsddn.load_state_dict(torch.load(opt.load_path))
    wsddn.to(DEVICE)
    print("loaded successfully!")
    
    resized_ssw_bbox = torch.Tensor(resized_ssw_bbox)
    batch_img = torch.unsqueeze(transform_img, dim=0)
    batch_img = batch_img.to(DEVICE)
    batch_ssw_bbox = torch.unsqueeze(resized_ssw_bbox, dim=0)
    batch_ssw_bbox = batch_ssw_bbox.to(DEVICE)
    combined_scores = wsddn(batch_img, batch_ssw_bbox)
    
    print(combined_scores.shape)
    print(combined_scores)

    selected_bbox = []
    for i in range(len(combined_scores)):
        for j in range(opt.classes_num):
            if combined_scores[i][j] > opt.choose_threshold:
                index = [i, j]
                selected_bbox.append(index)

    draw = ImageDraw.Draw(img)
    for m in range(len(selected_bbox)):
        choose_index = selected_bbox[m]
        pre_bbox = ssw_bbox[choose_index[0]]
        draw.rectangle(xy=(pre_bbox[0], pre_bbox[1], pre_bbox[2], pre_bbox[3]), fill=None, outline="red", width=1)
    
    img.save("./data/pred_000085.jpg")
    





    


