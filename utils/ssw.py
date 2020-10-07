import cv2 
import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math
import torch 
from PIL import Image
from skimage import io

def ssw(img,scale=500,sigma=0.7,min_size=20):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    # resize_img = cv2.resize(img, tuple(input_size))
    img_lbl,regions=selectivesearch.selective_search(img,scale=scale,sigma=sigma,min_size=min_size)
    candidates =set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        x, y, w, h = r['rect']
        # distorted rects
        if w  > 1.2*h or h > 1.2* w :
            continue
        candidates.add(r['rect'])
        
    '''
    #2)第二次过滤 大圈套小圈的目标 只保留大圈
    num_array=[]
    for i in candidates:
        if len(num_array)==0:
            num_array.append(i)
        else:
            content=False
            replace=-1
            index=0
        for j in num_array:
    ##新窗口在小圈 则滤
            if i[0]>=j[0] and i[0]+i[2]<=j[0]+j[2]and i[1]>=j[1] and i[1]+i[3]<=j[1]+j[3]:
                content=True
                break
            ##新窗口不在小圈 而在老窗口外部 替换老窗口
            elif i[0]<=j[0] and i[0]+i[2]>=j[0]+j[2]and i[1]<=j[1] and i[1]+i[3]>=j[1]+j[3]:
                replace=index
                break
                index+=1
            if not content:
                if replace>=0:
                    num_array[replace]=i
                else:
                    num_array.append(i)
            #窗口过滤完之后的数量
    num_array=set(num_array)
    '''
    return candidates

def feature_mapping(regions):   # 对region_proposal进行下采样，使得与feature_map的采样系数一样
    mapping=[]
    #如果保留pooling5，也就是映射到7*7
    # for ele in regions:
    #     mapping.append((math.floor(ele[0]/32)+1,math.floor(ele[1]/32)+1,max(math.ceil((ele[0]+ele[2])/32)-1-(math.floor(ele[0]/32)+1),0),
    #     max(0,math.ceil((ele[1]+ele[3])/32)-1-(math.floor(ele[1]/32)+1))))
    #如果不保留pooling5，也就是映射到14*14  
    for ele in regions:
        mapping.append([math.floor(ele[0]/16)+1,math.floor(ele[1]/16)+1,math.ceil((ele[0]+ele[2])/16)-1-(math.floor(ele[0]/16)+1),
        math.ceil((ele[1]+ele[3])/16)-1-(math.floor(ele[1]/16)+1)])   # 16个采样步长
    # mapping=list(set(mapping))
    return mapping

if __name__ == "__main__":
    img=Image.open('../data/000011.jpg')
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    candidates = ssw(img)
    # print(len(candidates))
    # mapping = np.array(feature_mapping(candidates))
    # print(mapping.shape)

    for x, y, w, h in candidates:
        # print(i)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    #for i in mapping:
    # print("xiacaiyang:", mapping)
    plt.show()





