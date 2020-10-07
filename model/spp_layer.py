import math
import torch.nn as nn
import torch

def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    for i in range(len(out_pool_size)):
        h_wid = math.ceil(previous_conv_size[0] / out_pool_size[i])
        w_wid = math.ceil(previous_conv_size[1] / out_pool_size[i]) # kernel_size(int or tuple) - max pooling的窗口大小
        h_pad = min(math.floor((h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2),math.floor(h_wid/2))
        w_pad = min(math.floor((w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2),math.floor(w_wid/2)) # padding(int or tuple, optional) - 输入的每一条边补充0的层数
        # 以上的计算就是为了解决无论取出来的region_proposal大小如何，最终经过maxpool后的大小都是一致的
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)  # [1, 512, 2, 2]后面两维是out_pool_size
        # print("x shape:", x.shape)
        if x.shape[2] == 1:
            x = torch.cat((x, x), 2)
        elif x.shape[3] == 1:
            x = torch.cat((x, x), 3)
        m = x.view(num_sample,-1)
        if(m.size(1) != 2048):
            continue
        if(i == 0):
            spp = x.view(num_sample,-1)
            # print("spp0 size:",spp.size())
        else:
            spp = torch.cat((spp,x.view(num_sample,-1)), 1)
            # print("spp1 size:", spp.size())
    return spp
