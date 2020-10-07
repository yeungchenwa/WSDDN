import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from tqdm.autonotebook import tqdm
import os
# from tensorboardX import SummaryWriter
from dataset.dataset import VOCDataset
from model.wsddn import WSDDN
from config import DefaultConfig

opt = DefaultConfig()
Transform = transforms.Compose([
    transforms.Resize(opt.input_size),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = opt.mean,
                         std  = opt.std),
    ])

# if opt.use_GPU:
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# print parameters
print("Parameters: ")
print("lr:",opt.lr, "  batch_size:",opt.batch, "  num_workers:",opt.num_workers, "  max_epochs:",opt.max_epochs, "  input_size:",opt.input_size)

# load model
wsddn = WSDDN(opt.backbone, pretrained=opt.pretrained, classes_num=opt.classes_num)
if opt.resume:
    wsddn.load_state_dict(torch.load(opt.resume))
    print("loaded successfully!")
wsddn.to(DEVICE)
wsddn.train()

# load data
trainval = VOCDataset(opt.train_root, Transform, classes_num=opt.classes_num, input_size=opt.input_size)
trainloader = DataLoader(trainval, batch_size=opt.batch, shuffle=True, num_workers=opt.num_workers)

# loss function and optimizer
criterion = nn.BCELoss(weight=None, size_average=True) 
optimizer1 = optim.SGD(wsddn.parameters(), lr = opt.lr, momentum = 0.9)
optimizer2 = optim.SGD(wsddn.parameters(), lr = opt.lr_decay * opt.lr, momentum = 0.9)
# writer = SummaryWriter('WSDDN')

# wsddn.train()
step = 0
for epoch in range(opt.max_epochs):
    running_loss = 0
    # progress_bar = tqdm(trainloader)
    for i, (img_data, labels, ssw_bbox) in enumerate(trainloader):
        # if i < len(trainval):
        #     progress_bar.update()
        #     continue
        input_data = Variable(img_data)
        target = Variable(labels)
        ssw = Variable(ssw_bbox)
        input_data = input_data.to(DEVICE)
        target = target.to(DEVICE)
        ssw = ssw.to(DEVICE)
        if (epoch+1) <= 10:
            optimizer1.zero_grad()
        else:
            optimizer2.zero_grad()
        class_result = wsddn(input_data, ssw)
        loss = wsddn.calculate_loss(class_result, target)
        loss.backward()
        if (epoch+1) <= 10:
            optimizer1.step()
        else:
            optimizer2.step()
        running_loss += loss.item()
        step += 1
        # print("finished！")

        # if i+1 % opt.print_freq == 0:
        print('step:[%d]  epoch[%d/%d: %2d/%d] loss: %.3f' % (step, epoch + 1 , opt.max_epochs, i + 1 , len(trainval), loss.item()))
        """
        progress_bar.set_description(
            'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}'.format(
                step, epoch + 1, opt.num_epochs, i + 1, len(trainval), loss.item()))"""
        # writer.add_scalar('Train/loss', loss.item(),step)
    print("Avg loss is %.3f"%(running_loss / len(trainval)))

    if (epoch+1) % opt.saved_epoch == 0:
        print("saving checkpoints...")
        torch.save(wsddn.state_dict(), os.path.join(opt.save_path, 'epoch_%d.pth'%(epoch)))
        print("saved successfully!")
    
print('Finished Training')
writer.close()





