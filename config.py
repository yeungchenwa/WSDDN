class DefaultConfig(object):
    backbone = "VGG16"
    classes_num = 20
    train_root = "/home/yeung/DATA/VOC2017/train"
    test_root = "/home/yeung/DATA/VOC2017/test"
    pretrained = False
    # resume = "/home/zhenhua/WSDDN_pytorch/checkpoints/voc_2.pth"
    resume = None
    save_path = "/home/yeung/Projects/WSDDN/checkpoints"
    load_path = "/home/yeung/Projects/WSDDN/checkpoints/vgg16-397923af.pth"
    input_size = [480, 480]
    mean = [ 0.485, 0.456, 0.406 ]
    std  = [ 0.229, 0.224, 0.225 ]

    batch = 1
    use_GPU = True
    num_workers = 4
    print_freq = 1
    max_epochs = 20
    saved_epoch = 1
    lr = 0.00001
    lr_decay = 0.1
    choose_threshold = 0.3

