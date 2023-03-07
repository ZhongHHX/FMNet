#from data_generators.datasets import cityscapes, coco, combine_dbs, pascal, sbd, deepfashion
from torch.utils.data import DataLoader
from data_generators.deepfashion import DeepFashionSegmentation
from data_generators.voc2012 import VOCSegmentation
import os
from glob import glob



def initialize_data_loader(config):


    if config['dataset']['dataset_name'] == 'deepfashion':
        train_set = DeepFashionSegmentation(config, split='train')
        val_set = DeepFashionSegmentation(config, split='test')
        #test_set = DeepFashionSegmentation(config, split='test')
    if config['dataset']['dataset_name'] == 'voc2012':
        train_set = VOCSegmentation(config, split='train')
        val_set = VOCSegmentation(config, split='val')
    else:
        raise Exception('dataset not implemented yet!')
    num_classes = train_set.num_classes
    train_loader = DataLoader(train_set, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['workers'], pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=config['training']['val_batch_size'], shuffle=False, num_workers=config['training']['workers'], pin_memory=True)
    #test_loader = DataLoader(test_set, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['workers'], pin_memory=True)


    return train_loader, val_loader#, test_loader, num_classes



#/root/zhonghao/gongchuang/myproject/fastfcn/dataset/deepfashion2_select77848/test/image/097580.jpg

if __name__ == '__main__':
    import yaml
    #path = '../configs/seg_hrnet_ocr_w48_cls59_520x520_sgd_lr1e-3_wd1e-4_bs_16_epoch200.yaml'
    #path = '../configs/config_hrnet_ocr.yml'
    path = '../configs/config_hrnet_ocr.yml'
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    #print(config)
    train,val = initialize_data_loader(config)

    #print(dataset.__len__())
    #print(len(data))

    for i,samlpe in enumerate(train):
        #print(samlpe)


        print(samlpe['label'].shape)
        if i>5:
            break