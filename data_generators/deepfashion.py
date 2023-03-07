from __future__ import print_function, division
import os
from PIL import Image
import json
import numpy as np
from torch.utils.data import Dataset,DataLoader

from torchvision import transforms
from preprocessing import custom_transforms as tr
import random
import copy
import torch


class DeepFashionSegmentation(Dataset):
    """
    DeepFashion dataset
    """
#    NUM_CLASSES = 14

    def __init__(self,
                 config,
#                 base_dir=config['dataset']['base_path'],
                 split='train',
                 ):
        super().__init__()
        self._base_dir = config['dataset']['base_path']
        self._image_dir = os.path.join(self._base_dir, 'train', 'image')
        self._cat_dir = os.path.join(self._base_dir, 'labels')

        self.train_img_dir = os.path.join(self._base_dir, 'train', 'image')
        self.train_label_dir = os.path.join(self._base_dir, 'train', 'label')
        self.val_img_dir = os.path.join(self._base_dir, 'val', 'image')
        self.val_label_dir = os.path.join(self._base_dir, 'val', 'label')
        self.test_img_dir = os.path.join(self._base_dir, 'test', 'image')
        self.test_label_dir = os.path.join(self._base_dir, 'test', 'label')

        self.config = config
        self.split = split

        # with open(os.path.join(self._base_dir, 'train_val_test.json')) as f:
        #     self.full_dataset = json.load(f)

        self.images = []
        self.categories = []
        self.num_classes = self.config['network']['num_classes']

        self.shuffle_dataset()

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def shuffle_dataset(self):
        #reset lists
        self.images.clear()
        self.categories.clear()


        if self.split == "train":
            trian_file_list = os.listdir(self.train_img_dir)
            train_ids = [os.path.splitext(os.path.basename(p))[0] for p in trian_file_list]
            for id in train_ids:
                self.images.append(os.path.join(self.train_img_dir, id + ".jpg"))
                self.categories.append(os.path.join(self.train_label_dir, id + ".png"))
        elif self.split == "val":
            val_file_list = os.listdir(self.val_img_dir)
            val_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_file_list]
            for id in val_ids:
                self.images.append(os.path.join(self.val_img_dir, id + ".jpg"))
                self.categories.append(os.path.join(self.val_label_dir, id + ".png"))
        elif self.split == "test":
            test_file_list = os.listdir(self.test_img_dir)
            test_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_file_list]
            for id in test_ids:
                self.images.append(os.path.join(self.test_img_dir, id + ".jpg"))
                self.categories.append(os.path.join(self.test_label_dir, id + ".png"))


        if len(self.images) % self.config['training']['batch_size'] != 0:
            addnum = self.config['training']['batch_size'] - len(self.images) % self.config['training']['batch_size']
            add_image_data = self.images[-addnum:]
            add_categories_data = self.categories[-addnum:]
            for i in range(addnum):
                self.images.append(add_image_data[i])
                self.categories.append(add_categories_data[i])


        # # be sure that total dataset size is divisible by 2
        # if len(self.images) % 2 != 0:
        #     self.images.append(os.path.join(self._image_dir, item['image']))
        #     self.categories.append(os.path.join(self._cat_dir, item['annotation']))

        assert (len(self.images) == len(self.categories))
#        print(self.images[0])
#        print(len(self.images))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target, _h, _w = self._make_img_gt_point_pair(index)
        # print(_img.size)

        # seg_labels = np.eye(self.num_classes)[_target.reshape([-1])]
        #
        # seg_labels = seg_labels.reshape((int(_img[0]), int(_img[1]), self.num_classes))


        # sample = {'image': _img, 'label': _target, 'seg_labels': seg_labels}
        sample = {'image': _img, 'label': _target}


        # # for split in self.split:
        # if self.split == "train":
        #     copy_target = copy.deepcopy(_target)
        #     copy_target = np.array(copy_target)
        #     print(type(copy_target.shape), copy_target.shape)
        #     seg_labels = np.eye(self.num_classes)[copy_target.reshape([-1])]
        #     seg_labels = seg_labels.reshape(512, 512, self.num_classes)
        #     #            print('train')
        #     traindata = self.transform_tr(sample)
        #     return traindata, seg_labels
        # elif self.split == 'val':
        #     copy_target = copy.deepcopy(_target)
        #     copy_target = np.array(copy_target)
        #     seg_labels = np.eye(self.num_classes)[copy_target.reshape([-1])]
        #     seg_labels = seg_labels.reshape(512, 512, self.num_classes)
        #     #            print('val')
        #     return self.transform_val(sample), seg_labels
        # elif self.split == 'test':
        #     #           print('in return')
        #     return self.transform_val(sample)

        # for split in self.split:
        if self.split == "train":
            #            print('train')
            traindata = self.transform_tr(sample)

            # print(type(traindata))

            target_tensor = traindata['label']
            # print(type(target_tensor), target_tensor.shape)

            target_numpy = target_tensor.numpy().astype(np.uint8)
            #print(np.unique(target_numpy))
            print(target_tensor.shape,traindata['image'].shape)
            print(target_numpy.reshape([-1]).shape)
            target_numpy_ = np.eye(self.num_classes + 1)[target_numpy.reshape([-1])]
            target_numpy_ = target_numpy_.reshape(target_tensor.shape[0], target_tensor.shape[1], self.num_classes + 1)

            seg_labels = np.array(target_numpy_).astype(np.float32)
            seg_labels = torch.from_numpy(seg_labels).float()

            traindata['seg_labels'] = seg_labels

            return traindata
        elif self.split == 'val':
            #            print('val')
            valdata = self.transform_val(sample)
            target_tensor = valdata['label']
            target_numpy = target_tensor.numpy().astype(np.uint8)
            target_numpy_ = np.eye(self.num_classes + 1)[target_numpy.reshape([-1])]
            target_numpy_ = target_numpy_.reshape(target_tensor.shape[0], target_tensor.shape[1], self.num_classes + 1)

            valdata['seg_labels'] = target_numpy_
            return valdata
        elif self.split == 'test':
            #           print('in return')
            #return self.transform_val(sample)
            valdata = self.transform_val(sample)
            target_tensor = valdata['label']
            target_numpy = target_tensor.numpy().astype(np.uint8)
            target_numpy_ = np.eye(self.num_classes + 1)[target_numpy.reshape([-1])]
            target_numpy_ = target_numpy_.reshape(target_tensor.shape[0], target_tensor.shape[1], self.num_classes + 1)

            valdata['seg_labels'] = target_numpy_
            valdata['name']  = str(self.images[index]).split('\\')[3].split('.')[0]
            valdata['h'] = _h
            valdata['w'] = _w
            return valdata

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])
        _h = _img.size[1]
        _w = _img.size[0]
        return _img, _target, _h, _w

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.config['image']['base_size'],
                               crop_size=self.config['image']['crop_size']),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            #            tr.FixScaleCrop(crop_size=crop_size),
            tr.FixScaleCrop(crop_size=self.config['image']['crop_size']),
            #            tr.FixScaleCrop(crop_size=513),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    @staticmethod
    def preprocess(sample, crop_size=513):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    #gc自己写的不裁剪
    @staticmethod
    def preprocess_no_crop(sample):

        composed_transforms = transforms.Compose([
            # tr.FixScaleCrop(crop_size=crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()
        ])

        return composed_transforms(sample)

    def __str__(self):
        return 'DeepFashion2(split=' + str(self.split) + ')'




if __name__ == '__main__':
    import yaml
    #path = '../configs/seg_hrnet_ocr_w48_cls59_520x520_sgd_lr1e-3_wd1e-4_bs_16_epoch200.yaml'
    #path = '../configs/config_hrnet_ocr.yml'
    path = '../configs/config.yml'
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    dataset = DeepFashionSegmentation(config,split='train')
    data = DataLoader(dataset,4,shuffle=True)
    print(dataset.__len__())
    print(len(data))

    for i,samlpe in enumerate(data):
        #print(samlpe)


        print(samlpe[0]['label'].shape)
        if i>5:
            break