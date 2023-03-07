import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as Ft
import matplotlib.pyplot as plt

from preprocessing import custom_transforms as tr


# 接下来，对标签图片进行处理，将其转换为对应的标签矩阵。先列出标签中每个RGB的值及其对应类别，一共21类:
classes = ['background',
           'aeroplane',
           'bicycle',
           'bird',
           'boat',
           'bottle',
           'bus',
           'car',
           'cat',
           'chair',
           'cow',
           'diningtable',
           'dog',
           'horse',
           'motorbike',
           'person',
           'potted plant',
           'sheep',
           'sofa',
           'train',
           'tv/monitor']

# RGB color for each class
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]
colormap = np.asarray(colormap)
cm2lbl = np.zeros(256 ** 3)

# 建立一个索引，将标签图片中每个像素的RGB值一对一映射到对应的类别索引:
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def image2label(im):
    data = np.array(im, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype='int64')









class VOCSegmentation(Dataset):
    """

    最后，通过torch.utls.data.Dataset自定义数据集类，
    通过._getitem__函数，访问数据集中索引为idx 的输入图像及其对应的标签矩阵。
    由于数据集中有些图像的尺寸可能小于随机裁剪所指定的输出尺寸，这些样本需要通过自定义的fiter 函数所移除。
    此外，还对输入图像的RGB三个通道的值分别做标准化。

    """

    def __init__(self, config, split='train' ):
        """
        :param voc_root: 放置数据集的位置
        :param year: 年份，我这里只放置了2012年的
        :param transforms: 是否对图片进行裁剪，transforms =None不进行裁剪
        :param txt_name:
        """

        super(VOCSegmentation, self).__init__()
        self.config = config
        self.split = split
        self.num_classes = self.num_classes = self.config['network']['num_classes']
        root = os.path.join(config['dataset']['base_path'], "VOCdevkit", f"VOC2010")
        # 拼接字符串
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, 'JPEGImages')
        # 掩膜的路径位置，就是分割好的图片
        mask_dir = os.path.join(root, 'SegmentationClass')
        #
        txt_path = ''
        if split=='train':
            txt_path = os.path.join(root, "ImageSets", "Segmentation", "train.txt")
        if split=='val':
            txt_path = os.path.join(root, "ImageSets", "Segmentation", "val.txt")

        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)

        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        # 根据Segmentation 文件夹下所以提供的train.txt,来进行图片的加载
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        # 掩膜图片位置
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))




    def __getitem__(self, index):
        _img, _target, _h, _w = self._make_img_gt_point_pair(index)
        # print(_img.size)

        # seg_labels = np.eye(self.num_classes)[_target.reshape([-1])]
        #
        # seg_labels = seg_labels.reshape((int(_img[0]), int(_img[1]), self.num_classes))


        # sample = {'image': _img, 'label': _target, 'seg_labels': seg_labels}
        sample = {'image': _img, 'label': _target}



        if self.split == "train":
            #            print('train')
            traindata = self.transform_tr(sample)

            # print(type(traindata))

            # target_tensor = traindata['label']
            # # print(type(target_tensor), target_tensor.shape)
            #
            # target_numpy = target_tensor.numpy().astype(np.uint8)
            # #print(np.unique(target_numpy))
            # print(target_tensor.shape,traindata['image'].shape)
            # print(target_numpy.reshape([-1]).shape)
            # target_numpy_ = np.eye(self.num_classes + 1)[target_numpy.reshape([-1])]
            # target_numpy_ = target_numpy_.reshape(target_tensor.shape[0], target_tensor.shape[1], self.num_classes + 1)
            #
            # seg_labels = np.array(target_numpy_).astype(np.float32)
            # seg_labels = torch.from_numpy(seg_labels).float()
            seg_labels = []
            traindata['seg_labels'] = seg_labels

            return traindata
        elif self.split == 'val':
            #            print('val')
            valdata = self.transform_val(sample)
            #target_tensor = valdata['label']
            # target_numpy = target_tensor.numpy().astype(np.uint8)
            # target_numpy_ = np.eye(self.num_classes + 1)[target_numpy.reshape([-1])]
            # target_numpy_ = target_numpy_.reshape(target_tensor.shape[0], target_tensor.shape[1], self.num_classes + 1)
            target_numpy_ = []   #one-hot 编码 只

            valdata['seg_labels'] = target_numpy_
            return valdata
        elif self.split == 'test':
            #           print('in return')
            #return self.transform_val(sample)
            valdata = self.transform_val(sample)
            # target_tensor = valdata['label']
            # target_numpy = target_tensor.numpy().astype(np.uint8)
            # target_numpy_ = np.eye(self.num_classes + 1)[target_numpy.reshape([-1])]
            # target_numpy_ = target_numpy_.reshape(target_tensor.shape[0], target_tensor.shape[1], self.num_classes + 1)
            target_numpy_ = []
            valdata['seg_labels'] = target_numpy_
            # valdata['name']  = str(self.images[index]).split('\\')[3].split('.')[0]
            # valdata['h'] = _h
            # valdata['w'] = _w
            return valdata

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.masks[index])
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

    def __len__(self):
        return len(self.images)




if __name__ == '__main__':
    import yaml
    #path = '../configs/seg_hrnet_ocr_w48_cls59_520x520_sgd_lr1e-3_wd1e-4_bs_16_epoch200.yaml'
    #path = '../configs/config_hrnet_ocr.yml'
    path = '../configs/config_hrnet_ocr.yml'
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    #print(config)
    dataset = VOCSegmentation(config,split='train')
    data = DataLoader(dataset,4,shuffle=True)
    #print(dataset.__len__())
    #print(len(data))

    for i,samlpe in enumerate(data):
        #print(samlpe)


        print(samlpe['label'].shape)
        if i>5:
            break


# def get_dataloader(mode=True, batch_size=4, shape=(512, 512)):
#     """
#     获取数据集加载
#     :param mode:
#     :return:
#     """
#     if mode:
#         # 2. 实例化，准备dataloader
#         dataset = VOCSegmentation(voc_root=r"E:\note\cv\data\VOC_Train",
#                                   shape=shape,
#                                   )
#         dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
#     else:
#         dataset = VOCSegmentation(voc_root=r"E:\note\cv\data\VOC_Train",
#                                   shape=shape,
#                                   txt_name="val.txt",
#                                   )
#         dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
#     return dataloader
#
#
# cm = np.array(colormap).astype('uint8')
#
#
# def show_image(tensor):
#     plt.figure()
#     image = tensor.numpy().transpose(1, 2, 0)
#     plt.imshow(image)
#     plt.show()
#
#
# def show_label(label):
#     print(label.shape)
#     plt.figure()
#     labels = cm[label]
#     plt.imshow(labels)
#     plt.show()
#
#
# def show_label_2(label):
#     plt.figure()
#     # print(np.unique(label.numpy()))
#     label = label.numpy().astype('uint8')
#     label[label == 255] = 0
#     label = colormap[label]
#     print(label.shape)
#     plt.imshow(label)
#     plt.show()
#
#
# if __name__ == '__main__':
#     train_dataloader = get_dataloader(True, batch_size=2)
#
#     for images, labels in train_dataloader:
#         show_image(images[0])
#         show_label_2(labels[0])
#         break

