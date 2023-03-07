from glob import glob
import os
import shutil

flag = 0
# flag = 1
if flag:

    all_train_image_path = "/home/user/Downloads/gc/deepfashion77848/mymodel/datasets/deepfashion2_select77848/train/image"
    all_train_label_path = "/home/user/Downloads/gc/deepfashion77848/mymodel/datasets/deepfashion2_select77848/train/label"

    all_val_image_path = "/home/user/Downloads/gc/deepfashion77848/mymodel/datasets/deepfashion2_select77848/val/image"
    all_val_label_path = "/home/user/Downloads/gc/deepfashion77848/mymodel/datasets/deepfashion2_select77848/val/label"

    all_test_image_path = "/home/user/Downloads/gc/deepfashion77848/mymodel/datasets/deepfashion2_select77848/test/image"
    all_test_label_path = "/home/user/Downloads/gc/deepfashion77848/mymodel/datasets/deepfashion2_select77848/test/label"

    new_image_path = "/home/user/Downloads/gc/pspnet/pspnet-pytorch-master/VOCdevkit/VOC2007/JPEGImages"
    new_label_path = "/home/user/Downloads/gc/pspnet/pspnet-pytorch-master/VOCdevkit/VOC2007/SegmentationClass"

    all_train_image_ids = glob(os.path.join(all_train_image_path, '*' + '.jpg'))
    all_train_image_ids = [os.path.splitext(os.path.basename(p))[0] for p in all_train_image_ids]

    print(len(all_train_image_ids))

    num = 0

    for id in all_train_image_ids:
        num = num + 1
        img_src = os.path.join(all_train_image_path, id + '.jpg')
        label_src = os.path.join(all_train_label_path, id + '.png')

        image_dst = os.path.join(new_image_path, id + '.jpg')
        label_dst = os.path.join(new_label_path, id + '.png')

        shutil.copy(img_src, image_dst)
        shutil.copy(label_src, label_dst)

        # if num == 10:
        #     break
        if num % 10000 == 0:
            print(num)


    print(num)

flag = 0
# flag = 1
if flag:

    all_train_image_path = "/home/user/Downloads/gc/deepfashion77848/mymodel/datasets/deepfashion2_select77848/train/image"
    all_train_label_path = "/home/user/Downloads/gc/deepfashion77848/mymodel/datasets/deepfashion2_select77848/train/label"

    all_val_image_path = "/home/user/Downloads/gc/deepfashion77848/mymodel/datasets/deepfashion2_select77848/val/image"
    all_val_label_path = "/home/user/Downloads/gc/deepfashion77848/mymodel/datasets/deepfashion2_select77848/val/label"

    all_test_image_path = "/home/user/Downloads/gc/deepfashion77848/mymodel/datasets/deepfashion2_select77848/test/image"
    all_test_label_path = "/home/user/Downloads/gc/deepfashion77848/mymodel/datasets/deepfashion2_select77848/test/label"

    new_image_path = "/home/user/Downloads/gc/pspnet/pspnet-pytorch-master/VOCdevkit/VOC2007/JPEGImages"
    new_label_path = "/home/user/Downloads/gc/pspnet/pspnet-pytorch-master/VOCdevkit/VOC2007/SegmentationClass"

    all_val_image_ids = glob(os.path.join(all_val_image_path, '*' + '.jpg'))
    all_val_image_ids = [os.path.splitext(os.path.basename(p))[0] for p in all_val_image_ids]

    print(len(all_val_image_ids))

    num = 0

    for id in all_val_image_ids:
        num = num + 1
        img_src = os.path.join(all_val_image_path, id + '.jpg')
        label_src = os.path.join(all_val_label_path, id + '.png')

        image_dst = os.path.join(new_image_path, id + '.jpg')
        label_dst = os.path.join(new_label_path, id + '.png')

        shutil.copy(img_src, image_dst)
        shutil.copy(label_src, label_dst)

        # if num == 10:
        #     break
        if num % 1000 == 0:
            print(num)


    print(num)


flag = 0
# flag = 1
if flag:

    all_train_image_path = "/home/user/Downloads/gc/deepfashion77848/mymodel/datasets/deepfashion2_select77848/train/image"
    all_train_label_path = "/home/user/Downloads/gc/deepfashion77848/mymodel/datasets/deepfashion2_select77848/train/label"

    all_val_image_path = "/home/user/Downloads/gc/deepfashion77848/mymodel/datasets/deepfashion2_select77848/val/image"
    all_val_label_path = "/home/user/Downloads/gc/deepfashion77848/mymodel/datasets/deepfashion2_select77848/val/label"

    all_test_image_path = "/home/user/Downloads/gc/deepfashion77848/mymodel/datasets/deepfashion2_select77848/test/image"
    all_test_label_path = "/home/user/Downloads/gc/deepfashion77848/mymodel/datasets/deepfashion2_select77848/test/label"

    new_image_path = "/home/user/Downloads/gc/pspnet/pspnet-pytorch-master/VOCdevkit/VOC2007/JPEGImages"
    new_label_path = "/home/user/Downloads/gc/pspnet/pspnet-pytorch-master/VOCdevkit/VOC2007/SegmentationClass"

    all_test_image_ids = glob(os.path.join(all_test_image_path, '*' + '.jpg'))
    all_test_image_ids = [os.path.splitext(os.path.basename(p))[0] for p in all_test_image_ids]

    print(len(all_test_image_ids))

    num = 0

    for id in all_test_image_ids:
        num = num + 1
        img_src = os.path.join(all_test_image_path, id + '.jpg')
        label_src = os.path.join(all_test_label_path, id + '.png')

        image_dst = os.path.join(new_image_path, id + '.jpg')
        label_dst = os.path.join(new_label_path, id + '.png')

        shutil.copy(img_src, image_dst)
        shutil.copy(label_src, label_dst)

        # if num == 10:
        #     break
        if num % 1000 == 0:
            print(num)


    print(num)




flag = 0
# flag = 1
if flag:

    project_path = os.path.dirname(os.path.abspath(""))

    test_image_path = os.path.join(project_path, "datasets/image")

    print(test_image_path)

    all_test_image_ids = glob(os.path.join(test_image_path, '*' + '.jpg'))
    all_test_image_ids = [os.path.splitext(os.path.basename(p))[0] for p in all_test_image_ids]

    print(len(all_test_image_ids))



def createFilelist(images_path, text_save_path):
    # 打开图片列表清单txt文件
    file_name = open(text_save_path, "w")
    # 查看文件夹下的图片
    images_name = os.listdir(images_path)
    # 遍历所有文件
    for eachname in images_name:
        # 按照需要的格式写入目标txt文件
        a = eachname.split('.')
        file_name.write(a[0] + '\n')

    print('生成txt成功！')

    file_name.close()

# flag = 0
flag = 1
if flag:

    all_train_image_path = "/home/user/Downloads/gc/deepfashion77848/mymodel/datasets/deepfashion2_select77848/train/image"
    all_train_label_path = "/home/user/Downloads/gc/deepfashion77848/mymodel/datasets/deepfashion2_select77848/train/label"

    all_val_image_path = "/home/user/Downloads/gc/deepfashion77848/mymodel/datasets/deepfashion2_select77848/val/image"
    all_val_label_path = "/home/user/Downloads/gc/deepfashion77848/mymodel/datasets/deepfashion2_select77848/val/label"

    all_test_image_path = "/home/user/Downloads/gc/deepfashion77848/mymodel/datasets/deepfashion2_select77848/test/image"
    all_test_label_path = "/home/user/Downloads/gc/deepfashion77848/mymodel/datasets/deepfashion2_select77848/test/label"

    text_save_path = "/home/user/Downloads/gc/pspnet/pspnet-pytorch-master/testcode/trainval.txt"

    createFilelist(all_test_image_path, text_save_path)




