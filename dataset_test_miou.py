import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from nets.HRnet_ocr import get_net as HRnet_ocr
from utils.UtilLoss import CE_Loss, Dice_loss, LossHistory, weights_init
import yaml
from data_generators.data_generator import initialize_data_loader
from utils.summaries import TensorboardSummary
from utils.saver import Saver
from utils.metrics import Evaluator

from utils.replicate import patch_replication_callback

import warnings

warnings.filterwarnings("ignore")
import scipy.misc
import os
import matplotlib
####### tensor -> img
import torch.nn.functional as F
from PIL import Image




# validation(net=model, val_loader=val_loader, config=config, writer=writer, saver=saver, epoch=epoch)
def validation(net, val_loader, config, writer, saver, epoch):
    global best_pred
    net.eval()
    # evaluator.reset()
    # print('Start Validation')
    evaluator.reset()
    tbar = tqdm(val_loader, desc='\r')
    val_loss = 0.0
    for i, sample in enumerate(tbar):
        image, target, target_onehot = sample['image'], sample['label'], sample['seg_labels']
        if i ==1:

            print(sample['name'][0],sample['h'].item(),sample['w'].item())


        target_ = target.type(torch.FloatTensor).long()

        if config['network']['use_cuda']:
            image, target, target_, target_onehot = image.cuda(), target.cuda(), target_.cuda(), target_onehot.cuda()

        with torch.no_grad():
            if aux_branch:
                aux_outputs, outputs = net(image)
                aux_loss = CE_Loss(aux_outputs, target_, num_classes=NUM_CLASSES)
                main_loss = CE_Loss(outputs, target_, num_classes=NUM_CLASSES)
                val_loss = aux_loss + main_loss
                if dice_loss:
                    aux_dice = Dice_loss(aux_outputs, target_onehot)
                    main_dice = Dice_loss(outputs, target_onehot)
                    val_loss = val_loss + aux_dice + main_dice
            else:
                outputs = net(image)

                val_loss = CE_Loss(outputs, target_, num_classes=NUM_CLASSES)
                if dice_loss:
                    main_dice = Dice_loss(outputs, target_onehot)

                    val_loss = val_loss + main_dice

        val_loss += val_loss.item()
        tbar.set_description('Val loss: %.3f' % (val_loss / (i + 1)))
        img_save = F.interpolate(outputs,(sample['h'].item(),sample['w'].item()))
        #print(img_save.shape)
        pred = outputs.data.cpu().numpy()
        img_save = img_save.data.cpu().numpy()
        #记录名称
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1).astype(np.uint8)
        img_save = np.argmax(img_save, axis=1).astype(np.uint8)

        im = Image.fromarray(img_save[0])
        im.save(os.path.join('./deepfashion2_select160/wo', '%s.png' % (sample['name'][0])))


            #matplotlib.image.imsave(os.path.join('./deepfashion2_select160/wo', '%d.png' % (i)),target)



        # Add batch sample into evaluator
        evaluator.add_batch(target, pred)

    # Fast test during the training
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()



    print("Validation:")
    print("[Epoch: %d, numImages: %5d" % (epoch, i * config['training']['batch_size'] + image.data.shape[0]))
    print("Acc: {}, Acc_class: {}, mIoU: {}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))

    new_pred = mIoU

    return new_pred


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == "__main__":
    with open("configs/config_hrnet_three_attention.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    log_dir = "logs/"
    # ------------------------------#
    #   输入图片的大小
    # ------------------------------#
    inputs_size = [512, 512, 3]  # hrnet+ocr
    # ---------------------#
    #   分类个数+1
    #   2+1
    # ---------------------#
    NUM_CLASSES = config['network']['num_classes']
    # --------------------------------------------------------------------#
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    # ---------------------------------------------------------------------#
    dice_loss = False

    pretrained = False

    aux_branch = False

    model = HRnet_ocr(config)

    if not pretrained:
        weights_init(model)

    # -------------------------------------------#
    #   权值文件的下载请看README
    #   权值和主干特征提取网络一定要对应
    # -------------------------------------------#
    # model_path = "model_data/pspnet_mobilenetv2.pth"
    # # 加快模型训练的效率
    # print('Loading weights into state dict...')
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load(model_path)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    # print('Finished!')

    loss_history = LossHistory("logs/")

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    # net = torch.nn.DataParallel(model)
    # # patch_replication_callback(net)
    # net = net.cuda()

    # Using cuda   是否使用cuda  在config['network']['use_cuda'] 修改
    if config['network']['use_cuda']:
        model = torch.nn.DataParallel(model)
        patch_replication_callback(model)
        model = model.cuda()

    # start

    writer = TensorboardSummary(config['training']['tensorboard']['log_dir']).create_summary()

    saver = Saver(config)

    evaluator = Evaluator(config['network']['num_classes'])

    best_pred = 0.0

    # end write

    mIoU = 0.0
    if True:
        lr = 1e-4
        optimizer = optim.Adam(model.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95, last_epoch=-1)
        # lr_scheduler  = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[round(config['training']['epochs'] * x) for x in [0.8, 0.9]])

        # 'epoch': epoch + 1,
        # 'state_dict': net.state_dict(),
        # 'optimizer': optimizer.state_dict(),
        # 'best_pred': best_pred,
        path = './experiments/checkpoint_best.pth'
        checkpoint = torch.load(path)
        continue_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_pred = checkpoint['best_pred']

        # dataloader
        # train_loader, val_loader, test_loader, nclass = initialize_data_loader(config)
        train_loader, val_loader = initialize_data_loader(config)
        # for param in model.backbone.parameters():
        #     param.requires_grad = False

        # criterion = SegmentationLosses(weight=weight, cuda=self.config['network']['use_cuda']).build_loss(
        #     mode=self.config['training']['loss_type'])
        print("\nTotal Epoches: {}".format(config['training']['epochs']))

            #training(net=model, train_loader=train_loader, config=config, writer=writer, saver=saver, epoch=epoch)
        mIoU = validation(net=model, val_loader=val_loader, config=config, writer=writer, saver=saver,
                                  epoch=0)
        print('----Test Dataset mIoU:{}'.format(mIoU))

