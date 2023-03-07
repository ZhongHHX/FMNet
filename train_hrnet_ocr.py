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


# training(net=model, train_loader=train_loader, config=config, writer=writer, saver=saver, epoch=epoch)
def training(net, train_loader, config, writer, saver, epoch):
# def training(config, epoch):

    train_loss = 0.0
    net.train()
    tbar = tqdm(train_loader)
    num_img_tr = len(train_loader)
    for i, sample in enumerate(tbar):
        image, target, target_onehot = sample['image'], sample['label'], sample['seg_labels']
        # print(type(image), image.shape)
        # print(type(target), target.shape)
        # print(type(target_onehot), target_onehot.shape)
        #                   imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
        #                 pngs = torch.from_numpy(pngs).type(torch.FloatTensor).long()
        #                 labels = torch.from_numpy(labels).type(torch.FloatTensor)

        target_ = target.type(torch.FloatTensor).long()

        if config['network']['use_cuda']:
            image, target, target_, target_onehot = image.cuda(), target.cuda(), target_.cuda(), target_onehot.cuda()

        optimizer.zero_grad()
        if aux_branch:
            aux_outputs, outputs = net(image)
            aux_loss  = CE_Loss(aux_outputs, target_, num_classes = NUM_CLASSES)
            main_loss = CE_Loss(outputs, target_, num_classes=NUM_CLASSES)
            loss      = aux_loss + main_loss
            if dice_loss:
                aux_dice  = Dice_loss(aux_outputs, target_onehot)
                main_dice = Dice_loss(outputs, target_onehot)
                loss      = loss + aux_dice + main_dice

        else:
            outputs = net(image)
            loss    = CE_Loss(outputs, target_, num_classes=NUM_CLASSES)
            if dice_loss:
                main_dice = Dice_loss(outputs, target_onehot)
                loss      = loss + main_dice

        loss.backward()
        optimizer.step()
        lr = get_lr(optimizer)
        train_loss  += loss.item()
        tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
        writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

    writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
    print('[Epoch: %d, numImages: %5d]' % (epoch, i * config['training']['batch_size'] + image.data.shape[0]))
    # print('[Epoch: %d, numImages: %5d]' % (epoch, i * config['training']['batch_size'] + image.data.shape[0]))
    print('Loss: %.3f' % train_loss)

    #save last checkpoint

    saver.save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_pred': best_pred,
    }, is_best = False, filename='checkpoint_last.pth')


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
        pred = outputs.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)

        # Add batch sample into evaluator
        evaluator.add_batch(target, pred)

    # Fast test during the training
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

    writer.add_scalar('val/total_loss_epoch', val_loss, epoch)
    writer.add_scalar('val/mIoU', mIoU, epoch)
    writer.add_scalar('val/Acc', Acc, epoch)
    writer.add_scalar('val/Acc_class', Acc_class, epoch)
    writer.add_scalar('val/fwIoU', FWIoU, epoch)

    print("Validation:")
    print("[Epoch: %d, numImages: %5d" % (epoch, i * config['training']['batch_size'] + image.data.shape[0]))
    print("Acc: {}, Acc_class: {}, mIoU: {}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))

    new_pred = mIoU
    if new_pred > best_pred:
        best_pred = new_pred
        saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_pred': best_pred,
        }, is_best = True, filename='checkpoint_best.pth')
    return new_pred

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']





if __name__ == "__main__":
    with open("configs/config_hrnet_ocr.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    log_dir = "logs/"
    # ------------------------------#
    #   输入图片的大小
    # ------------------------------#
    inputs_size = [512, 512, 3]  #hrnet+ocr
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
        #lr_scheduler  = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[round(config['training']['epochs'] * x) for x in [0.8, 0.9]])


        # dataloader
        #train_loader, val_loader, test_loader, nclass = initialize_data_loader(config)
        train_loader, val_loader = initialize_data_loader(config)
        # for param in model.backbone.parameters():
        #     param.requires_grad = False

        # criterion = SegmentationLosses(weight=weight, cuda=self.config['network']['use_cuda']).build_loss(
        #     mode=self.config['training']['loss_type'])
        print("\nTotal Epoches: {}".format(config['training']['epochs']))
        for epoch in range(config['training']['start_epoch'], config['training']['epochs']):
            print('\n=>Epoches %i, learning rate = %.8f, \
                            previous best = %.8f' % (epoch, optimizer.param_groups[0]['lr'], best_pred))
            training(net=model, train_loader=train_loader, config=config, writer=writer, saver=saver, epoch=epoch)
            if not config['training']['no_val'] and epoch % config['training']['val_interval'] == (config['training']['val_interval'] - 1):
                mIoU = validation(net=model, val_loader=val_loader, config=config, writer=writer, saver=saver, epoch=epoch)
            loss_history.append_mIoU(mIoU)
            lr_scheduler.step()

        print('previous best = %.8f'  % (best_pred))
