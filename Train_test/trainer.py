###################################################
#
#
###################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class Trainer(object):
    '''
    用于训练和验证
    '''
    def __init__(self):
        self.seg_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.running_score = None
