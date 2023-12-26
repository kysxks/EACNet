from torch.utils.data import DataLoader

from geoseg.losses import *
from geoseg.losses.ea_loss import ECELoss
from geoseg.losses.mix_loss import MixSoftmaxCrossEntropyOHEMLoss

from geoseg.datasets.vaihingen_dataset import *
from geoseg.models.BANet import BANet
from geoseg.models.EACNet import EACNet
#from geoseg.models.UNetFormer import UNetFormer
from geoseg.models.MAResUNet import MAResUNet
from geoseg.models.EaNet import EaNet
from geoseg.models.A2FPN import A2FPN
from geoseg.models.bisenet import BiSeNet
from geoseg.models.bisenetv2 import BiSeNetV2
from geoseg.models.modeling.deeplab import DeepLab
from geoseg.models.pointflow_resnet_with_max_avg_pool import DeepR50_PF_maxavg_deeply
from geoseg.models.MACUNet import MACUNet
from geoseg.models.ABCNet import ABCNet

from catalyst.contrib.nn import Lookahead
from catalyst import utils

# training hparam
max_epoch = 105
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 8
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01

accumulate_n = 1
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "unetformer-r18-512-crop-ms-e105"
weights_path = "model_weights/vaihingen/{}".format(weights_name)
test_weights_name = "lsk_4"
#test_weights_name = "joint_class"
log_name = 'vaihingen/{}'.format(weights_name)
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = [0]  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None
strategy = None

#  define the network

#net = EaNet(num_classes)
#net = BiSeNet(num_classes, backbone='resnet18')
#net = BiSeNetV2(num_classes)
#net = EaNet(n_classes=num_classes)
#net = DeepLab(num_classes=num_classes)
#net = A2FPN(class_num=num_classes)
#net = MACUNet(3, num_classes)
#net = ABCNet(3, num_classes)
#net = BANet(num_classes=num_classes)
net = EACNet(num_classes=num_classes)
#net = MAResUNet(num_channels = 3, num_classes=num_classes)
#net = DeepR50_PF_maxavg_deeply(num_classes=num_classes)
# define the loss

EA_loss = ECELoss(thresh=0.7, n_min=1*512*512//16, n_classes=num_classes, ignore_lb=ignore_index)

stage1_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)

stage2_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)

stage3_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)

ABCNetLoss = ABCNetLoss(ignore_index=ignore_index)

deeplab_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)


loss = UnetFormerLoss(ignore_index=ignore_index)

ohem_loss = MixSoftmaxCrossEntropyOHEMLoss(ignore_index=ignore_index)

ohem_loss_no_aux = MixSoftmaxCrossEntropyOHEMLoss(aux=False, ignore_index=ignore_index)

use_aux_loss = True

# define the dataloader

train_dataset = VaihingenDataset(data_root='../GeoSeg/data/vaihingen/train', mode='train',
                                 mosaic_ratio=0.25, transform=train_aug)

val_dataset = VaihingenDataset(transform=val_aug)
test_dataset = VaihingenDataset(data_root='../GeoSeg/data/vaihingen/test',
                                transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

