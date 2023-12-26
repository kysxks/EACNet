from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.potsdam_dataset import *
from geoseg.models.lsk_module_6 import UNetFormer
#from geoseg.models.EACNet import EACNet

#from geoseg.models.UNetFormer import UNetFormer
from geoseg.models.MAResUNet import MAResUNet
from geoseg.models.BANet import BANet
from geoseg.models.ABCNet import ABCNet
from geoseg.models.bisenet import BiSeNet
from geoseg.models.pspnet import PSPNet
from geoseg.models.bisenetv2 import BiSeNetV2
from geoseg.models.modeling.deeplab import DeepLab
#from geoseg.models.Point_class import UNetFormer
from geoseg.models.danet import DANet
from geoseg.models.pointflow_resnet_with_max_avg_pool import DeepR50_PF_maxavg_deeply
from catalyst.contrib.nn import Lookahead
from catalyst import utils

from loss.focal import Point_class_auxLoss

# training hparam
max_epoch = 45
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 8
lr = 1e-3
weight_decay = 2.5e-4
backbone_lr = 1e-4
backbone_weight_decay = 2.5e-4
accumulate_n = 1
num_classes = len(CLASSES)
classes = CLASSES

test_time_aug = 'd4'
output_mask_dir, output_mask_rgb_dir = None, None
weights_name = "unetformer-r18-768crop-ms-e45"
weights_path = "model_weights/potsdam/{}".format(weights_name)
#test_weights_name = "class_joint_edge _new_2"
test_weights_name = "lsk_5"
log_name = 'potsdam/{}'.format(weights_name)
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
gpus = [0]
strategy = None
pretrained_ckpt_path = None
resume_ckpt_path = None

#  define the network
#net = PSPNet(bins=(1, 2, 3, 6), dropout=0.1, classes=num_classes, zoom_factor=1, use_ppm=True, pretrained=True)
#net = ABCNet(3, n_classes=num_classes)
#net = BiSeNetV2(num_classes)
#net = DeepLab(num_classes=num_classes)
#net = BiSeNet(num_classes, backbone='resnet18')
net = EACNet(num_classes=num_classes)
#net = MAResUNet(num_channels = 3, num_classes=num_classes)
#net = BANet(num_classes=num_classes)
#net = DeepR50_PF_maxavg_deeply(num_classes=num_classes)
#net = DANet(nclass=6, aux=True)
# define the loss
stage1_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)

stage2_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)

stage3_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
                 

criterion3_1 = Point_class_auxLoss(ignore_index=ignore_index, bin_size=(4,4))
criterion3_2 = Point_class_auxLoss(ignore_index=ignore_index, bin_size=(8,8))
criterion3_3 = Point_class_auxLoss(ignore_index=ignore_index, bin_size=(16,16))

loss = UnetFormerLoss(ignore_index=ignore_index)

deeplab_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)

ABCNetLoss = ABCNetLoss(ignore_index=ignore_index)

use_aux_loss = False

# define the dataloader

train_dataset = PotsdamDataset(data_root='../GeoSeg/data/po/train', mode='train',
                               mosaic_ratio=0.25, transform=train_aug)

val_dataset = PotsdamDataset(transform=val_aug)
test_dataset = PotsdamDataset(data_root='../GeoSeg/data/po/test',
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

