import imageio
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from Res2Net import res2net50_v1b_26w_4s
from Net_v1 import Network as Net
from utils.dataloader import test_dataset
import torch.nn as nn
import time
import cv2
from tqdm import tqdm
import torchvision.utils as vutils
from torch.autograd import Variable
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=448, help='testing size')
parser.add_argument('--pth_path', type=str, default='/home/q/ours/ZCX/FGNet/checkpoints/FBv2-xrbii/Net_epoch_best.pth')
# /home/q/ours/ZCX/v4/Net_epoch_best.pth
#/home/q/ours/ZCX/FGNet/v0_rf/adjust_lr_80_de20/Net_epoch_best.pth refine 0.65
str='/home/q/checkpoints/AFBNet/AFBNet-'
#/home/q/checkpoints/AFBNet-/modelv10-confr1
#CHAMELEON CAMO COD10K AFBNet_epoch_best NC4K
# pip install pysodmetrics

import os
import timm
import pandas as pd
import py_sod_metrics as pysodmetrics
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

def excel_save(name,eval_params):
    path='/media/fiona/3bab5e8f-eee1-471f-9466-f383b18459a0/zhangchenxi/result/txt/'
    df = pd.DataFrame.from_dict(eval_params, orient='index', columns=['Value'])
    excel_file =path+name+'evaluation_results.xlsx'
    sheet_name =name
    df.to_excel(excel_file, sheet_name=sheet_name)

#CHAMELEON CAMO
#'CAMO','COD10K','NC4K','CHAMELEON'

    
def foreground_sign(pred):
    b, c, w, h = pred.size()
    p = pred.gt(0).float()
    #这一行将 pred 张量中大于零的元素设置为 1，小于等于零的元素设置为 0，并将结果转换为浮点型。这样可以得到一个二值化的张量 p，其中前景部分被设置为 1，背景部分被设置为 0。
    num_pos = p[:, :, 0, 0] + p[:, :, w - 1, 0] + p[:, :, w - 1, h - 1] + p[:, :, 0, h - 1]
    #这段代码的目的是计算每个图像在四个角位置的像素值之和，存储在 num_pos 变量中。具体而言，它通过对张量 p 进行索引来获取四个角的像素值，并将它们相加。
    sign = ((num_pos < 2).float() * 2 - 1).view(b, c, 1, 1)
    return sign
# weights_dir = '/home/q/checkpoints/AFBNet-9/'
# weights_files = [f for f in os.listdir(weights_dir) if f.endswith('.pth')]
from pvtv2 import pvt_v2_b4
for _data_name in ['CAMO','COD10K','NC4K','CHAMELEON']:
    #'CAMO','COD10K','NC4K'
    #data_path = './data/TestDataset/{}/'.format(_data_name)
     print(_data_name)
     data_path = '/home/q/ours/ZYL/BGNet-master/data/TestDataset/'
     save_path = '/home/q/ours/ZCX/results/{}/'.format(_data_name)
     opt = parser.parse_args()
   
   #  for weight_file in weights_files:
   #   weight_path = os.path.join(weights_dir, weight_file)
   #   print(weight_path)
    # 加载模型权重
     resnet=pvt_v2_b4()
     config = {}
     schedule = {}
     path = '/home/q/ours/ZCX/HASNet/pvt_v2_b4.pth'
     save_model = torch.load(path)
     model_dict = resnet.state_dict()
     state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
     model_dict.update(state_dict)
     resnet.load_state_dict(model_dict)
     encoder = resnet
     fl = [64,128,320,512]
    
     model = Net(config,resnet,fl)
     model.load_state_dict(torch.load(opt.pth_path))
     model.cuda()
     model.eval()
     os.makedirs(save_path, exist_ok=True)
     image_root = data_path+'/{}/Imgs/'.format(_data_name)
     gt_root = data_path+'/{}/GT/'.format(_data_name)
     test_loader = test_dataset(image_root, gt_root, opt.testsize)
     time_t = 0.0
     for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        image = Variable(image).cuda()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        time_start = time.time()
        o11,o22,o33,e11= model(image)
      #   e=e2-e1
      #   o1=o1
        torch.cuda.synchronize()
        time_end = time.time()
        time_t = time_t + time_end - time_start
        # print(o11.size(),gt.shape)
        pred = F.interpolate(o11, gt.shape, mode='bilinear', align_corners=False)
        pred = pred.sigmoid().data.cpu().numpy().squeeze()  # N*H*W
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        cv2.imwrite(save_path + name, pred * 255)
     fps = test_loader.size / time_t
     print(fps)
     mask_root = '/home/q/ours/ZYL/BGNet-master/data/TestDataset/{}/GT/'.format(_data_name)

     pred_root = '/home/q/ours/ZCX/results/{}/'.format(_data_name)
     print(pred_root)
     mask_name_list = sorted(os.listdir(mask_root))
     FM = Fmeasure()
     WFM = WeightedFmeasure()
     SM = Smeasure()
     EM = Emeasure()
     M = MAE()


    # ------------------------------------------------------
     sample_gray = dict(with_adaptive=True, with_dynamic=True)
     sample_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=True)
     overall_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=False)
    #FMv2 = py_sod_metrics.FmeasureV2(
    #     metric_handlers={
    #         "iou": py_sod_metrics.IOUHandler(**sample_gray),
    #         "dice": py_sod_metrics.DICEHandler(**sample_gray),

    #         # 二值化数据指标的特殊情况一：各个样本独立计算指标后取平均
    #         # "sample_biiou": py_sod_metrics.IOUHandler(**sample_bin),
    #         # "sample_bidice": py_sod_metrics.DICEHandler(**sample_bin),

    #         "overall_biiou": py_sod_metrics.IOUHandler(**overall_bin),
    #         "overall_bidice": py_sod_metrics.DICEHandler(**overall_bin),
    #     }
    # )
    # ------------------------------------------------------

     for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
        mask_path = os.path.join(mask_root, mask_name)
        pred_path = os.path.join(pred_root, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        M.step(pred=pred, gt=mask)
        #FMv2.step(pred=pred, gt=mask)

     fm = FM.get_results()["fm"]
     wfm = WFM.get_results()["wfm"]
     sm = SM.get_results()["sm"]
     em = EM.get_results()["em"]
     mae = M.get_results()["mae"]
    # ------------------------------------------------------
    #fmv2 = FMv2.get_results()
    # ------------------------------------------------------
     results = {
        # ------------------------------------------------------
        # "meandice": fmv2["dice"]["dynamic"].mean(),
        # "meaniou": fmv2["iou"]["dynamic"].mean(),
        # ------------------------------------------------------
        "MAE": mae,
        "wFmeasure": wfm,
        "Smeasure": sm,
        "adpEm": em["adp"],
        "meanEm": em["curve"].mean(),
        "maxEm": em["curve"].max(),
        "adpFm": fm["adp"],
        "meanFm": fm["curve"].mean(),
        "maxFm": fm["curve"].max(),

     }

     print(results)
