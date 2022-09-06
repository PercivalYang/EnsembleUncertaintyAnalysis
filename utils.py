import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import PIL.Image as Image

import torch
import os

from torch.utils.data import DataLoader
from torchvision import datasets

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def ImageTransforms(mode):
    '''
    mode 为挑选ImageTransform的类型，'full'针对包含全部背景的图片，'cropped'针对仅包含眼睛区域的图片
    '''
    if mode == 'full':
        image_transforms = {
            'Train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            'Test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        }
        return image_transforms
    elif mode == 'cropped':
        image_transforms = {
            'Train': transforms.Compose([
                Expand2SquareImg(), # 其中自带Resize((224,224))
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            'Test': transforms.Compose([
                Expand2SquareImg(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        }
        return image_transforms

def DataLoad(DatasetPath, batch_size, img_mode):
    '''
    DatasetPath: Full eyes(~/Datasets/TrainValidDataset), Cropped eyes(~/Datasets/Cropped)
    '''
    Train_directory = os.path.join(DatasetPath, 'train')
    test_directory = os.path.join(DatasetPath, 'valid')

    data = {
        'Train': datasets.ImageFolder(root=Train_directory, transform=ImageTransforms(img_mode)['Train']),
        'Test': datasets.ImageFolder(root=test_directory, transform=ImageTransforms(img_mode)['Test'])
    }

    Train_data = DataLoader(data['Train'], batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_data = DataLoader(data['Test'], batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    DataTotal = {"Train":Train_data,"Test":test_data}

    return DataTotal

def AddDropoutLayer(Net):
    '''
    Add Dropout Layer behind each Conv2d Layer, only use for VGG network now.
    Note:
        1. Gradient explode will happen if network is training and using this method;
        2. The results of MC-Dropout by using this method is bad, the accuracy drop down hardly.
    '''
    feats_list = list(Net.features)
    new_feats_list = []
    for feat in feats_list:
        new_feats_list.append(feat)
        if isinstance(feat, nn.Conv2d):
            new_feats_list.append(nn.Dropout(p=0.5, inplace=True))
    Net.features = nn.Sequential(*new_feats_list)
    return Net


def ForwardStdEntropyEvalute(model, data):

    '''
    用于训练好的模型做前向传播，并计算模型的不确定性包括方差和预测熵在内的数值
    model: Ensemble model
    data: tensor格式为(batch_size, Channel, width, height)
    Metrics(dict type, includes Std and Entropy), Outputs(batch_size, num_class, num_estimators)
    '''

    outputs_list = []
    for estimator in model.estimators_:
        estimator.eval()
        with torch.no_grad():
            outputs_list.append(F.softmax(estimator(data), dim=1))
            
    Outputs = torch.stack(outputs_list,2)
    LogOut = Outputs.clamp_min(1e-6).log()
    OutStd = Outputs.std(dim=2).mean(dim=1)
    OutEntropy = Outputs * LogOut

    Metrics = {'std': OutStd,'entropy':OutEntropy}
    return Metrics, Outputs

class Expand2SquareImg:
    '''
    针对长宽比较大的眼睛区域的图片，使用该方法能在图片不发生失真的情况下，满足图片以规定尺寸大小输入网络。
    '''
    def __call__(self, PILImg):
        width, height = PILImg.size
        if width == height:
            return PILImg
        elif width > height:
            new_img = Image.new(PILImg.mode, (width,width)) # 创建一个正方形黑色背景
            new_img.paste(PILImg, (0, (width - height) // 2))   # 将原图粘贴过来
        else:
            new_img = Image.new(PILImg.mode, (width,width))
            new_img.paste(PILImg, (0, (height - width) // 2, 0))
        new_img = new_img.resize((224,224))
        return new_img