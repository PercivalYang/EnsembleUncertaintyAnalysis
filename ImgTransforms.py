from torchvision.transforms import transforms
from models.r18 import ResNet18
from torchensemble import VotingClassifier
from torchensemble.utils import io
from tqdm import tqdm

from utils import DataLoad, EnsembleFoward, Expand2SquareImg

import torch
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

data_path = "/home/yangziyin/Datasets/Cropped"
num_iter = 16
Pred_list = []

# Initialize model
model = VotingClassifier(
    estimator=ResNet18,
    n_estimators=2,
    cuda=True,
)

# Load trained model
io.load(model, save_dir='../Checkpoint/CroppedDataset/Expand2Square/ResNet18_2')

# set new image transforms
image_transforms = transforms.Compose([
    # transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    transforms.ColorJitter(brightness=.3, contrast=.3, saturation=.3),
    Expand2SquareImg(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

for i in range(num_iter):
    print('The {} iters.'.format(i))
    Batch_list = []
    # Load Data
    valid_data = DataLoad(data_path, batch_size=32, CustImgTransf=image_transforms)['Test']
    for data, _ in tqdm(valid_data):
        data = data.cuda()
        Batch_list.append(EnsembleFoward(model=model, data=data))
    Pred_list.append(torch.cat(Batch_list))

Pred = torch.stack(Pred_list, 3)  # (num_fig, num_class, num_estimators, num_iters)

# (num_fig, num_class, num_estimators * num_iters)，即将网络个数和变换个数的预测集合在一个维度，方便做不确定性分析
PredNewView = Pred.view(len(Pred), 2, -1)

Labels = torch.tensor(valid_data.dataset.targets)

# 找出错误预测的样例
Pred_Labels = PredNewView.mean(-1).argmax(1)
FalsePredIndex = torch.where(~(Pred_Labels.cpu() == Labels))[0]
FalsePredData = PredNewView[FalsePredIndex, :, :]

for i in range(len(FalsePredData)):
    SingleFPD = FalsePredData[i,:,:].T
    FPredSoft = F.softmax(SingleFPD, dim=1)
    StraProbs = FPredSoft[:,1]
    plt.hist(StraProbs.cpu().numpy())
    plt.xlabel('P(Strabismus)')
    plt.ylabel('Density')
    plt.show()

    # # 不确定性指标分析
    # FLog = -FPredSoft.clamp_min(1e-6).log()
    # FEntropy = FLog * FPredSoft
    # FStd = SingleFPD.std(0)

# 通过Debug可知
