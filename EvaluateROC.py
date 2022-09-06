from torchensemble.utils import io
from models.r18 import ResNet18
from torchensemble import VotingClassifier
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

import os
import torch
import matplotlib.pyplot as plt

from utils import DataLoad, ImageTransforms

os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(1)

model = VotingClassifier(
    estimator=ResNet18,
    n_estimators=2,
    cuda=True,
)

Image_Transform = ImageTransforms()
train_loader, valid_loader = DataLoad(batch_size=128)

# Reload
io.load(model, save_dir='./Checkpoint/ResNet18_2')

ModeName = ['Train', 'Test']

for i, data_loader in enumerate([train_loader, valid_loader]):
    ModeNow = ModeName[i]
    Labels = []
    PredsStra = []
    for data, label in tqdm(data_loader):
        data, label = data.cuda(), label.cuda()
        Preds = model.predict(data)
        PredsStra.append(Preds[:, 1])
        Labels.append(label)

    PredsStra = torch.cat(PredsStra).cpu()
    Labels = torch.cat(Labels).cpu()

    FPR, TPR, Thresholds = roc_curve(Labels, PredsStra, pos_label=1)
    ROC_AUC = auc(FPR, TPR)

    plt.plot(FPR, TPR, label='{} Area={:.2f}'.format(ModeNow, ROC_AUC))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

print('Done!')