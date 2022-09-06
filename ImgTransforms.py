from torchvision.transforms import transforms
from PIL import Image
from models.r18 import ResNet18
from torchensemble import VotingClassifier
from torchensemble.utils import io
from tqdm import tqdm

import torch

img_path = "/home/yangziyin/Datasets/TrainValidDataset/valid/normal/DSC00669.JPG"
test_img = Image.open(img_path)
label = 0
num_iter = 128
Pred_list = []

# Initialize model
model = VotingClassifier(
    estimator=ResNet18,
    n_estimators=5,
    cuda=True,
)

# Load trained model
io.load(model, save_dir='../Checkpoint/ResNet18_5')

for i in tqdm(range(num_iter)):
    image_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=.5, hue=.3),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = image_transforms(test_img).unsqueeze(0)
    Pred_list.append(model.predict(img))

Pred = torch.cat(Pred_list)
PredLog = Pred.log()
Pred_mean = Pred.mean(0)
Pred_std = Pred.std(0).mean()

