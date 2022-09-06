from torchensemble.utils import io
from models.r18 import ResNet18
from torchensemble import VotingClassifier

from utils import DataLoad, ImageTransforms, ForwardStdEntropyEvalute

model = VotingClassifier(
    estimator=ResNet18,
    n_estimators=10,
    cuda=True,
)

Image_Transform = ImageTransforms()
train_loader, valid_loader = DataLoad(batch_size=128)

# Reload
io.load(model, save_dir='./Checkpoint')

# Evaluating
for data, label in valid_loader:
    data, label = data.cuda(), label.cuda()
    Metrics = ForwardStdEntropyEvalute(model, data) # Metrics includes {'std','entropy'}
    # RightF = (pred == label)