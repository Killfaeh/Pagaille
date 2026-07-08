import torch

from suisei.deeplearning.dataset import *
from suisei.deeplearning.networks import *
from suisei.deeplearning.discriminator import *
from suisei.deeplearning.train import *

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"PyTorch   : {torch.__version__}")
print(f"Device    : {device}")

startEpoch = 0
endEpoch = 100
startBatch = 3300
batchSize = 5
learningRate = 1e-4

inputDir = "/Volumes/AIstuff/Peguy/dataset/inking/all/pencil/"
targetDir = "/Volumes/AIstuff/Peguy/dataset/inking/all/ink/"
modelDir = "/Volumes/AIstuff/Peguy/models/inking/"

dataset = GlobalLocalTrainDataset(inputDir, targetDir, 512, 5)
model = InkingModel(n_channels=1).to(device)
trainGlobalLocalUNet(dataset, model, modelDir, startEpoch, endEpoch, startBatch, batchSize, learningRate)