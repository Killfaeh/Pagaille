import torch

from suisei.deeplearning.dataset import *
from suisei.deeplearning.networks import *
from suisei.deeplearning.train import *

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"PyTorch   : {torch.__version__}")
print(f"Device    : {device}")

startEpoch = 0
endEpoch = 100
startBatch = 0
batchSize = 10
learningRate = 1e-3

inputDir = "/Volumes/AIstuff/Peguy/dataset/antialiasing/input/"
targetDir = "/Volumes/AIstuff/Peguy/dataset/antialiasing/target/"
modelDir = "/Volumes/AIstuff/Peguy/models/antialiasing/"

dataset = TrainDataset(inputDir, targetDir)
model = UNet512x512(n_channels=1).to(device)

trainUNet(dataset, model, modelDir, startEpoch, endEpoch, startBatch, batchSize, learningRate)