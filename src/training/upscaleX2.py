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
startBatch = 0
batchSize = 10
learningRate = 1e-3

inputDir = "/Volumes/AIstuff/Peguy/dataset/upscaleLineartX2/input/"
targetDir = "/Volumes/AIstuff/Peguy/dataset/upscaleLineartX2/target/"
modelDir = "/Volumes/AIstuff/Peguy/models/upscaleLineartX2/"

dataset = TrainDataset(inputDir, targetDir)

#### Version simple ####

model = UNet512To1024(n_channels=1).to(device)
trainUNet(dataset, model, modelDir, startEpoch, endEpoch, startBatch, batchSize, learningRate)

#### Version Pix2Pix ####

#G = UNet512To1024(n_channels=1).to(device)
#D = GANDiscriminator512To1024(n_channels=1).to(device)
#trainPix2Pix(dataset, G, D, modelDir, startEpoch, endEpoch, startBatch, batchSize, learningRate)