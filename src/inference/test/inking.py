import torch
import numpy as np
import cv2

from suisei.deeplearning.dataset import *
from suisei.deeplearning.networks import *
from suisei.deeplearning.checkpoint import *
from suisei.deeplearning.imagepreprocessing import *
from suisei.deeplearning.inference import *

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"PyTorch   : {torch.__version__}")
print(f"Device    : {device}")

modelNum = 0
index = 300

modelPath = "/Volumes/AIstuff/Peguy/models/inking/model_{}_{}.pth".format(modelNum, index)
#modelPath = "/Volumes/AIstuff/Peguy/models/inking/model_{}.pth".format(modelNum)

#inputFile = "/Users/suisei/Desktop/Spock-line-002.png"
#inputFile = "/Users/suisei/Workspace/01_DessinIllustration-TravailEnCours/DPSchool/Challenges/Challenge015/03-Illustrations/Lirana-Crayon-scan.jpg"
inputFile = "/Users/suisei/Workspace/01_DessinIllustration-TravailEnCours/StarTrek/Inktober2023/Inktober2023-001-scan.jpeg"

outputFile = "/Users/suisei/Desktop/test-inking-{}.png".format(modelNum)

model = InkingModel(n_channels=1).to(device)
load_model(model, modelPath)

npImg = cv2.imread(str(inputFile), cv2.IMREAD_GRAYSCALE)
npImg = inkingPreprocessing(npImg)

cv2.imwrite("/Users/suisei/Desktop/processedInput.png", npImg)

result = execGlobalLocalUNet(model, npImg, tile_size=512, overlap=64, device=device)

cv2.imwrite(str(outputFile), result)