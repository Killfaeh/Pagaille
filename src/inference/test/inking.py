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

def extract_red(image):
	red = image[:, :, 2]
	return red

def normalize(image):
	img = image/255.0
	min = img.min()
	max = img.max()
	amplitude = max-min
	img = (img-min)/amplitude
	img = (img * 255).clip(0, 255).astype(np.uint8)
	return img

def normalize_with_paper(image, paper):

	if paper.shape != image.shape:
		paper = cv2.resize(paper, (image.shape[1], image.shape[0]))
	
	background = cv2.GaussianBlur(image, (201, 201), 0)
	
	tmpPaper = 1.0 - paper/255.0
	tmpBackground = 1.0 - background/255.0
	tmpImage = 1.0 - image/255.0
	tmpImage = tmpImage-tmpBackground*0.5
	tmpImage = (1.0 - tmpImage)
	
	normalized = (tmpImage * 255).clip(0, 255).astype(np.uint8)
	
	return normalized

modelNum = 0
index = 100
modelPath = "/Volumes/AIstuff/Peguy/models/inking/model_{}_{}.pth".format(modelNum, index)
#modelPath = "/Volumes/AIstuff/Peguy/models/inking/model_{}.pth".format(modelNum)
#inputFile = "/Users/suisei/Desktop/Spock-line-002.png"
#inputFile = "/Users/suisei/Workspace/01_DessinIllustration-TravailEnCours/DPSchool/Challenges/Challenge015/03-Illustrations/Lirana-Crayon-scan.jpg"
inputFile = "/Users/suisei/Workspace/01_DessinIllustration-TravailEnCours/StarTrek/Inktober2023/Inktober2023-001-scan.jpeg"
outputFile = "/Users/suisei/Desktop/test-inking-{}.png".format(modelNum)

paperPath = '/Volumes/AIstuff/Peguy/dataset/inking/paper001.jpg'
paper = cv2.imread(paperPath, cv2.IMREAD_GRAYSCALE)

model = InkingModel(n_channels=1).to(device)
load_model(model, modelPath)

npImg = cv2.imread(str(inputFile), cv2.IMREAD_GRAYSCALE)
npImg = inkingPreprocessing(npImg)

cv2.imwrite("/Users/suisei/Desktop/processedInput.png", npImg)

result = execGlobalLocalUNet(model, npImg, tile_size=1024, overlap=64, device=device)

cv2.imwrite(str(outputFile), result)