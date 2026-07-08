import torch
import numpy as np

from suisei.deeplearning.dataset import *
from suisei.deeplearning.networks import *
from suisei.deeplearning.checkpoint import *

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"PyTorch   : {torch.__version__}")
print(f"Device    : {device}")

modelNum = 0
modelPath = "/Volumes/AIstuff/Peguy/models/upscaleLineartX2/model_{}.pth".format(modelNum)
#inputFile = "/Users/suisei/Desktop/Spock-line-002.png"
inputFile = "/Users/suisei/Desktop/test-antialiasing-0.png"
outputFile = "/Users/suisei/Desktop/test-upscaleX2-{}.png".format(modelNum)

model = UNet512To1024(n_channels=1).to(device)
load_model(model, modelPath)

tensorImg = load_image(inputFile)

with torch.no_grad():
	output = model(tensorImg)

result = output.squeeze().cpu().numpy()
#result = (result * 255).clip(0, 255).astype(np.uint8)
result = (result + 1.0) * 127.5
result = result.clip(0, 255).astype(np.uint8)

cv2.imwrite(str(outputFile), result)