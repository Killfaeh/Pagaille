import albumentations as A
import cv2
import numpy as np
import os
from os import listdir
from pathlib import Path

targetSize = 1024

datasetRoot = '/Volumes/AIstuff/Peguy/dataset/'
inputPath = '{}upscaleLineartX2/target/'.format(datasetRoot)
outputPath = '{}upscaleLineartX2/input/'.format(datasetRoot)

#### Définition de la transformation ####

transform = A.ReplayCompose([
	A.Resize(height=targetSize/2, width=targetSize/2),
])

#### Exécution de la boucle ####

offset = 0
sourceFilesList = listdir(inputPath)

for filename in sourceFilesList:
		
	if filename != '.DS_Store':
		inputFilePath = '{}{}'.format(inputPath, filename)
		outputFilePath = '{}{}'.format(outputPath, filename)
		
		print('resize {}...'.format(inputPath))
	
		inputImg = cv2.imread(str(inputFilePath))
		result = transform(image=inputImg)
		outputImg = result["image"]
		
		if (outputImg.max() > 128 and outputImg.min() < 128):
			cv2.imwrite(outputFilePath, outputImg)
		else:
			print("	MONOCHROME ! J'enregistre pas")
			