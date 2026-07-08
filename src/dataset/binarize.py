import cv2
import os
from os import listdir

datasetRoot = '/Volumes/AIstuff/Peguy/dataset/'
inputPath = '{}antialiasing/input/'.format(datasetRoot)
outputPath = '{}antialiasing/output/'.format(datasetRoot)

sourceFilesList = listdir(inputPath)

for filename in sourceFilesList:
		
	if filename != '.DS_Store':
		inputFilePath = '{}{}'.format(inputPath, filename)
		ouputFilePath = '{}{}'.format(outputPath, filename)
		inputImp = cv2.imread(inputFilePath, cv2.IMREAD_GRAYSCALE)
		_, binarized = cv2.threshold(inputImp, 127, 255, cv2.THRESH_BINARY)
		print(ouputFilePath)
		
		if (binarized.max() > 128 and binarized.min() < 128):
			cv2.imwrite(ouputFilePath, binarized)
		else:
			print("	MONOCHROME ! J'enregistre pas")
