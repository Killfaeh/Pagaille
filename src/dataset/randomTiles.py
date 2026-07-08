import albumentations as A
import cv2
import numpy as np
import os
from os import listdir
from pathlib import Path

tileSize = 1024

datasetRoot = '/Volumes/AIstuff/Peguy/dataset/'

#inputPath = '{}raw/illuPerso/'.format(datasetRoot)
inputPath = '{}raw/lineart/real/ink/'.format(datasetRoot) # Pour faire mes manip' sur lineart

#outputPath = '{}tiles/{}/'.format(datasetRoot, tileSize)
outputPath = '{}upscaleLineartX2/target/'.format(datasetRoot)

#### Définition de la transformation ####

transform = A.ReplayCompose([
	A.RandomCrop(width=tileSize, height=tileSize),
])

def extractTiles(imagePath, outputDir, offset=0, n_tiles=50):
	print(imagePath)
	inputImg = cv2.imread(str(imagePath))

	for i in range(n_tiles):
		print('	crop {}...'.format(i))
	
		result = transform(image=inputImg)
		tile = result["image"]
		
		ouputPath = '{}tile{}.png'.format(outputDir, offset)
		print('	output file : {}'.format(ouputPath))
		
		cv2.imwrite(ouputPath, tile)
		offset = offset + 1
	
	return offset

#### Exécution de la boucle ####

offset = 0
sourceFilesList = listdir(inputPath)

for filename in sourceFilesList:
		
	if filename != '.DS_Store':
		filePath = '{}{}'.format(inputPath, filename)
		offset = extractTiles(filePath, outputPath, offset, 500)