import albumentations as A
import cv2
import numpy as np
import os
from os import listdir
from pathlib import Path

tileSize = 512

datasetRoot = '/Volumes/AIstuff/Peguy/dataset/'

inputPencilPath = '{}raw/lineart/real/pencil/'.format(datasetRoot)
inputInkPath = '{}raw/lineart/real/ink/'.format(datasetRoot)

outputPencilPath = '{}tiles/paired/{}/pencil/'.format(datasetRoot, tileSize)
outputInkPath = '{}tiles/paired/{}/ink/'.format(datasetRoot, tileSize)

#### Définition de la transformation ####

transform = A.ReplayCompose([
	A.RandomCrop(width=tileSize, height=tileSize),
])

def extractTiles(pencilImagePath, inkImagePath, pencilDir, inkDir, offset=0, n_tiles=50):
	print(pencilImagePath)
	pencilImage = cv2.imread(str(pencilImagePath))
	inkImage = cv2.imread(str(inkImagePath))

	for i in range(n_tiles):
		print('	crop {}...'.format(i))
	
		pencilResult = transform(image=pencilImage)
		pencilTile = pencilResult["image"]
		replay   = pencilResult["replay"]
		
		inkResult = A.ReplayCompose.replay(replay, image=inkImage)
		inkTile   = inkResult["image"]
		
		pencilPath = '{}tile{}.png'.format(pencilDir, offset)
		inkPath = '{}tile{}.png'.format(inkDir, offset)
		print('	output file : {}'.format(pencilPath))
		
		cv2.imwrite(pencilPath, pencilTile)
		cv2.imwrite(inkPath, inkTile)
		offset = offset + 1
	
	return offset

#### Exécution de la boucle ####

offset = 0
sourceFilesList = listdir(inputPencilPath)

for pencilFilename in sourceFilesList:
		
	if pencilFilename != '.DS_Store':
		inkFilename = pencilFilename.replace('pencil', 'ink')
		pencilPath = '{}{}'.format(inputPencilPath, pencilFilename)
		inkPath = '{}{}'.format(inputInkPath, inkFilename)
		offset = extractTiles(pencilPath, inkPath, outputPencilPath, outputInkPath, offset, 50)