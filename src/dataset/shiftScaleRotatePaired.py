import albumentations as A
import cv2
import numpy as np
import os
from os import listdir
from pathlib import Path

from suisei.deeplearning.imagepreprocessing import *

datasetRoot = '/Volumes/AIstuff/Peguy/dataset/'

"""
inputPencilPath = '{}inking/noRotate/pencilProcessed/'.format(datasetRoot)
inputInkPath = '{}inking/noRotate/ink/'.format(datasetRoot)

outputPencilPath = '{}inking/noRotate/pencilExtended/'.format(datasetRoot)
outputInkPath = '{}inking/noRotate/inkExtended/'.format(datasetRoot)
#"""

"""
inputPencilPath = '{}inking/rotate/pencilProcessed/'.format(datasetRoot)
inputInkPath = '{}inking/rotate/ink/'.format(datasetRoot)

outputPencilPath = '{}inking/rotate/pencilExtended/'.format(datasetRoot)
outputInkPath = '{}inking/rotate/inkExtended/'.format(datasetRoot)
#"""

#"""
inputPencilPath = '{}inking/real/pencil/'.format(datasetRoot)
inputInkPath = '{}inking/real/ink/'.format(datasetRoot)

outputPencilPath = '{}inking/real/pencilExtended/'.format(datasetRoot)
outputInkPath = '{}inking/real/inkExtended/'.format(datasetRoot)
#"""

paperPath = '/Volumes/AIstuff/Peguy/dataset/inking/paper001.jpg'
paper = cv2.imread(paperPath, cv2.IMREAD_GRAYSCALE)

#### Définition de la transformation ####

transform = A.ReplayCompose([
	A.ShiftScaleRotate(
						shift_limit=0.2,      				# translation : ±x% de la dimension de l'image
						scale_limit=0.2,      				# échelle : ±x%
						rotate_limit=15,      				# rotation : ±x degrés
						border_mode=cv2.BORDER_REFLECT,  	# remplissage des bords par réflexion
						p=1.0 								# Probabilité que la transformation se déclenche (0.0 : jamais, 1.0 : toujours)
	),
])

def extendFiles(pencilImagePath, inkImagePath, pencilDir, inkDir, offset=0, n_tiles=50):
	print(pencilImagePath)
	pencilImage = cv2.imread(str(pencilImagePath))
	inkImage = cv2.imread(str(inkImagePath))
	
	pencilImage = extractRed(pencilImage)
	pencilImage = inkingPreprocessing(pencilImage)
	inkImage = extractRed(inkImage)

	for i in range(n_tiles):
		print('	crop {}...'.format(i))
	
		pencilResult = transform(image=pencilImage)
		pencilTile = pencilResult["image"]
		replay   = pencilResult["replay"]
		
		inkResult = A.ReplayCompose.replay(replay, image=inkImage)
		inkTile   = inkResult["image"]
		
		pencilPath = '{}img{}.png'.format(pencilDir, offset)
		inkPath = '{}img{}.png'.format(inkDir, offset)
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
		offset = extendFiles(pencilPath, inkPath, outputPencilPath, outputInkPath, offset, 50)