import albumentations as A
import cv2
import numpy as np
import os
from os import listdir
from pathlib import Path
import random

datasetRoot = '/Volumes/AIstuff/Peguy/dataset/'

input1PencilPath = '{}inking/noRotate/pencilExtended/'.format(datasetRoot)
input1InkPath = '{}inking/noRotate/inkExtended/'.format(datasetRoot)

input2PencilPath = '{}inking/rotate/pencilExtended/'.format(datasetRoot)
input2InkPath = '{}inking/rotate/inkExtended/'.format(datasetRoot)

outputPencilPath = '{}inking/compose/pencil/'.format(datasetRoot)
outputInkPath = '{}inking/compose/ink/'.format(datasetRoot)

input1List = listdir(input1PencilPath)
random.shuffle(input1List)
input1PencilList = [ "{}{}".format(input1PencilPath, filename) for filename in input1List ]
input1InkList = [ "{}{}".format(input1InkPath, filename) for filename in input1List ]

input2List = listdir(input2PencilPath)
random.shuffle(input2List)
input2PencilList = [ "{}{}".format(input2PencilPath, filename) for filename in input2List ]
input2InkList = [ "{}{}".format(input2InkPath, filename) for filename in input2List ]

inputPencilList = input1PencilList + input2PencilList
inputInkList = input1InkList + input2InkList

indiciesList = range(len(inputPencilList))

print(indiciesList)

CANVAS_SIZE   = 2048
extendedCanvasSize = CANVAS_SIZE*2
N_CANVAS      = 1300
N_SPRITES_MIN = 5
N_SPRITES_MAX = 10
N_SIDE = 3
STEP = int(CANVAS_SIZE/N_SIDE)
VARIATION = 0.25


def removeBackground(image):

	background = cv2.GaussianBlur(image, (201, 201), 0)
	
	tmpBackground = 1.0 - background/255.0
	tmpImage = 1.0 - image/255.0
	tmpImage = tmpImage-tmpBackground*0.5
	tmpImage = (1.0 - tmpImage)
	
	normalized = (tmpImage * 255).clip(0, 255).astype(np.uint8)
	
	return normalized

for i in range(N_CANVAS):

	print("Creating canvas {}...".format(i))

	canvasPencil = np.full((extendedCanvasSize, extendedCanvasSize), 255, dtype=np.uint8)
	canvasInk = np.full((extendedCanvasSize, extendedCanvasSize), 255, dtype=np.uint8)

	n_sprites = random.randint(N_SPRITES_MIN, N_SPRITES_MAX)
	selection = random.choices(indiciesList, k=N_SIDE*N_SIDE)
	selectIndex = 0
	
	for j in range(N_SIDE):
		for k in range(N_SIDE):
			pathIndex = selection[selectIndex]
			pencilPath = inputPencilList[pathIndex]
			inkPath = inputInkList[pathIndex]
			
			print(pencilPath, inkPath)
			
			pencilSprite = cv2.imread(str(pencilPath), cv2.IMREAD_GRAYSCALE)
			inkSprite = cv2.imread(str(inkPath), cv2.IMREAD_GRAYSCALE)
			
			sHeight, sWidth = pencilSprite.shape
			
			x = int(CANVAS_SIZE/2 + k*STEP + VARIATION*STEP*(random.random()-0.5) - sHeight*0.0)
			y = int(CANVAS_SIZE/2 + j*STEP + VARIATION*STEP*(random.random()-0.5) - sWidth*0.0)
			
			canvasPencil[y:y+sHeight, x:x+sWidth] = np.minimum(canvasPencil[y:y+sHeight, x:x+sWidth], pencilSprite)
			canvasInk[y:y+sHeight, x:x+sWidth] = np.minimum(canvasInk[y:y+sHeight, x:x+sWidth], inkSprite)
			
			selectIndex = selectIndex + 1
	
	canvasPencil = canvasPencil[int(CANVAS_SIZE/2):int(CANVAS_SIZE/2) + CANVAS_SIZE, int(CANVAS_SIZE/2):int(CANVAS_SIZE/2) + CANVAS_SIZE]
	#canvasPencil = removeBackground(canvasPencil)
	canvasInk = canvasInk[int(CANVAS_SIZE/2):int(CANVAS_SIZE/2) + CANVAS_SIZE, int(CANVAS_SIZE/2):int(CANVAS_SIZE/2) + CANVAS_SIZE]
	
	cv2.imwrite("{}img{}.png".format(outputPencilPath, i), canvasPencil)
	cv2.imwrite("{}img{}.png".format(outputInkPath, i), canvasInk)