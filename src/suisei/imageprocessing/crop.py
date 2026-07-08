import numpy as np
from numpy import zeros
from numpy import ones

import cv2
import random

#### Coordonnées de zone ####

def getFrameCoordFromIndex(i, j, tileSize):
	x1 = j*tileSize
	y1 = i*tileSize
	x2 = (j+1)*tileSize
	y2 = (i+1)*tileSize
	
	return x1, y1, x2, y2
	
def getFrameCoord(inputImg, x1, y1, x2, y2):
	
	outputX1 = x1
	outputY1 = y1
	outputX2 = x2
	outputY2 = y2
	
	inputImgWidth = inputImg.shape[1]
	inputImgHeight = inputImg.shape[0]
	
	if x2 <= inputImgWidth and y2 <= inputImgHeight:
		outputX1 = x1
		outputY1 = y1
		outputX2 = x2
		outputY2 = y2
	else:
		
		if x2 <= inputImgWidth:
			outputY2 = inputImgHeight
		elif y2 <= inputImgHeight:
			outputX2 = inputImgWidth
		else:
			outputX2 = inputImgWidth
			outputY2 = inputImgHeight
	
	return outputX1, outputY1, outputX2, outputY2

#### Crop ####

def cropImg(inputImg, x1, y1, x2, y2, ifInput=False):
	
	(inputImgHeight, inputImgWidth) = inputImg.shape[:2]
	
	outputImg = np.zeros((y2-y1, x2-x1))
	
	if ifInput == True:
		outputImg = np.ones((y2-y1, x2-x1))*255
	
	if x2 <= inputImgWidth and y2 <= inputImgHeight:
		outputImg = inputImg[y1:y2,x1:x2]
	else:
		
		if x2 <= inputImgWidth:
			outputImg = inputImg[y1:inputImgHeight,x1:x2]
		elif y2 <= inputImgHeight:
			outputImg = inputImg[y1:y2,x1:inputImgWidth]
		else:
			outputImg = inputImg[y1:inputImgHeight,x1:inputImgWidth]
		
		if x2 > inputImgWidth:
			colsToAdd = np.zeros((outputImg.shape[0], x2-inputImgWidth))
			
			if len(outputImg.shape) > 2:
				colsToAdd = np.zeros((outputImg.shape[0], x2-inputImgWidth, outputImg.shape[2]))
			
			if ifInput == True:
				colsToAdd = np.ones((outputImg.shape[0], x2-inputImgWidth))*255
				
				if len(outputImg.shape) > 2:
					colsToAdd = np.ones((outputImg.shape[0], x2-inputImgWidth, outputImg.shape[2]))*255
			
			#print(outputImg.shape)
			#print(colsToAdd.shape)
			outputImg = np.append(outputImg, colsToAdd, axis=1)
			
		if y2 > inputImgHeight:
			rowsToAdd = np.zeros((y2-inputImgHeight, outputImg.shape[1]))
			
			if len(outputImg.shape) > 2:
				rowsToAdd = np.zeros((y2-inputImgHeight, outputImg.shape[1], outputImg.shape[2]))
			
			if ifInput == True:
				rowsToAdd = np.ones((y2-inputImgHeight, outputImg.shape[1]))*255
				
				if len(outputImg.shape) > 2:
					rowsToAdd = np.ones((y2-inputImgHeight, outputImg.shape[1], outputImg.shape[2]))*255
			
			#print(outputImg.shape)
			#print(rowsToAdd.shape)
			outputImg = np.append(outputImg, rowsToAdd, axis=0)
	
	#print(outputImg.shape)
	
	return outputImg

#### Tuiles ####

def randomTile(inputImgList, tileSize):

	if len(inputImgList) > 0:
		
		imgHeight = inputImgList[0].shape[0]
		imgWidth = inputImgList[0].shape[1]
		
		x = random.randint(0, imgWidth - tileSize)
		y = random.randint(0, imgHeight - tileSize)
		
		outputList = []
		
		for inputImg in inputImgList:
			ouput = cropImg(inputImg, x, y, x + tileSize, y + tileSize)
			outputList.append(ouput)
		
		return outputList
	
	return []
	
