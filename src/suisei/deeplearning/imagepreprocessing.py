import numpy as np
import cv2

import time

def extractRed(image):
	red = image[:, :, 2]
	return red
	
def removeBackground(image):

	background = cv2.GaussianBlur(image, (201, 201), 0)

	tmpBackground = 1.0 - background/255.0
	tmpImage = 1.0 - image/255.0
	tmpImage = tmpImage-tmpBackground
	tmpImage = (1.0 - tmpImage)
	
	normalized = (tmpImage * 255).clip(0, 255).astype(np.uint8)
	
	return normalized


def adjustLevels(image, black_point=0.50, white_point=0.85, gamma=1.0):

	img = image.astype(np.float32) / 255.0
	img = (img - black_point) / (white_point - black_point)
	img = np.clip(img, 0.0, 1.0)
	
	if gamma != 1.0:
		img = np.power(img, 1.0 / gamma)
	
	return (img * 255).astype(np.uint8)


def inkingPreprocessing(inputImg):

	outputImg = adjustLevels(inputImg, 0.55, 0.85)

	blurred = outputImg.copy()

	for _ in range(15):
		blurred = cv2.GaussianBlur(blurred, (5, 5), 0)

	outputImg = blurred
	
	kernel = np.array([
		[-1, -1, -1],
		[-1,  9, -1],
		[-1, -1, -1]
	])
	
	outputImg = cv2.filter2D(outputImg, -1, kernel)
	outputImg = cv2.normalize(outputImg, None, 0, 255, cv2.NORM_MINMAX)
	
	startTime = time.time()
	BFtolerance = 40
	BFSize = 100
	
	# Le flou bilatéral est facilement long ! 
	print("	Bilateral filter {}, {}...".format(BFSize, BFtolerance))
	outputImg = cv2.bilateralFilter(outputImg, BFSize, BFtolerance, 75)
	endTime = time.time()
	execTime = (endTime-startTime)/60.0
	print("	Bilateral filter {}, {} execution : {}".format(BFSize, BFtolerance, execTime))
	
	return outputImg