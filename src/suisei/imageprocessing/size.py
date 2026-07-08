import numpy as np
from numpy import zeros
from numpy import ones

import cv2

#### Fonctions de redimensionnement du canvas ####

def enlargeCanvas(inputImg, width, height):

	# On crée un canevas vide en niveau de gris
	output = np.zeros((height, width)) # Vérifier l'ordre des dimensions
	
	# Cas où il y a de la couleur
	if len(inputImg.shape) == 3:
		output = np.zeros((height, width, inputImg.shape[2]))
	
	# Calcul de la position de l'image originale
	start0 = 	int((height-inputImg.shape[0])/2)
	end0 = 		int((height-inputImg.shape[0])/2 + inputImg.shape[0])
	start1 = 	int((width-inputImg.shape[1])/2)
	end1 = 		int((width-inputImg.shape[1])/2 + inputImg.shape[1])
	
	# Insertion de l'image originale en la centrant
	output[start0:end0, start1:end1] = inputImg
	return output

def cropCanvas(inputImg, width, height):

	# Calcul de la position dans la source
	start0 = 	int((inputImg.shape[0]-height)/2)
	end0 = 		int((inputImg.shape[0]-height)/2 + inputImg.shape[0]/2)
	start1 = 	int((inputImg.shape[1]-width)/2)
	end1 = 		int((inputImg.shape[1]-width)/2 + inputImg.shape[1]/2)
	
	# Recadrage centré
	output = inputImg[start0:end0, start1:end1]
	return output

def doubleCanvas(inputImg):
	return enlargeCanvas(inputImg, 2*inputImg.shape[1], 2*inputImg.shape[0])

def halfCanvas(inputImg):
	return cropCanvas(inputImg, inputImg.shape[1]/2, inputImg.shape[0]/2)

#### Fonction de redimensionnement ####

# Surcouche d'OpenCV
def resize(inputImg, width, height, interpolation=cv2.INTER_AREA):
	output = cv2.resize(inputImg.astype(np.float32), (height, width), interpolation = interpolation)
	return output

# Redimensionnement par matrice de perspective
def resizeImg(inputImg, scale):
	
	initWidth = inputImg.shape[1]
	initHeight = inputImg.shape[0]
	scaledWidth = initWidth*scale
	scaledHeight = initHeight*scale
	
	xA = (initWidth-scaledWidth)/2.0
	xB = initWidth + (scaledWidth-initWidth)/2.0
	yA = (initHeight-scaledHeight)/2.0
	yB = initHeight + (scaledHeight-initHeight)/2.0
	
	points_A = np.float32([[0,0], [initWidth, 0], [0, initHeight], [initWidth, initHeight]])
	
	points_B = np.float32([[xA, yA], [xB, yA], [xA, yB], [xB, yB]])
	
	M = cv2.getPerspectiveTransform(points_A, points_B)
	output = cv2.warpPerspective(inputImg, M, (initWidth, initHeight))
	
	return output

# Contrainte par la longueur min
def resizeMin(inputImg, size):
	
	imgHeight = inputImg.shape[0]
	imgWidth = inputImg.shape[1]
	
	ratio = imgWidth/imgHeight
	
	targetWidth = size
	targetHeight = size
	
	if imgHeight < imgWidth:
		targetWidth = int(ratio*targetHeight)
	elif imgHeight > imgWidth:
		targetHeight = int(targetWidth/ratio)
	
	output = resize(inputImg, targetWidth, targetHeight, interpolation = cv2.INTER_AREA)
	return output

# Contrainte par la longueur max
def resizeMax(inputImg, size):
	
	imgHeight = inputImg.shape[0]
	imgWidth = inputImg.shape[1]
	
	ratio = imgWidth/imgHeight
	
	targetWidth = size
	targetHeight = size
	
	if imgHeight > imgWidth:
		targetWidth = int(ratio*targetHeight)
	elif imgHeight < imgWidth:
		targetHeight = int(targetWidth/ratio)
	
	output = resize(inputImg, targetWidth, targetHeight, interpolation = cv2.INTER_AREA)
	return output

# Contrainte par la largeur
def resizeWidth(inputImg, size):
	
	imgHeight = inputImg.shape[0]
	imgWidth = inputImg.shape[1]
	
	ratio = imgWidth/imgHeight
	
	targetWidth = size
	targetHeight = int(targetWidth/ratio)
	
	output = resize(inputImg, targetWidth, targetHeight, interpolation = cv2.INTER_AREA)
	return output

# Contrainte par la hauteur
def resizeHeight(inputImg, size):
	
	imgHeight = inputImg.shape[0]
	imgWidth = inputImg.shape[1]
	
	ratio = imgWidth/imgHeight
	
	targetHeight = size
	targetWidth = int(ratio*targetHeight)
	
	output = resize(inputImg, targetWidth, targetHeight, interpolation = cv2.INTER_AREA)
	return output