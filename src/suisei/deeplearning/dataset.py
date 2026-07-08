import torch
from torch.utils.data import Dataset
import cv2
import os
from os import listdir

from suisei.imageprocessing.size import *
from suisei.imageprocessing.crop import *

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class TrainDataset(Dataset):

	def __init__(self, inputDir, targetDir, colorMode=cv2.IMREAD_GRAYSCALE):

		self.inputPaths = []
		self.targetPaths = []
		self.colorMode = colorMode
		
		filesList = listdir(inputDir)
		
		for filename in filesList:
		
			if filename != '.DS_Store':
				inputFilePath = '{}{}'.format(inputDir, filename)
				targetFilePath = '{}{}'.format(targetDir, filename)
				self.inputPaths.append(inputFilePath)
				self.targetPaths.append(targetFilePath)
	
	def __len__(self):
		return len(self.inputPaths)
	
	def __getitem__(self, index):
		input = cv2.imread(str(self.inputPaths[index]),  self.colorMode)
		target = cv2.imread(str(self.targetPaths[index]), self.colorMode)
		
		input = torch.from_numpy(input).unsqueeze(0).float() / 127.5 - 1.0
		target = torch.from_numpy(target).unsqueeze(0).float() / 127.5 - 1.0
		
		return input, target

		
class GlobalLocalTrainDataset(Dataset):
		
	def __init__(self, inputDir, targetDir, tile_size=512, n_tiles=20, colorMode=cv2.IMREAD_GRAYSCALE):
		
		self.tile_size = tile_size
		self.n_tiles = n_tiles
		self.inputPaths = []
		self.targetPaths = []
		self.colorMode = colorMode
		
		filesList = listdir(inputDir)
		
		for filename in filesList:
		
			if filename != '.DS_Store':
				inputFilePath = '{}{}'.format(inputDir, filename)
				targetFilePath = '{}{}'.format(targetDir, filename)
				self.inputPaths.append(inputFilePath)
				self.targetPaths.append(targetFilePath)
	
	def __len__(self):
		return len(self.inputPaths) * self.n_tiles
	
	def __getitem__(self, index):
		return index
	
	def getItem(self, index, batchSize):
	
		img_index  = index // self.n_tiles
	
		input = cv2.imread(str(self.inputPaths[img_index]),  self.colorMode)
		target = cv2.imread(str(self.targetPaths[img_index]), self.colorMode)
		
		fullInput = resize(input, self.tile_size, self.tile_size)
		
		fullInput = torch.from_numpy(fullInput).unsqueeze(0).float() / 127.5 - 1.0
		input = torch.from_numpy(input).unsqueeze(0).float() / 127.5 - 1.0
		target = torch.from_numpy(target).unsqueeze(0).float() / 127.5 - 1.0
	
		fullList = []
		inputList = []
		targetList = []
	
		for i in range(batchSize):
	
			tiles = randomTile([input[0], target[0]], self.tile_size)
			
			inputTile = tiles[0].reshape((1, tiles[0].shape[0], tiles[0].shape[1]))
			targetTile = tiles[1].reshape((1, tiles[1].shape[0], tiles[1].shape[1]))
			
			fullList.append(fullInput)
			inputList.append(inputTile)
			targetList.append(targetTile)
			
		fullList = torch.stack(fullList, dim=0)
		inputList= torch.stack(inputList, dim=0)
		targetList = torch.stack(targetList, dim=0)
		
		return fullList, inputList, targetList
		
		
class InferenceDataset(Dataset):
		
	def __init__(self, inputDir, colorMode=cv2.IMREAD_GRAYSCALE):
		
		self.inputPaths = []
		self.colorMode = colorMode
		
		filesList = listdir(inputDir)
		
		for filename in filesList:
		
			if filename != '.DS_Store':
				inputFilePath = '{}{}'.format(inputDir, filename)
				self.inputPaths.append(inputFilePath)
	
	def __len__(self):
		return len(self.inputPaths)
	
	def __getitem__(self, index):
		img = cv2.imread(str(self.inputPaths[index]), self.colorMode)
		tensor = torch.from_numpy(img).unsqueeze(0).float() / 127.5 - 1.0
		return tensor, self.inputPaths[index].name 

def load_image(imagePath, colorMode=cv2.IMREAD_GRAYSCALE):
	img = cv2.imread(str(imagePath), colorMode)
	tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float() / 127.5 - 1.0
	tensor = tensor.to(device)
	return tensor