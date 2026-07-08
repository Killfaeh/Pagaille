import albumentations as A
import cv2
import numpy as np
import os
from os import listdir
from pathlib import Path
import shutil
import random

datasetRoot = '/Volumes/AIstuff/Peguy/dataset/'

realPencilPath = '{}inking/real/pencilExtended/'.format(datasetRoot)
realInkPath = '{}inking/real/inkExtended/'.format(datasetRoot)

composePencilPath = '{}inking/compose/pencil/'.format(datasetRoot)
composeInkPath = '{}inking/compose/ink/'.format(datasetRoot)

allPencilPath = '{}inking/all/pencil/'.format(datasetRoot)
allInkPath = '{}inking/all/ink/'.format(datasetRoot)

realList = listdir(realPencilPath)
random.shuffle(realList)
realPencilList = [ "{}{}".format(realPencilPath, filename) for filename in realList ]
realInkList = [ "{}{}".format(realInkPath, filename) for filename in realList ]

composeList = listdir(composePencilPath)
random.shuffle(composeList)
composePencilList = [ "{}{}".format(composePencilPath, filename) for filename in composeList ]
composeInkList = [ "{}{}".format(composeInkPath, filename) for filename in composeList ]

pencilList = realPencilList + composePencilList
realList = realInkList + composeInkList

for index in range(len(pencilList)):
	print("Copy file {}".format(index))
	shutil.copyfile(pencilList[index], "{}img{}.png".format(allPencilPath, index))
	shutil.copyfile(realList[index], "{}img{}.png".format(allInkPath, index))