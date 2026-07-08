import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import time

from suisei.deeplearning.checkpoint import *
from suisei.deeplearning.networks import *

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#### /!\ Plein de trucs à factoriser ! /!\ ####

def trainUNet(dataset, model, modelDir, startEpoch, endEpoch, startBatch, batchSize, learningRate=1e-3):

	dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=0)

	optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
	criterion = nn.MSELoss()
	
	load_checkpoint(modelDir, startEpoch, startBatch, model, optimizer, scheduler)
	
	for epoch in range(startEpoch, endEpoch):
	
		print("Training epoch {}".format(epoch))
	
		startTime = time.time()
	
		model.train()
		epoch_loss = 0.0
		
		index = 0
		
		# Boucle sur tous les batch
		for inputs, targets in dataloader:
		
			if index >= startBatch:
				print("	train batch {}/{}".format(index, len(dataloader)))
				inputs, targets = inputs.to(device), targets.to(device)
			
				preds = model(inputs)
				loss  = criterion(preds, targets)
				
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				
				epoch_loss += loss.item()
				
				# Je fais une petite sauvegarde des checkpoints tous les 20 batch afin de ne pas perdre trop de temps d'entraînement en cas d'interruption
				if index%20 == 0:
					save_checkpoint(modelDir, epoch, index, model, optimizer, scheduler, epoch_loss)
			else:
				print("	skip batch {}/{}/{}".format(index, startBatch, len(dataloader)))
			
			index += 1
		
		epoch_loss /= len(dataloader)
		scheduler.step(epoch_loss)
		
		path = "{}model_{}.pth".format(modelDir, epoch)
		torch.save(model.state_dict(), path)
		clean_epoch(modelDir, epoch)
		save_checkpoint(modelDir, epoch, index, model, optimizer, scheduler, epoch_loss)
		
		endTime = time.time()
		execTime = (endTime-startTime)/60.0

		index = 0
		startBatch = 0
		
		print(f"	Epoch {epoch + 1:04d}/{endEpoch} — Time : {execTime} — Loss : {epoch_loss:.6f} — LR : {scheduler.get_last_lr()[0]:.2e}")
		

def attention_regularization(model, lambda_att=0.01):
	
	reg = torch.tensor(0.0, device=next(model.parameters()).device)
	
	for module in model.modules():
		if isinstance(module, AttentionGate) and module.last_attention is not None:
			reg += module.last_attention.mean()
	
	return reg * lambda_att


def attention_entropy_loss(model, lambda_att=0.01):

	loss = torch.tensor(0.0, device=next(model.parameters()).device)

	for module in model.modules():
		if isinstance(module, AttentionGate) and module.last_attention is not None:
			att = module.last_attention
			entropy = -(att * torch.log(att + 1e-6) + (1 - att) * torch.log(1 - att + 1e-6))
			loss += entropy.mean()
	
	return loss * lambda_att


def trainGlobalLocalUNet(dataset, model, modelDir, startEpoch, endEpoch, startBatch, batchSize, learningRate=1e-4):

	dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=0)

	optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, betas=(0.5, 0.999))
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
	criterion = nn.L1Loss()
	
	load_checkpoint(modelDir, startEpoch, startBatch, model, optimizer, scheduler)
	
	for epoch in range(startEpoch, endEpoch):
	
		print("Training epoch {}".format(epoch))
	
		startTime = time.time()
	
		model.train()
		epoch_loss = 0.0
		
		index = 0
		
		# Boucle sur tous les batch
		for _ in dataloader:
		
			if index >= startBatch:
		
				print("	train batch {}/{}".format(index, len(dataloader)))
		
				fullInputs, inputTiles, targetTiles = dataset.getItem(index, batchSize)
				fullInputs, inputTiles, targetTiles = fullInputs.to(device), inputTiles.to(device), targetTiles.to(device)
			
				preds = model(inputTiles, fullInputs)
				loss  = criterion(preds, targetTiles)
				
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				#print(model.local_branch.output[0].weight.abs().mean())
				#print(model.local_branch.output[0].weight.abs().max())
				
				print("Loss: ", loss.item())
				epoch_loss += loss.item()
				
				# Je fais une petite sauvegarde des checkpoints tous les 20 batch afin de ne pas perdre trop de temps d'entraînement en cas d'interruption
				if index%20 == 0:
					save_checkpoint(modelDir, epoch, index, model, optimizer, scheduler, epoch_loss)
				
				# J'enregistre un modèles tous les 100 batchs pour pouvoir tester et vérifier rapidement si je ne fais pas fausse route
				if index%100 == 0:
					path = "{}model_{}_{}.pth".format(modelDir, epoch, index)
					torch.save(model.state_dict(), path)
					model.eval()
					model.test(epoch, index)
					model.train()
			else:
				print("	skip batch {}/{}/{}".format(index, startBatch, len(dataloader)))
			
			index += 1
		
		epoch_loss /= len(dataloader)
		scheduler.step(epoch_loss)
		
		path = "{}model_{}.pth".format(modelDir, epoch)
		torch.save(model.state_dict(), path)
		clean_epoch(modelDir, epoch)
		save_checkpoint(modelDir, epoch, index, model, optimizer, scheduler, epoch_loss)
		
		endTime = time.time()
		execTime = (endTime-startTime)/60.0

		index = 0
		startBatch = 0
		
		print(f"	Epoch {epoch + 1:04d}/{endEpoch} — Time : {execTime} — Loss : {epoch_loss:.6f} — LR : {scheduler.get_last_lr()[0]:.2e}")
		
		
def trainPix2Pix(dataset, G, D, modelDir, startEpoch, endEpoch, startBatch, batchSize, learningRate=2e-4):
	
	dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=0)
	
	opt_G = torch.optim.Adam(G.parameters(), lr=learningRate, betas=(0.5, 0.999))
	opt_D = torch.optim.Adam(D.parameters(), lr=learningRate, betas=(0.5, 0.999))
	
	criterion_GAN = nn.BCEWithLogitsLoss()
	criterion_L1  = nn.L1Loss()
	
	LAMBDA_L1      = 100 
	
	load_pix2pix_checkpoint(modelDir, startEpoch, startBatch, G, D, opt_G, opt_D)
	
	for epoch in range(startEpoch, endEpoch):
	
		print("Training epoch {}".format(epoch))
	
		startTime = time.time()
	
		G.train()
		D.train()
		epoch_loss_G = 0.0
		epoch_loss_D = 0.0
		
		index = 0
		
		# Boucle sur tous les batch
		for inputs, targets in dataloader:
		
			if index >= startBatch:
				print("	train batch {}/{}".format(index, len(dataloader)))
				inputs, targets = inputs.to(device), targets.to(device)
				
				fake_targets = G(inputs)
				
				inputs_upscaled = torch.nn.functional.interpolate(
					inputs, size=(targets.shape[2], targets.shape[3]), mode='bilinear', align_corners=False
				)
				
				# Entraînement du discriminateur
				opt_D.zero_grad()
				
				# Vrais patches
				pred_real = D(targets, inputs_upscaled)
				labels_real = torch.ones_like(pred_real)
				loss_D_real = criterion_GAN(pred_real, labels_real)
				
				# Faux patches
				pred_fake = D(fake_targets.detach(), inputs_upscaled)
				labels_fake = torch.zeros_like(pred_fake)
				loss_D_fake = criterion_GAN(pred_fake, labels_fake)
				
				loss_D = (loss_D_real + loss_D_fake) * 0.5
				loss_D.backward()
				opt_D.step()
				
				# Entraînement du générateur
				opt_G.zero_grad()
				
				# Le générateur veut tromper le discriminateur
				pred_fake_for_G = D(fake_targets, inputs_upscaled)
				loss_G_adv = criterion_GAN(pred_fake_for_G, torch.ones_like(pred_fake_for_G))
				
				loss_G_L1 = criterion_L1(fake_targets, targets) * LAMBDA_L1
				
				loss_G = loss_G_adv + loss_G_L1
				loss_G.backward()
				opt_G.step()
				
				epoch_loss_G += loss_G.item()
				epoch_loss_D += loss_D.item()
				
				# Je fais une petite sauvegarde des checkpoints tous les 20 batch afin de ne pas perdre trop de temps d'entraînement en cas d'interruption
				if index%20 == 0:
					save_pix2pix_checkpoint(modelDir, epoch, index, G, D, opt_G, opt_D, epoch_loss_G, epoch_loss_D)
				
			else:
				print("	skip batch {}/{}/{}".format(index, startBatch, len(dataloader)))
			
			index += 1
			
		epoch_loss_G /= len(dataloader)
		epoch_loss_D /= len(dataloader)
		
		path = "{}model_{}.pth".format(modelDir, epoch)
		torch.save(G.state_dict(), path)
		clean_epoch(modelDir, epoch)
		save_pix2pix_checkpoint(modelDir, epoch, index, G, D, opt_G, opt_D, epoch_loss_G, epoch_loss_D)
		
		endTime = time.time()
		execTime = (endTime-startTime)/60.0

		index = 0
		startBatch = 0

		print(f"Epoch {epoch + 1:04d}/{endEpoch} — Time : {execTime} — loss_G : {epoch_loss_G:.4f} — loss_D : {epoch_loss_D:.4f}")

