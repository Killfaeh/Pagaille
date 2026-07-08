import torch
import os
from os import listdir

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def save_checkpoint(dirPath, epoch, index, model, optimizer, scheduler, loss):
	
	path = "{}checkpoint_{}_{}.pth".format(dirPath, epoch, index)
		
	torch.save({
					"epoch":                epoch,
					"model_state_dict":     model.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"scheduler_state_dict": scheduler.state_dict(),
					"loss":                 loss,
				}, path)
	
	print(f"  → Checkpoint sauvegardé : {path}")

        
def load_checkpoint(dirPath, epoch, index, model, optimizer, scheduler):
	
	path = "{}checkpoint_{}_{}.pth".format(dirPath, epoch, index)
	
	if os.path.isfile(path):
		print("Load checkpoint {}...".format(path))
		ckpt = torch.load(path, map_location=device)
		#model.state_dict(ckpt["model_state_dict"])
		model.load_state_dict(ckpt["model_state_dict"])
		optimizer.load_state_dict(ckpt["optimizer_state_dict"])
		scheduler.load_state_dict(ckpt["scheduler_state_dict"])


def save_pix2pix_checkpoint(dirPath, epoch, index, G, D, opt_G, opt_D, loss_G, loss_D):

	path = "{}checkpoint_{}_{}.pth".format(dirPath, epoch, index)

	torch.save({
					"epoch":        epoch,
					"G_state":      G.state_dict(),
					"D_state":      D.state_dict(),
					"opt_G_state":  opt_G.state_dict(),
					"opt_D_state":  opt_D.state_dict(),
					"loss_G":       loss_G,
					"loss_D":       loss_D,
				}, path)
					
	print(f"  → Checkpoint sauvegardé : epoch {epoch}")


def load_pix2pix_checkpoint(dirPath, epoch, index, G, D, opt_G, opt_D):

	path = "{}checkpoint_{}_{}.pth".format(dirPath, epoch, index)

	if os.path.isfile(path):
		print("Load checkpoint {}...".format(path))
		ckpt = torch.load(path, map_location=device)
		G.load_state_dict(ckpt["G_state"])
		D.load_state_dict(ckpt["D_state"])
		opt_G.load_state_dict(ckpt["opt_G_state"])
		opt_D.load_state_dict(ckpt["opt_D_state"])

def clean_epoch(dirPath, epoch):

	filesList = listdir(dirPath)

	for filename in filesList:
		if "checkpoint_{}_".format(epoch) in filename or "model_{}_".format(epoch) in filename:
			path = "{}{}".format(dirPath, filename)
			print("Delete {}".format(path))
			
			try:
				os.remove(path)
			except FileNotFoundError:
				print("	Le fichier n'existe pas.")
			except PermissionError:
				print("	Permission refusée.")


def load_model(model, path):
	model.load_state_dict(torch.load(path, map_location=device))
	model.eval()