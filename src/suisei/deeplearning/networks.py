import torch
import torch.nn as nn

from suisei.deeplearning.imagepreprocessing import *
from suisei.deeplearning.checkpoint import *
from suisei.deeplearning.inference import *

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class DownBlock(nn.Module):

	def __init__(self, in_channels, out_channels, batchnorm=True):

		super().__init__()

		layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=not batchnorm)]
		
		if batchnorm:
			layers.append(nn.GroupNorm(32, out_channels))
			#layers.append(nn.BatchNorm2d(out_channels))
		
		layers.append(nn.LeakyReLU(0.2))
		self.block = nn.Sequential(*layers)
		
		nn.init.normal_(self.block[0].weight, mean=0.0, std=0.02)
	
	def forward(self, x):
		return self.block(x)


class UpBlock(nn.Module):

	def __init__(self, in_channels, out_channels, dropout=False):

		super().__init__()

		layers = [
			nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
			nn.GroupNorm(32, out_channels)
			#nn.BatchNorm2d(out_channels),
		]
			
		if dropout:
			layers.append(nn.Dropout(0.5))
		
		layers.append(nn.ReLU())
		self.block = nn.Sequential(*layers)
		
		nn.init.normal_(self.block[0].weight, mean=0.0, std=0.02)
		
	def forward(self, x, skip=None):
		
		x = self.block(x)
		
		if skip is not None:
			x = torch.cat([x, skip], dim=1)
		
		return x


class AttentionGate(nn.Module):

	def __init__(self, F_g, F_l, F_int):

		super().__init__()

		self.W_g   = nn.Conv2d(F_g, F_int, 1, bias=False)
		self.W_x   = nn.Conv2d(F_l, F_int, 1, bias=False)
		self.psi   = nn.Conv2d(F_int, 1, 1, bias=False)
		self.norm  = nn.InstanceNorm2d(F_int)
		self.relu  = nn.ReLU()
		self.sigma = nn.Sigmoid()
		self.last_attention = None
	
	def forward(self, g, x):
	
		g_up = torch.nn.functional.interpolate(
			self.W_g(g), size=x.shape[2:], mode='bilinear', align_corners=False
		)
			
		attention = self.sigma(
			self.psi(self.relu(self.norm(g_up + self.W_x(x))))
		)
			
		self.last_attention = attention
		
		return x * attention


class UNet512x512(nn.Module):

	def __init__(self, n_channels=3):

		super().__init__()
		
		# Encodeur
		self.enc1 = DownBlock(n_channels, 64,  batchnorm=False)
		self.enc2 = DownBlock(64,          128, batchnorm=True)
		self.enc3 = DownBlock(128,         256, batchnorm=True)
		self.enc4 = DownBlock(256,         512, batchnorm=True)
		self.enc5 = DownBlock(512,         512, batchnorm=True)
		self.enc6 = DownBlock(512,         512, batchnorm=True)
		self.enc7 = DownBlock(512,         512, batchnorm=True)
		
		# Bottleneck
		self.bottleneck = nn.Sequential(
			nn.Conv2d(512, 512, 4, stride=2, padding=1, bias=False),
			nn.ReLU()
		)
		
		nn.init.normal_(self.bottleneck[0].weight, mean=0.0, std=0.02)
		
		# Décodeur
		self.dec1 = UpBlock(512,        512, dropout=True)
		self.dec2 = UpBlock(512 + 512,  512, dropout=True)
		self.dec3 = UpBlock(512 + 512,  512, dropout=True)
		self.dec4 = UpBlock(512 + 512,  512, dropout=False)
		self.dec5 = UpBlock(512 + 512,  256, dropout=False)
		self.dec6 = UpBlock(256 + 256,  128, dropout=False)
		self.dec7 = UpBlock(128 + 128,  64,  dropout=False)
		
		# Sortie
		self.output = nn.Sequential(
			nn.ConvTranspose2d(64 + 64, n_channels, 4, stride=2, padding=1),
			nn.Tanh()
		)
		
		nn.init.normal_(self.output[0].weight, mean=0.0, std=0.02)

	
	def forward(self, x):

		e1 = self.enc1(x)
		e2 = self.enc2(e1)
		e3 = self.enc3(e2)
		e4 = self.enc4(e3)
		e5 = self.enc5(e4)
		e6 = self.enc6(e5)
		e7 = self.enc7(e6)
		
		x = self.bottleneck(e7)
		
		x = self.dec1(x, e7)
		x = self.dec2(x, e6)
		x = self.dec3(x, e5)
		x = self.dec4(x, e4)
		x = self.dec5(x, e3)
		x = self.dec6(x, e2)
		x = self.dec7(x, e1)
		
		return self.output(x)


class UNet512To1024(nn.Module):

	def __init__(self, n_channels=1):

		super().__init__()

		# Encodeur
		self.enc1 = DownBlock(n_channels, 64,  batchnorm=False)
		self.enc2 = DownBlock(64,          128, batchnorm=True)
		self.enc3 = DownBlock(128,         256, batchnorm=True)
		self.enc4 = DownBlock(256,         512, batchnorm=True)
		self.enc5 = DownBlock(512,         512, batchnorm=True)
		self.enc6 = DownBlock(512,         512, batchnorm=True)
		self.enc7 = DownBlock(512,         512, batchnorm=True)
		
		# Bottleneck
		self.bottleneck = nn.Sequential(
			nn.Conv2d(512, 512, 4, stride=2, padding=1, bias=False),
			nn.ReLU()
		)
		
		nn.init.normal_(self.bottleneck[0].weight, mean=0.0, std=0.02)
		
		# Décodeur
		self.dec1 = UpBlock(512,       512, dropout=True)
		self.dec2 = UpBlock(512 + 512, 512, dropout=True)
		self.dec3 = UpBlock(512 + 512, 512, dropout=True)
		self.dec4 = UpBlock(512 + 512, 512, dropout=False)
		self.dec5 = UpBlock(512 + 512, 256, dropout=False)
		self.dec6 = UpBlock(256 + 256, 128, dropout=False)
		self.dec7 = UpBlock(128 + 128, 64,  dropout=False)
		self.dec8 = UpBlock(64  + 64,  64,  dropout=False)
		
		self.output = nn.Sequential(
			nn.ConvTranspose2d(64, n_channels, 4, stride=2, padding=1),
			nn.Tanh()
		)
		
		nn.init.normal_(self.output[0].weight, mean=0.0, std=0.02)
			
	def forward(self, x):
		
		e1 = self.enc1(x)
		e2 = self.enc2(e1)
		e3 = self.enc3(e2)
		e4 = self.enc4(e3)
		e5 = self.enc5(e4)
		e6 = self.enc6(e5)
		e7 = self.enc7(e6)
		
		x = self.bottleneck(e7)
		
		x = self.dec1(x,  e7)
		x = self.dec2(x,  e6)
		x = self.dec3(x,  e5)
		x = self.dec4(x,  e4)
		x = self.dec5(x,  e3)
		x = self.dec6(x,  e2)
		x = self.dec7(x,  e1)
		x = self.dec8(x)
		
		return self.output(x)


class GlobalBranch(nn.Module):

	def __init__(self, n_channels=1):

		super().__init__()

		self.enc1 = DownBlock(n_channels, 64,  batchnorm=False)
		self.enc2 = DownBlock(64,          128, batchnorm=True)
		self.enc3 = DownBlock(128,         256, batchnorm=True)
		self.enc4 = DownBlock(256,         512, batchnorm=True)
		self.enc5 = DownBlock(512,         512, batchnorm=True)
		self.enc6 = DownBlock(512,         512, batchnorm=True)
	
	def forward(self, x):
	
		f1 = self.enc1(x)
		f2 = self.enc2(f1)
		f3 = self.enc3(f2)
		f4 = self.enc4(f3)
		f5 = self.enc5(f4)
		f6 = self.enc6(f5)
		
		return f1, f2, f3, f4, f5, f6
		
		
class LocalBranch(nn.Module):
		
	def __init__(self, n_channels=1):
		
		super().__init__()
		
		# Encodeur — identique
		#"""
		self.enc1 = DownBlock(n_channels, 64,  batchnorm=False)
		self.enc2 = DownBlock(64,          128, batchnorm=True)
		self.enc3 = DownBlock(128,         256, batchnorm=True)
		self.enc4 = DownBlock(256,         512, batchnorm=True)
		self.enc5 = DownBlock(512,         512, batchnorm=True)
		self.enc6 = DownBlock(512,         512, batchnorm=True)
		self.enc7 = DownBlock(512,         512, batchnorm=True)
		
		# Bottleneck — identique
		self.bottleneck = nn.Sequential(
			nn.Conv2d(512, 512, 4, stride=2, padding=1, bias=False),
			nn.ReLU()
		)
		
		nn.init.normal_(self.bottleneck[0].weight, mean=0.0, std=0.02)
		
		# Couches de fusion — identiques
		self.fuse6 = nn.Conv2d(512 + 512, 512, 1)
		self.fuse5 = nn.Conv2d(512 + 512, 512, 1)
		self.fuse4 = nn.Conv2d(512 + 512, 512, 1)
		self.fuse3 = nn.Conv2d(256 + 256, 256, 1)
		self.fuse2 = nn.Conv2d(128 + 128, 128, 1)
		self.fuse1 = nn.Conv2d(64  + 64,  64,  1)
		
		"""
		for layer in [ self.fuse6, self.fuse5, self.fuse4, self.fuse3, self.fuse2, self.fuse1, ]:
			nn.init.normal_(layer.weight, mean=0.0, std=0.02)
			
			if layer.bias is not None:
				nn.init.zeros_(layer.bias)
		#"""
		
		# Décodeur — identique jusqu'à dec7
		self.dec1 = UpBlock(512,  512, dropout=True)
		self.dec2 = UpBlock(1024, 512, dropout=True)
		self.dec3 = UpBlock(1024, 512, dropout=True)
		self.dec4 = UpBlock(1024, 512, dropout=False)
		self.dec5 = UpBlock(1024, 256, dropout=False)
		self.dec6 = UpBlock(512,  128, dropout=False)
		self.dec7 = UpBlock(256,  64,  dropout=False)
		
		# Sortie 1024×1024
		self.output = nn.Sequential(
			nn.ConvTranspose2d(128, n_channels, 4, stride=2, padding=1),
			nn.Tanh()
		)
		
		nn.init.normal_(self.output[0].weight, mean=0.0, std=0.02)
		
	def _fuse(self, local_feat, global_feat, fuse_layer):
		
		global_resized = torch.nn.functional.interpolate(
			global_feat, size=local_feat.shape[2:],
			mode='bilinear', align_corners=False
		)
		
		return fuse_layer(torch.cat([local_feat, global_resized], dim=1))

        
	def forward(self, x, global_features):
		
		gf1, gf2, gf3, gf4, gf5, gf6 = global_features
		
		# Encodeur local
		e1 = self.enc1(x)
		e2 = self.enc2(e1)
		e3 = self.enc3(e2)
		e4 = self.enc4(e3)
		e5 = self.enc5(e4)
		e6 = self.enc6(e5)
		e7 = self.enc7(e6)
		
		# Fusion des skip connections avec le contexte global
		e6_fused = self._fuse(e6, gf6, self.fuse6)
		e5_fused = self._fuse(e5, gf5, self.fuse5)
		e4_fused = self._fuse(e4, gf4, self.fuse4)
		e3_fused = self._fuse(e3, gf3, self.fuse3)
		e2_fused = self._fuse(e2, gf2, self.fuse2)
		e1_fused = self._fuse(e1, gf1, self.fuse1)
		
		# Bottleneck
		x = self.bottleneck(e7)
		
		"""
		print("gf6", gf6.abs().max().item())
		print("e6", e6.abs().max().item())
		print("e7", e7.abs().max().item())
		print("e6_fused", e6_fused.abs().max().item())
		print("Bottleneck", x.abs().max().item())
		#"""
		
		# Décodeur
		x = self.dec1(x,  e7)
		#print("dec1", x.abs().max().item())
		x = self.dec2(x,  e6_fused)
		#print("dec2", x.abs().max().item())
		x = self.dec3(x,  e5_fused)
		#print("dec3", x.abs().max().item())
		x = self.dec4(x,  e4_fused)
		#print("dec4", x.abs().max().item())
		x = self.dec5(x,  e3_fused)
		#print("dec5", x.abs().max().item())
		x = self.dec6(x,  e2_fused)
		#print("dec6", x.abs().max().item())
		x = self.dec7(x,  e1_fused)
		#print("dec7", x.abs().max().item())
		
		#return self.output(x)

		conv = self.output[0](x)
		
		#print( conv.min().item(), conv.max().item(), conv.mean().item() )
		out = torch.tanh(conv)
		#print( out.min().item(), out.max().item(), out.mean().item() )
		
		return out
		
		
class InkingModel(nn.Module):
		
	def __init__(self, n_channels=1):
		super().__init__()
		self.global_branch = GlobalBranch(n_channels)
		self.local_branch  = LocalBranch(n_channels)

	def forward(self, tile, full_image):
		global_features = self.global_branch(full_image)
		return self.local_branch(tile, global_features)
	
	def test(self, epoch, batch):
		print("Test epoch {}, batch {}".format(epoch, batch))
		modelPath = "/Volumes/AIstuff/Peguy/models/inking/model_{}_{}.pth".format(epoch, batch)
		inputFile = "/Users/suisei/Workspace/01_DessinIllustration-TravailEnCours/StarTrek/Inktober2023/Inktober2023-001-scan.jpeg"
		outputFile = "/Users/suisei/Desktop/training-test-inking-{}-{}.png".format(epoch, batch)
		npImg = cv2.imread(str(inputFile), cv2.IMREAD_GRAYSCALE)
		npImg = inkingPreprocessing(npImg)
		result = execGlobalLocalUNet(self, npImg, tile_size=512, overlap=64, device=device)
		cv2.imwrite(str(outputFile), result)
