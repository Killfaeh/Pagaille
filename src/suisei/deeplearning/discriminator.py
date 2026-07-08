import torch
import torch.nn as nn

def disc_block(in_channels, out_channels, batchnorm=True):

	layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]

	if batchnorm:
		layers.append(nn.InstanceNorm2d(out_channels))
	
	layers.append(nn.LeakyReLU(0.2))
	block = nn.Sequential(*layers)
	nn.init.normal_(block[0].weight, mean=0.0, std=0.02)
	
	return block


class GANDiscriminator512To1024(nn.Module):

	def __init__(self, n_channels=1):

		super().__init__()

		self.net = nn.Sequential(
									disc_block(n_channels * 2, 64,  batchnorm=False),
									disc_block(64,             128, batchnorm=True),
									disc_block(128,            256, batchnorm=True),
									disc_block(256,            512, batchnorm=True),
									nn.Conv2d(512, 1, 4, stride=1, padding=1)
		)
		
		nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.02)
	
	def forward(self, image, condition):
		x = torch.cat([condition, image], dim=1)
		return self.net(x)