import torch
import numpy as np
import cv2

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def get_tile_positions(H, W, tile_size, step):
	ys = list(range(0, H - tile_size, step)) + [H - tile_size]
	xs = list(range(0, W - tile_size, step)) + [W - tile_size]
	return ys, xs

def execUNet(model, inputImg):
	pass

def execGlobalLocalUNet(model, input_np, tile_size=512, overlap=64, device='cpu'):

	model.eval()

	# Contexte global — toujours le même pour toutes les tuiles
	full_resized   = cv2.resize(input_np, (2*tile_size, 2*tile_size))
	full_tensor    = torch.from_numpy(full_resized).unsqueeze(0).unsqueeze(0).float() / 127.5 - 1.0
	full_tensor    = full_tensor.to(device)
	global_features = model.global_branch(full_tensor)
	
	H, W     = input_np.shape
	step     = tile_size - overlap
	output   = np.zeros((H, W), dtype=np.float32)
	weights  = np.zeros((H, W), dtype=np.float32)
	
	# Masque gaussien pour le fondu aux jointures
	mask = np.ones((tile_size, tile_size), dtype=np.float32)
	mask = cv2.GaussianBlur(mask, (overlap * 2 + 1, overlap * 2 + 1), overlap / 2)
	
	#print(full_tensor.min(), full_tensor.max(), full_tensor.mean())
	#print(full_tensor)
	
	with torch.no_grad():
		ys, xs = get_tile_positions(H, W, tile_size, step)
		
		for y in ys:
			for x in xs:
				print("		Compute tile {}, {}...".format(x, y))
			
				tile    = input_np[y:y+tile_size, x:x+tile_size]
			
				tensorTile  = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).float() / 127.5 - 1.0
				tensorTile  = tensorTile.to(device)
				
				#print(tensorTile.min(), tensorTile.max(), tensorTile.mean())
				#print(full_tensor.min(), full_tensor.max(), full_tensor.mean())
				
				pred    = model.local_branch(tensorTile, global_features)
				#print(pred.min(), pred.max(), pred.mean())
				result  = pred.squeeze().cpu().numpy()
				
				#print(result.min(), result.max(), result.mean())
				
				output [y:y+tile_size, x:x+tile_size] += result * mask
				weights[y:y+tile_size, x:x+tile_size] += mask

	weights = np.where(weights == 0, 1, weights)
	output = (output/weights + 1.0) * 127.5
	#output = (output + 1.0) * 127.5
	output = output.clip(0, 255).astype(np.uint8)
	
	#print(output.min(), output.max(), output.mean())
	#print(output)
	
	return output