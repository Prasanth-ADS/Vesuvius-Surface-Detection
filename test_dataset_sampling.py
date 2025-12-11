
import numpy as np
import tifffile
import os
import torch
from src.dataset import ScrollDataset

def test_surface_sampling():
    # 1. Create dummy data: 128^3 volume
    # Mask: only a small 10x10x10 block at (50,50,50) is 1. Rest is 0.
    os.makedirs("test_sampling", exist_ok=True)
    
    vol_shape = (128, 128, 128)
    dummy_vol = np.random.randint(0, 255, vol_shape, dtype=np.uint8)
    dummy_lbl = np.zeros(vol_shape, dtype=np.uint8)
    
    # Set a target region
    dummy_lbl[50:60, 50:60, 50:60] = 1
    
    vol_path = "test_sampling/vol.tif"
    lbl_path = "test_sampling/lbl.tif"
    tifffile.imwrite(vol_path, dummy_vol)
    tifffile.imwrite(lbl_path, dummy_lbl)
    
    # 2. Init dataset
    # Patch size 32
    ds = ScrollDataset([vol_path], [lbl_path], patch_size=(32, 32, 32), train=True)
    
    print(f"Valid indices found: {len(ds.valid_indices[0][0])}")
    
    # 3. Sample 20 patches
    hits = 0
    start_time = 0
    
    for i in range(20):
        img, lbl = ds[0]
        # Check if label has any 1s
        if lbl.sum() > 0:
            hits += 1
            
    print(f"Sampled 20 patches. {hits} contained surface pixels.")
    
    if hits == 20:
        print("SUCCESS: All sampled patches contained surface data.")
    else:
        print(f"FAIL: Only {hits}/20 patches contained surface data.")
        
    # Check normalization and dtype
    print(f"Image dtype: {img.dtype}")
    print(f"Image mean: {img.mean():.4f}, std: {img.std():.4f}") # std might not be exactly 1 due to padding/0s if any, but valid region should be normalized
    
    if img.dtype == torch.float32:
        print("SUCCESS: Dtype is float32")
    else:
        print("FAIL: Dtype is NOT float32")

if __name__ == "__main__":
    test_surface_sampling()
