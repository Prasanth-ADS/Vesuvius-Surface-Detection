
import torch
from src.dataset import ScrollDataset
import numpy as np
import tifffile
import os

def check_dtype():
    # Setup dummy data
    os.makedirs("test_data_dtype", exist_ok=True)
    dummy_vol = np.random.randint(0, 255, (32, 32, 32), dtype=np.uint8)
    dummy_lbl = np.random.randint(0, 2, (32, 32, 32), dtype=np.uint8)
    vol_path = "test_data_dtype/vol.tif"
    lbl_path = "test_data_dtype/lbl.tif"
    tifffile.imwrite(vol_path, dummy_vol)
    tifffile.imwrite(lbl_path, dummy_lbl)

    # Instantiate dataset with train=False
    ds = ScrollDataset([vol_path], [lbl_path], patch_size=(16, 16, 16), train=False)
    
    img, lbl = ds[0]
    print(f"Image dtype: {img.dtype}")
    print(f"Label dtype: {lbl.dtype}")
    
    if img.dtype == torch.float64:
        print("FAIL: Image is float64 (double), expected float32")
    elif img.dtype == torch.float32:
        print("SUCCESS: Image is float32")
    else:
        print(f"Unknown dtype: {img.dtype}")

if __name__ == "__main__":
    check_dtype()
