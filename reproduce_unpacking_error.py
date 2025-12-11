
import torch
from torch.utils.data import DataLoader
from src.dataset import ScrollDataset
import numpy as np
import tifffile
import os

def reproduction():
    # Setup dummy data
    os.makedirs("test_data_repro", exist_ok=True)
    dummy_vol = np.random.randint(0, 255, (32, 32, 32), dtype=np.uint8)
    dummy_lbl = np.random.randint(0, 2, (32, 32, 32), dtype=np.uint8)
    vol_path = "test_data_repro/vol.tif"
    lbl_path = "test_data_repro/lbl.tif"
    tifffile.imwrite(vol_path, dummy_vol)
    tifffile.imwrite(lbl_path, dummy_lbl)

    # Instantiate dataset with train=False
    ds = ScrollDataset([vol_path], [lbl_path], patch_size=(16, 16, 16), train=False)
    
    # Instantiate DataLoader
    loader = DataLoader(ds, batch_size=1)
    
    print("Attempting to iterate loader...")
    try:
        for images, labels in loader:
            print("Successfully unpacked images and labels")
            break
    except ValueError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")

if __name__ == "__main__":
    reproduction()
