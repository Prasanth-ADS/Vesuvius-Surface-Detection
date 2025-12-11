import torch
from torch.utils.data import Dataset
import tifffile
import numpy as np
import zarr
from pathlib import Path
import random

class ScrollDataset(Dataset):
    def __init__(self, volume_paths, label_paths=None, patch_size=(128, 128, 128), train=True):
        """
        Args:
            volume_paths (list): List of paths to the 3D volume TIFFs or Zarr arrays.
            label_paths (list, optional): List of paths to the corresponding labels (masks/ink).
            patch_size (tuple): Size of the 3D patch (D, H, W).
            train (bool): Whether in training mode (enables augmentation/random sampling).
        """
        self.volume_paths = volume_paths
        self.label_paths = label_paths
        self.patch_size = patch_size
        self.train = train
        
        # Cache for loaded volumes
        self.volumes = []
        self.labels = []
        # Store valid center indices for each volume: list of (z, y, x) tuples
        self.valid_indices = []
        
        print(f"Loading dataset... {len(volume_paths)} volumes.")
        
        pd, ph, pw = self.patch_size
        
        for i, path in enumerate(volume_paths):
            vol = tifffile.imread(path)
            self.volumes.append(vol)
            
            if label_paths:
                lbl = tifffile.imread(label_paths[i])
                self.labels.append(lbl)
                
                # Pre-calculate valid indices where mask exists
                if self.train:
                    print(f"Computing valid surface indices for volume {i}...")
                    # Get indices where label > 0
                    # We want the center of the patch to be such that the patch is valid
                    # Patch range: [z:z+pd, y:y+ph, x:x+pw]
                    # Valid start indices for z: 0 to D-pd, etc.
                    
                    # Optimization: Instead of full dense search, we can just find non-zero pixels
                    # and filter those that can be valid patch centers (or top-left corners)
                    
                    # Find all points where label is present
                    z_idxs, y_idxs, x_idxs = np.where(lbl > 0)
                    
                    # Filter to keep only those that allow a full patch extraction
                    # We want the 'ink/surface' pixel to be SOMEWHERE in the patch.
                    # Best strategy: Pick a surface pixel, then randomly offset the patch top-left 
                    # such that the surface pixel is inside.
                    
                    # Store these surface pixels
                    # Zip them into a list of tuples or keep as arrays
                    # Let's keep as 3 arrays for memory efficiency
                    
                    # Filter out pixels too close to edges if we want strict containment
                    # But actually, we just need the patch D,H,W to fit in volume.
                    # The patch top-left (z,y,x) must be in [0, D-pd] etc.
                    
                    d, h, w = vol.shape
                    
                    # Filter valid surface points (optional, but ensures we don't pick edge noise if any)
                    # For now keep all > 0
                    
                    self.valid_indices.append((z_idxs, y_idxs, x_idxs))
                    print(f"Found {len(z_idxs)} surface voxels for volume {i}")
            else:
                 # Inference mode / no labels - no valid indices pre-calc
                 self.valid_indices.append(None)
                
    def __len__(self):
        return 1000 if self.train else len(self.volume_paths)

    def __getitem__(self, idx):
        if self.train:
            return self.sample_patch()
        else:
            vol_idx = idx % len(self.volumes)
            vol = self.volumes[vol_idx]
            vol = self.normalize(vol)
            
            if self.labels:
                 lbl = self.labels[vol_idx]
                 return (torch.from_numpy(vol).float().unsqueeze(0), torch.from_numpy(lbl).float().unsqueeze(0))
            else:
                 return (torch.from_numpy(vol).float().unsqueeze(0), torch.zeros_like(torch.from_numpy(vol)).float().unsqueeze(0))

    def normalize(self, volume):
        """Standardization with explicit float32 cast."""
        # Ensure float32 first
        volume = volume.astype(np.float32)
        mean = np.mean(volume)
        std = np.std(volume)
        return (volume - mean) / (std + 1e-6)

    def sample_patch(self):
        """
        Samples a 3D patch centered around a valid surface voxel.
        """
        vol_idx = random.randint(0, len(self.volumes) - 1)
        vol = self.volumes[vol_idx]
        lbl = self.labels[vol_idx] if self.labels else None
        
        d, h, w = vol.shape
        pd, ph, pw = self.patch_size
        
        # Surface sampling strategy
        if self.train and self.valid_indices[vol_idx] is not None:
             z_idxs, y_idxs, x_idxs = self.valid_indices[vol_idx]
             if len(z_idxs) > 0:
                 # Pick a random surface voxel
                 rand_point_idx = random.randint(0, len(z_idxs) - 1)
                 pz, py, px = z_idxs[rand_point_idx], y_idxs[rand_point_idx], x_idxs[rand_point_idx]
                 
                 # Now verify we can place a patch containing this point
                 # The patch covers [z, z+pd]. So z must be in [pz - pd + 1, pz]
                 # AND z must be in [0, d - pd]
                 
                 min_z = max(0, pz - pd // 2) # Center roughly
                 min_y = max(0, py - ph // 2)
                 min_x = max(0, px - pw // 2)
                 
                 # Adjust max to ensure we don't go out of bounds
                 z = min(d - pd, min_z)
                 y = min(h - ph, min_y)
                 x = min(w - pw, min_x)
                 
                 # Clamp to 0 again just in case d < pd (datasets should be validated for this)
                 z = max(0, z)
                 y = max(0, y)
                 x = max(0, x)
                 
             else:
                 # Fallback if mask is empty
                 z = random.randint(0, max(0, d - pd))
                 y = random.randint(0, max(0, h - ph))
                 x = random.randint(0, max(0, w - pw))
        else:
            # Random sampling
            z = random.randint(0, max(0, d - pd))
            y = random.randint(0, max(0, h - ph))
            x = random.randint(0, max(0, w - pw))
            
        patch_vol = vol[z:z+pd, y:y+ph, x:x+pw]
        
        if lbl is not None:
            patch_lbl = lbl[z:z+pd, y:y+ph, x:x+pw]
        else:
            patch_lbl = np.zeros_like(patch_vol)

        # Normalize patch
        patch_vol = self.normalize(patch_vol)
        
        # Augmentations
        if self.train:
            k = random.randint(0, 3)
            patch_vol = np.rot90(patch_vol, k=k, axes=(1, 2))
            patch_lbl = np.rot90(patch_lbl, k=k, axes=(1, 2))
            
            if random.random() < 0.5:
                patch_vol = np.flip(patch_vol, axis=1)
                patch_lbl = np.flip(patch_lbl, axis=1)
            if random.random() < 0.5:
                patch_vol = np.flip(patch_vol, axis=2)
                patch_lbl = np.flip(patch_lbl, axis=2)
            if random.random() < 0.5:
                patch_vol = np.flip(patch_vol, axis=0) # Flip D
                patch_lbl = np.flip(patch_lbl, axis=0)
        
        # Convert to tensor and add channel dim
        return (torch.from_numpy(patch_vol.copy()).float().unsqueeze(0), 
                torch.from_numpy(patch_lbl.copy()).float().unsqueeze(0))

if __name__ == "__main__":
    # Test the dataset
    # Create dummy data for testing
    import os
    os.makedirs("test_data", exist_ok=True)
    dummy_vol = np.random.randint(0, 255, (200, 200, 200), dtype=np.uint8)
    dummy_lbl = np.random.randint(0, 2, (200, 200, 200), dtype=np.uint8)
    tifffile.imwrite("test_data/vol.tif", dummy_vol)
    tifffile.imwrite("test_data/lbl.tif", dummy_lbl)
    
    ds = ScrollDataset(["test_data/vol.tif"], ["test_data/lbl.tif"], patch_size=(64, 64, 64))
    img, lbl = ds[0]
    print(f"Patch shape: {img.shape}, Label shape: {lbl.shape}")
    print(f"Mean: {img.mean()}, Std: {img.std()}")
