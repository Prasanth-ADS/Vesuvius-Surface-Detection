import torch
import numpy as np
from tqdm import tqdm
import math
import glob
import os
import tifffile
import zipfile
from model import UNet3D
from postprocess import topology_postprocess

def predict_sliding_window(volume, model, patch_size=(128, 128, 128), overlap=0.5, device='cuda', batch_size=1):
    """
    Performs sliding window inference on a 3D volume.
    
    Args:
        volume (np.ndarray): Input 3D volume (D, H, W).
        model (torch.nn.Module): Trained model.
        patch_size (tuple): (D, H, W) of the patch.
        overlap (float): Overlap fraction [0, 1).
        device (str): Device to run inference on.
        batch_size (int): Batch size for inference (currently 1 supported for simplicity).
        
    Returns:
        np.ndarray: Probability map of the same shape as input volume.
    """
    model.eval()
    d, h, w = volume.shape
    pd, ph, pw = patch_size
    
    # Calculate stride
    stride_d = int(pd * (1 - overlap))
    stride_h = int(ph * (1 - overlap))
    stride_w = int(pw * (1 - overlap))
    
    # Padding to ensure we cover the whole volume
    pad_d = math.ceil(d / stride_d) * stride_d + pd - d
    pad_h = math.ceil(h / stride_h) * stride_h + ph - h
    pad_w = math.ceil(w / stride_w) * stride_w + pw - w
    
    # We might need less padding if we just stop at the end, but let's pad to be safe
    # Actually, a simpler strategy is to pad such that the last patch ends at or beyond the volume
    # For simplicity, let's pad with reflection or zeros
    padded_vol = np.pad(volume, ((0, pd), (0, ph), (0, pw)), mode='reflect') # simplified padding
    
    # Output accumulators
    prob_map = np.zeros_like(padded_vol, dtype=np.float32)
    count_map = np.zeros_like(padded_vol, dtype=np.float32)
    
    # Generate patch coordinates
    z_steps = range(0, d, stride_d)
    y_steps = range(0, h, stride_h)
    x_steps = range(0, w, stride_w)
    
    # Create a list of coordinates to iterate
    coords = []
    for z in z_steps:
        for y in y_steps:
            for x in x_steps:
                coords.append((z, y, x))
                
    # Inference loop
    with torch.no_grad():
        for z, y, x in tqdm(coords, desc="Inference"):
            # Extract patch
            patch = padded_vol[z:z+pd, y:y+ph, x:x+pw]
            
            # Normalize (using same stats as training, here assuming local z-score for simplicity)
            # Ideally should use global stats or the dataset's normalize method
            mean = patch.mean()
            std = patch.std()
            patch_norm = (patch - mean) / (std + 1e-6)
            
            # To tensor
            tensor = torch.from_numpy(patch_norm).float().unsqueeze(0).unsqueeze(0).to(device) # (1, 1, D, H, W)
            
            # Predict
            logits = model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze() # (D, H, W)
            
            # Accumulate
            prob_map[z:z+pd, y:y+ph, x:x+pw] += probs
            count_map[z:z+pd, y:y+ph, x:x+pw] += 1.0
            
    # Average
    avg_map = prob_map / (count_map + 1e-6)
    
    # Crop back to original size
    return avg_map[:d, :h, :w]

if __name__ == "__main__":
    # Test sliding window
    from model import UNet3D
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet3D(in_ch=1, out_ch=1, base_ch=16).to(device)
    dummy_vol = np.random.randn(100, 100, 100).astype(np.float32)
    
    print("Running sliding window inference...")
    output = predict_sliding_window(dummy_vol, model, patch_size=(64, 64, 64), overlap=0.25, device=device)
    
    print("Inference test passed!")

def run_inference(config):
    # Setup paths
    test_images = sorted(glob.glob(os.path.join(config.DATA_DIR, 'test_images', '*.tif')))
    
    if not test_images:
        print("Warning: No test images found. Generating dummy test volume for verification...")
        dummy_path = os.path.join(config.DATA_DIR, 'test_images', 'dummy_test.tif')
        # Create a simple dummy volume (random content)
        dummy_vol = np.random.randint(0, 255, (config.PATCH_SIZE[0], config.PATCH_SIZE[1], config.PATCH_SIZE[2]), dtype=np.uint8)
        tifffile.imwrite(dummy_path, dummy_vol)
        test_images = [dummy_path]
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    
    # Load Model(s) for Ensembling
    # We can look for multiple checkpoints or just use the best one
    # For this implementation, let's assume we might have 'best_model.pth' and maybe others if we trained folds
    # We'll just load the single best model for now, but structure it for ensembling
    
    checkpoint_paths = []
    
    # 1. Look for explicit best_model.pth
    best_model_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint_paths.append(best_model_path)
    
    # 2. Look for numbered epochs if no best model
    if not checkpoint_paths:
        epoch_models = glob.glob(os.path.join(config.CHECKPOINT_DIR, 'best_model_epoch*.pth'))
        if epoch_models:
            # Sort by primitive epoch number extraction to find the "latest"
            try:
                epoch_models.sort(key=lambda x: int(os.path.basename(x).split('epoch')[1].split('.pth')[0]))
            except Exception:
                epoch_models.sort() # Fallback
            
            print(f"Found {len(epoch_models)} checkpoint(s). Using latest: {os.path.basename(epoch_models[-1])}")
            checkpoint_paths.append(epoch_models[-1])
            
    # 3. Look for final model
    if not checkpoint_paths:
        final_model_path = os.path.join(config.CHECKPOINT_DIR, 'final_model.pth')
        if os.path.exists(final_model_path):
            checkpoint_paths.append(final_model_path)
    # Example: checkpoint_paths = glob.glob(os.path.join(config.CHECKPOINT_DIR, 'fold*.pth'))
    
    models = []
    for cp_path in checkpoint_paths:
        if os.path.exists(cp_path):
            print(f"Loading checkpoint from {cp_path}")
            m = UNet3D(in_ch=config.IN_CHANNELS, out_ch=config.OUT_CHANNELS, base_ch=config.BASE_CHANNELS).to(config.DEVICE)
            
            # Load checkpoint, handling both raw state_dict and training dict
            state_dict = torch.load(cp_path, map_location=config.DEVICE)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
                
            m.load_state_dict(state_dict)
            m.eval()
            models.append(m)
        else:
            print(f"Warning: Checkpoint {cp_path} not found!")
            
    if not models:
        print("Warning: No models loaded! Using random weights.")
        m = UNet3D(in_ch=config.IN_CHANNELS, out_ch=config.OUT_CHANNELS, base_ch=config.BASE_CHANNELS).to(config.DEVICE)
        m.eval()
        models.append(m)
    
    submission_files = []
    
    for img_path in tqdm(test_images, desc="Processing Test Volumes"):
        vol_name = os.path.basename(img_path)
        print(f"Processing {vol_name}...")
        
        # Load volume
        vol = tifffile.imread(img_path)
        
        # Inference with Ensembling
        avg_prob = None
        
        for model in models:
            pred_prob = predict_sliding_window(vol, model, patch_size=config.PATCH_SIZE, overlap=0.25, device=config.DEVICE)
            if avg_prob is None:
                avg_prob = pred_prob
            else:
                avg_prob += pred_prob
        
        avg_prob /= len(models)
        
        # Postprocessing
        pred_bin = topology_postprocess(avg_prob)
        
        # Save prediction
        save_path = os.path.join(config.SUBMISSION_DIR, vol_name)
        tifffile.imwrite(save_path, pred_bin)
        submission_files.append(save_path)
        
    # Create ZIP
    zip_path = os.path.join(config.SUBMISSION_DIR, 'submission.zip')
    print(f"Creating submission zip at {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in submission_files:
            zipf.write(file, os.path.basename(file))
            
    print("Inference and submission creation complete!")

if __name__ == "__main__":
    # Test sliding window
    from model import UNet3D
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Quick test of sliding window function
    model = UNet3D(in_ch=1, out_ch=1, base_ch=16).to(device)
    dummy_vol = np.random.randn(64, 64, 64).astype(np.float32)
    
    print("Running sliding window inference test...")
    output = predict_sliding_window(dummy_vol, model, patch_size=(32, 32, 32), overlap=0.25, device=device)
    assert dummy_vol.shape == output.shape
    print("Sliding window test passed!")
    
    # Test full pipeline if test data exists
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add project root to path
    from src.config import Config
    
    # Mock config for testing
    class TestConfig(Config):
        DATA_DIR = '../data'
        SUBMISSION_DIR = '../submission'
        CHECKPOINT_DIR = '../checkpoints'
        PATCH_SIZE = (32, 32, 32)
        BASE_CHANNELS = 16 # Match the dummy model
        
    # Create dummy test data
    os.makedirs('../data/test_images', exist_ok=True)
    if not glob.glob('../data/test_images/*.tif'):
        import tifffile
        print("Creating dummy test data...")
        dummy_test_vol = np.random.randint(0, 255, (64, 64, 64), dtype=np.uint8)
        tifffile.imwrite('../data/test_images/test_vol_01.tif', dummy_test_vol)
        
    # Run inference
    # We need to save a dummy checkpoint first for the script to load, or it will warn
    os.makedirs('../checkpoints', exist_ok=True)
    torch.save(model.state_dict(), '../checkpoints/best_model.pth')
    
    run_inference(TestConfig)
