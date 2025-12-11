import torch
from src.model import UNet3D
from src.config import Config
import sys
import os

# Add src to path
sys.path.append('vesuvius_project/src')

def test_model():
    print(f"Testing model on {Config.DEVICE}...")
    print(f"Patch size: {Config.PATCH_SIZE}")
    
    try:
        model = UNet3D(in_ch=Config.IN_CHANNELS, out_ch=Config.OUT_CHANNELS, base_ch=Config.BASE_CHANNELS).to(Config.DEVICE)
        print("Model initialized.")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
        scaler = torch.cuda.amp.GradScaler(enabled=(Config.DEVICE != 'cpu'))
        
        # Create dummy input
        x = torch.randn(1, 1, *Config.PATCH_SIZE).to(Config.DEVICE)
        y_true = torch.randn(1, 1, *Config.PATCH_SIZE).to(Config.DEVICE)
        
        print(f"Input shape: {x.shape}")
        
        # Forward pass with autocast
        print("Starting forward pass with autocast...")
        # Note: torch.cuda.amp.autocast is for CUDA. For CPU, use torch.cpu.amp.autocast or torch.autocast(device_type='cpu')
        # But the code uses torch.cuda.amp.autocast. Let's see if that crashes on CPU.
        
        use_cuda_amp = (Config.DEVICE != 'cpu')
        
        with torch.cuda.amp.autocast(enabled=use_cuda_amp):
            y = model(x)
            loss = ((y - y_true)**2).mean()
        
        print(f"Output shape: {y.shape}")
        print("Forward pass successful.")
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print("Backward pass successful.")
        
    except Exception as e:
        print(f"Caught exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
