import numpy as np
from skimage import morphology, measure

def topology_postprocess(pred_prob, threshold=0.5, min_size=100):
    """
    Applies topology-aware postprocessing to the prediction volume.
    
    Args:
        pred_prob (np.ndarray): Probability volume (D, H, W).
        threshold (float): Binarization threshold.
        min_size (int): Minimum size of connected components to keep.
        
    Returns:
        np.ndarray: Cleaned binary volume.
    """
    # 1. Binarize
    binary = pred_prob > threshold
    
    # 2. Morphological Cleaning
    # Remove small noise (Opening)
    binary = morphology.binary_opening(binary, morphology.ball(1))
    # Fill holes (Closing)
    binary = morphology.binary_closing(binary, morphology.ball(2))
    
    # Remove small holes (Area Closing) - effectively filling small internal cavities
    # binary = morphology.area_closing(binary, area_threshold=100) # Available in newer skimage versions
    # If not available, binary_closing with larger kernel helps.
    
    # 3. Keep Largest Component (or remove small ones)
    labels = measure.label(binary)
    if labels.max() > 0:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0 # Ignore background
        
        # Filter components smaller than min_size
        mask = np.zeros_like(binary, dtype=bool)
        for i in range(1, labels.max() + 1):
            if sizes[i] >= min_size:
                mask |= (labels == i)
        binary = mask
        
        # Optional: Keep only the largest component if we expect a single sheet
        # largest_label = sizes.argmax()
        # binary = (labels == largest_label)
    
    # 4. Skeletonization (Optional, depending on if we want a surface or a volume)
    # For surface detection, we often want a thin sheet.
    # skeleton = morphology.skeletonize(binary)
    
    # For now, let's return the cleaned binary volume. 
    # If the goal is a surface for metrics like Surface Dice, skeletonization is good.
    # But for submission, sometimes a slightly thick volume is safer.
    # Let's return the binary volume for now, and expose skeletonization as an option.
    
    return binary.astype(np.uint8)

def skeletonize_volume(binary_vol):
    """
    Skeletonizes the binary volume to get a 1-pixel wide surface.
    """
    return morphology.skeletonize(binary_vol).astype(np.uint8)

if __name__ == "__main__":
    # Test postprocessing
    print("Testing postprocessing...")
    
    # Create a dummy volume with a "sheet" and some noise
    shape = (64, 64, 64)
    vol = np.zeros(shape, dtype=np.float32)
    
    # Create a plane (sheet)
    vol[30:34, :, :] = 0.9
    
    # Add noise
    vol[10, 10, 10] = 0.8 # Small noise
    vol[50, 50, 50] = 0.8 # Small noise
    
    # Run postprocess
    cleaned = topology_postprocess(vol, min_size=10)
    
    print(f"Original sum: {(vol > 0.5).sum()}")
    print(f"Cleaned sum: {cleaned.sum()}")
    
    # Check if noise is removed (should be, if min_size is appropriate)
    # The plane is 4*64*64 = 16384 pixels. Noise is 1 pixel.
    assert cleaned[10, 10, 10] == 0, "Noise not removed!"
    assert cleaned[32, 32, 32] == 1, "Sheet removed!"
    
    print("Postprocessing test passed!")
