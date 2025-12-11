import numpy as np

def compute_dice(pred, target, threshold=0.5):
    """
    Computes Dice coefficient for binary segmentation.
    
    Args:
        pred (np.ndarray): Probability map or binary prediction.
        target (np.ndarray): Ground truth binary mask.
        threshold (float): Threshold to binarize predictions.
        
    Returns:
        float: Dice score.
    """
    pred_bin = (pred > threshold).astype(np.float32)
    target_bin = (target > 0.5).astype(np.float32)
    
    intersection = np.sum(pred_bin * target_bin)
    union = np.sum(pred_bin) + np.sum(target_bin)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
        
    return 2.0 * intersection / union

def load_volume(path):
    import tifffile
    return tifffile.imread(path)

def save_volume(volume, path):
    import tifffile
    tifffile.imwrite(path, volume)

def compute_surface_dice(pred, target, tolerance=1.0):
    """
    Computes Surface Dice score.
    
    Args:
        pred (np.ndarray): Binary prediction.
        target (np.ndarray): Binary ground truth.
        tolerance (float): Tolerance in pixels.
        
    Returns:
        float: Surface Dice score.
    """
    from skimage import morphology
    
    # Extract boundaries
    pred_border = morphology.binary_dilation(pred) ^ morphology.binary_erosion(pred)
    target_border = morphology.binary_dilation(target) ^ morphology.binary_erosion(target)
    
    # In a real implementation, we would use distance transforms (scipy.ndimage.distance_transform_edt)
    # to find pixels within tolerance.
    # For simplicity/speed in this demo, we'll just check overlap of dilated borders.
    
    # Dilate target border by tolerance
    if tolerance > 0:
        target_border_dilated = morphology.binary_dilation(target_border, morphology.ball(tolerance))
        pred_border_dilated = morphology.binary_dilation(pred_border, morphology.ball(tolerance))
    else:
        target_border_dilated = target_border
        pred_border_dilated = pred_border
        
    # Precision: Fraction of pred surface that is close to target surface
    precision = np.sum(pred_border & target_border_dilated) / (np.sum(pred_border) + 1e-6)
    
    # Recall: Fraction of target surface that is close to pred surface
    recall = np.sum(target_border & pred_border_dilated) / (np.sum(target_border) + 1e-6)
    
    if precision + recall == 0:
        return 0.0
        
    return 2 * precision * recall / (precision + recall)

def compute_voi(pred, target):
    """
    Computes Variation of Information.
    """
    from skimage.metrics import variation_of_information
    # VOI returns (split, merge). Lower is better. We return the sum.
    split, merge = variation_of_information(target.astype(np.uint8), pred.astype(np.uint8))
    return split + merge
