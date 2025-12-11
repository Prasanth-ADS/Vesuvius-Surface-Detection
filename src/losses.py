import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (B, 1, D, H, W)
        # targets: (B, 1, D, H, W)
        
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, logits, targets):
        # Placeholder for true SDF loss. 
        # A simple proxy is to penalize predictions far from the boundary if we had distance maps.
        # Without pre-computed SDF, we can use a weighted BCE or similar.
        # For this implementation, let's assume targets are just binary masks.
        # We'll implement a simple "distance" penalty by eroding/dilating the mask on the fly? 
        # No, that's too slow.
        # Let's implement a weighted BCE where edges are weighted higher.
        
        # For now, let's stick to a placeholder that returns 0 if we don't have SDF maps.
        # Or better, let's implement Tversky loss as an "advanced" loss.
        return torch.tensor(0.0, device=logits.device)

class VesuviusLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, boundary_weight=0.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        boundary_loss = self.boundary(logits, targets)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss + self.boundary_weight * boundary_loss

if __name__ == "__main__":
    # Test losses
    loss_fn = VesuviusLoss()
    logits = torch.randn(2, 1, 32, 32, 32)
    targets = torch.randint(0, 2, (2, 1, 32, 32, 32)).float()
    
    loss = loss_fn(logits, targets)
    print(f"Loss: {loss.item()}")
