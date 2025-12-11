# src/train.py
import os
import glob
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# use new amp API
from torch import amp

import tifffile
import numpy as np

from dataset import ScrollDataset
from model import UNet3D
from losses import VesuviusLoss
from config import Config


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    count = 0
    # validation on patches (assumes loader yields patches)
    with torch.no_grad():
        with amp.autocast(device_type=device, enabled=(device != 'cpu')):
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                total_loss += float(loss.item())
                count += 1

    return total_loss / max(1, count)


def train_loop(config: Config):
    # instantiate config
    config = config() if callable(config) else config

    # build file lists
    train_images = sorted(glob.glob(os.path.join(config.DATA_DIR, 'train_images', '*.tif')))
    train_labels = sorted(glob.glob(os.path.join(config.DATA_DIR, 'train_labels', '*.tif')))

    assert len(train_images) > 0, f"No train images found in {os.path.join(config.DATA_DIR, 'train_images')}"
    assert len(train_labels) > 0, f"No train labels found in {os.path.join(config.DATA_DIR, 'train_labels')}"

    # quick train/val split (keep small val set)
    n_total = len(train_images)
    n_val = int(0.05 * n_total)  # 5%
    
    if n_val == 0:
        # If dataset is too small, use all for training and same for validation (or empty)
        # For safety in this demo, let's use all for training
        train_img_paths = train_images
        train_lbl_paths = train_labels
        # And maybe use the same for validation to avoid empty loader issues if any
        val_img_paths = train_images
        val_lbl_paths = train_labels
    else:
        n_val = min(6, n_val)
        train_img_paths = train_images[:-n_val]
        train_lbl_paths = train_labels[:-n_val]
        val_img_paths = train_images[-n_val:]
        val_lbl_paths = train_labels[-n_val:]

    # Dataset & DataLoader
    train_ds = ScrollDataset(train_img_paths, train_lbl_paths, patch_size=config.PATCH_SIZE, train=True)
    val_ds = ScrollDataset(val_img_paths, val_lbl_paths, patch_size=config.PATCH_SIZE, train=False)

    pin_memory = True if config.DEVICE.startswith('cuda') else False
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=max(0, config.NUM_WORKERS // 2), pin_memory=pin_memory
    )

    # Model, Loss, Optimizer
    device = config.DEVICE
    model = UNet3D(in_ch=config.IN_CHANNELS, out_ch=config.OUT_CHANNELS, base_ch=config.BASE_CHANNELS).to(device)
    criterion = VesuviusLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.EPOCHS))
    scaler = amp.GradScaler(enabled=(device != 'cpu'))

    # prepare checkpoints directory
    Path(config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print(f"Starting training on {device} with {len(train_loader)} steps per epoch...")

    for epoch in range(config.EPOCHS):
        model.train()
        epoch_loss = 0.0
        iters = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}", unit="it")
        optimizer.zero_grad()

        for i, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with amp.autocast(device_type=device, enabled=(device != 'cpu')):
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / config.ACCUMULATION_STEPS

            # backward with scaler
            scaler.scale(loss).backward()

            # step
            if (i + 1) % config.ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # accumulate stats (use the un-divided loss)
            epoch_loss += float(loss.item()) * config.ACCUMULATION_STEPS
            iters += 1

            pbar.set_postfix({'loss': f"{(epoch_loss / max(1, iters)):.4f}"})

            # optional: early break to limit steps per epoch
            if hasattr(config, 'STEPS_PER_EPOCH') and config.STEPS_PER_EPOCH is not None:
                if iters >= config.STEPS_PER_EPOCH:
                    break

        # scheduler step after epoch
        scheduler.step()

        avg_train_loss = epoch_loss / max(1, iters)
        train_losses.append(avg_train_loss)

        # Validation
        if (epoch + 1) % config.VAL_INTERVAL == 0:
            val_loss = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")

            # checkpoint best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(config.CHECKPOINT_DIR, f'best_model_epoch{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict()
                }, save_path)
                print(f"Saved best model to {save_path}")

    # After training: save final model
    torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, "final_model.pth"))

    # return losses for plotting if needed
    return train_losses, val_losses


if __name__ == "__main__":
    # create minimal dummy data if none exists (safe test)
    os.makedirs('../data/train_images', exist_ok=True)
    os.makedirs('../data/train_labels', exist_ok=True)
    os.makedirs('../checkpoints', exist_ok=True)

    if len(glob.glob('../data/train_images/*.tif')) == 0:
        print("Creating dummy data for quick test...")
        dummy_vol = np.random.randint(0, 255, (128, 128, 128), dtype=np.uint8)
        dummy_lbl = np.random.randint(0, 2, (128, 128, 128), dtype=np.uint8)
        tifffile.imwrite('../data/train_images/vol01.tif', dummy_vol)
        tifffile.imwrite('../data/train_labels/vol01.tif', dummy_lbl)

    # Use Config class
    cfg = Config()
    # override for quick local test
    cfg.EPOCHS = 2
    cfg.PATCH_SIZE = (64, 64, 64)
    cfg.STEPS_PER_EPOCH = 50
    train_losses, val_losses = train_loop(cfg)
    print("Training finished.")
