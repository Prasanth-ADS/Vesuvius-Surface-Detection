class Config:
    # Data
    PATCH_SIZE = (64, 64, 64)
    BATCH_SIZE = 2               # Faster with gradient accumulation
    NUM_WORKERS = 4              # Use 4 on Windows (with spawn)
    
    # Model
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    BASE_CHANNELS = 16           # Reduce model size by 4× (much faster)
    
    # Training
    EPOCHS = 12                  # 12 epochs is enough for convergence
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    ACCUMULATION_STEPS = 2       # Higher effective batch size
    STEPS_PER_EPOCH = 1000       # Add this to control training speed
    VAL_INTERVAL = 1
    
    # Paths
    DATA_DIR = '../data'
    CHECKPOINT_DIR = '../checkpoints'
    SUBMISSION_DIR = '../submission'
    
    # Device
    DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
