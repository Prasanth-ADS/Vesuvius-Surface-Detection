import torch
import sympy
print(f"Torch version: {torch.__version__}")
print(f"Sympy version: {sympy.__version__}")
try:
    import sympy.core
    print("sympy.core imported successfully")
except ImportError as e:
    print(f"Failed to import sympy.core: {e}")

try:
    from torch.optim import AdamW
    from torch.nn import Linear
    model = Linear(10, 10)
    optimizer = AdamW(model.parameters(), lr=0.01)
    print("AdamW initialized successfully")
except Exception as e:
    print(f"Failed to initialize AdamW: {e}")
