import json
import os

nb_path = 'vesuvius_project/notebooks/Training.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell with train_loop(TrainConfig)
target_cell = None
for cell in nb['cells']:
    source = "".join(cell['source'])
    if "train_loop(TrainConfig)" in source:
        target_cell = cell
        break

if target_cell:
    new_code = [
        "if __name__ == \"__main__\":\n",
        "    # Ensure data exists\n",
        "    import glob\n",
        "    import numpy as np\n",
        "    import tifffile\n",
        "    import os\n",
        "    \n",
        "    if not glob.glob('../data/train_images/*.tif'):\n",
        "        print(\"Creating dummy data for training test...\")\n",
        "        os.makedirs('../data/train_images', exist_ok=True)\n",
        "        os.makedirs('../data/train_labels', exist_ok=True)\n",
        "        dummy_vol = np.random.randint(0, 255, (128, 128, 128), dtype=np.uint8)\n",
        "        dummy_lbl = np.random.randint(0, 2, (128, 128, 128), dtype=np.uint8)\n",
        "        tifffile.imwrite('../data/train_images/vol01.tif', dummy_vol)\n",
        "        tifffile.imwrite('../data/train_labels/vol01.tif', dummy_lbl)\n",
        "\n",
        "    train_loop(TrainConfig)"
    ]
    target_cell['source'] = new_code
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
    print("Notebook updated successfully.")
else:
    print("Target cell not found.")
