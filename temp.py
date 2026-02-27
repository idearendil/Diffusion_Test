import os
import torch

target_dir = "backtest/tensor_data/2020-01-01/train"

shape_dict = {}

for filename in os.listdir(target_dir):
    if filename.endswith("_x.pt"):
        file_path = os.path.join(target_dir, filename)
        tensor = torch.load(file_path, map_location="cpu")
        shape_dict[filename] = tuple(tensor.shape)

if shape_dict:
    filenames = list(shape_dict.keys())
    standard_shape = shape_dict[filenames[0]]

    print(f"Standard shape (first file): {standard_shape}\n")
    print("Files with different shape:")

    for filename, shape in shape_dict.items():
        if shape != standard_shape:
            print(f"{filename} → shape: {shape}")