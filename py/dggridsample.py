import torch
import numpy as np
import os
def grid_sample_cpu(input_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=True):
    # Get input dimensions
    n, c, h, w = input_tensor.shape
    _, grid_h, grid_w, _ = grid.shape  # 修复为适应 4D grid
    
    # Compute scaling factor
    if align_corners:
        x_scale = (w - 1) / 2
        y_scale = (h - 1) / 2
    else:
        x_scale = w / 2
        y_scale = h / 2
    
    # Normalize grid to pixel indices
    x = (grid[..., 0] + 1) * x_scale
    y = (grid[..., 1] + 1) * y_scale
    
    # Clamp indices for padding_mode
    if padding_mode == 'zeros':
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
    
    # Compute integer and fractional parts
    x0 = x.astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = y.astype(int)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x_frac = x - x0
    y_frac = y - y0
    
    # Gather values for bilinear interpolation
    q00 = input_tensor[:, :, y0, x0]
    q01 = input_tensor[:, :, y1, x0]
    q10 = input_tensor[:, :, y0, x1]
    q11 = input_tensor[:, :, y1, x1]
    
    # Perform bilinear interpolation
    interp_top = q00 * (1 - x_frac) + q10 * x_frac
    interp_bottom = q01 * (1 - x_frac) + q11 * x_frac
    output = interp_top * (1 - y_frac) + interp_bottom * y_frac
    
    return output


# Test the function and compare with PyTorch
# 文件路径
input_tensor_file = "input_tensor.bin"
grid_file = "grid.bin"

# 检查文件是否存在并加载数据
if os.path.exists(input_tensor_file) and os.path.exists(grid_file):
    print("Loading data from bin files...")
    input_tensor = np.fromfile(input_tensor_file, dtype=np.float32).reshape(1, 3, 10, 10)
    grid = np.fromfile(grid_file, dtype=np.float32).reshape(1, 10, 10, 2)
else:
    print("Generating new data and saving to bin files...")
    input_tensor = torch.randn(1, 3, 10, 10, requires_grad=False).numpy()  # Input feature map
    grid = torch.rand(1, 10, 10, 2, requires_grad=False).numpy() * 2 - 1  # Normalized grid
    input_tensor.tofile(input_tensor_file)
    grid.tofile(grid_file)
    input_tensor.tofile(input_tensor_file)
    grid.tofile(grid_file)

print(f"input_tensor {input_tensor}")
print(f"grid {grid}")
# PyTorch grid_sample
torch_output = torch.nn.functional.grid_sample(torch.tensor(input_tensor), torch.tensor(grid), mode='bilinear', padding_mode='zeros', align_corners=True)

# Custom CPU implementation
cpu_output = grid_sample_cpu(input_tensor, grid)
torch_output = torch_output.flatten()
cpu_output = cpu_output.flatten()
print(f"torch {torch_output}")
print(f"cpu {cpu_output}")
# Compute difference
difference = np.mean((torch_output.numpy() - cpu_output) ** 2)
print(f"Mean Squared Difference: {difference}")
print(f"outputshape {cpu_output.shape}")

# save result
torch_output_path = "torch_output.bin"
cpu_output_path = "cpu_output.bin"
cpu_output.astype(np.float32).tofile(cpu_output_path)
torch_output.numpy().tofile(torch_output_path)
