import numpy as np

def load_data(file_path):
    """Load binary data from a file."""
    return np.fromfile(file_path, dtype=np.float32)

def compute_metrics(gt, cmp_data):
    """Compute similarity, variance, max error, and mean error between ground truth and comparison data."""
    # Compute the absolute difference between the ground truth and comparison data
    diff = np.abs(gt - cmp_data)
    
    # Similarity (normalized correlation)
    similarity = np.corrcoef(gt.flatten(), cmp_data.flatten())[0, 1]
    
    # Variance
    variance = np.var(diff)
    
    # Maximum error
    max_error = np.max(diff)
    
    # Mean error
    mean_error = np.mean(diff)
    
    return similarity, variance, max_error, mean_error

def compare_outputs(gt_path, to_cmp_list):
    """Compare the outputs from the list of comparison files with the ground truth."""
    # Load ground truth data
    gt = load_data(gt_path)
    
    for cmp_path in to_cmp_list:
        cmp_data = load_data(cmp_path)
        
        # Ensure the shapes match
        if gt.shape != cmp_data.shape:
            print(f"Shape mismatch between {gt_path} and {cmp_path}")
            continue
        
        # Compute metrics
        similarity, variance, max_error, mean_error = compute_metrics(gt, cmp_data)
        
        # Print results
        print(f"Comparing {cmp_path} with {gt_path}:")
        print(f"  Similarity: {similarity:.4f}")
        print(f"  Variance: {variance:.4e}")
        print(f"  Maximum error: {max_error:.4e}")
        print(f"  Mean error: {mean_error:.4e}\n")

# Define paths
torch_output_path = "torch_output.bin"
cpu_output_path = "cpu_output.bin"
cpp_output_path = "cpp_output.bin"

# Ground truth and comparison list
gt_path = torch_output_path
to_cmp_list = [cpu_output_path,cpp_output_path]

# Run comparison
compare_outputs(gt_path, to_cmp_list)
