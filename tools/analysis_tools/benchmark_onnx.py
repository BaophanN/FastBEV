import onnxruntime as ort
import numpy as np
import time

# Path to your ONNX model
plugin = 'fastbev'
bitwidth = 'int8'

onnx_model_path = f"work_dirs/{plugin}_trt{plugin}_{bitwidth}_fuse.onnx"
# onnx_model_path = f"work_dirs/fastbev_trtfastbev_fp16_fuse.onnx"



# Load the ONNX model using GPU (if available), otherwise use CPU
session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])

# Get model input details
inputs_info = session.get_inputs()

print("Model expects the following inputs:")
for i, input_info in enumerate(inputs_info):
    print(f"Input {i}: Name={input_info.name}, Shape={input_info.shape}, Type={input_info.type}")

# Create random input data for each input tensor
input_data = {}

for input_info in inputs_info:
    input_shape = input_info.shape
    input_name = input_info.name

    # Handle dynamic batch size by setting to 1 if it's undefined (None)
    input_shape = [1 if dim is None else dim for dim in input_shape]

    # Generate random input tensor with correct shape and type
    if input_info.type == 'tensor(float)':
        input_data[input_name] = np.random.randn(*input_shape).astype(np.float32)
    elif input_info.type == 'tensor(int64)':
        input_data[input_name] = np.random.randint(0, 100, size=input_shape).astype(np.int64)
    elif input_info.type == 'tensor(int32)':
        input_data[input_name] = np.random.randint(0, 100, size=input_shape).astype(np.int32)
    else:
        raise ValueError(f"Unsupported input type: {input_info.type}")

    print(f"Prepared input: {input_name} with shape {input_shape}")

# Warm-up runs to avoid cold-start effects
print("Running warm-up iterations...")
for _ in range(10):
    session.run(None, input_data)

# Measure inference time over multiple runs
num_iterations = 100
print("Starting benchmark...")
start_time = time.time()

for _ in range(num_iterations):
    outputs = session.run(None, input_data)

end_time = time.time()

# Compute inference speed metrics
total_time = end_time - start_time
avg_inference_time = total_time / num_iterations
fps = 1 / avg_inference_time

print(f"\nBenchmark Results:")
print(f"Average inference time: {avg_inference_time:.6f} seconds")
print(f"Frames per second (FPS): {fps:.2f}")

# Print output shape
print("\nSample output shapes:")
for i, output in enumerate(outputs):
    print(f"Output {i}: Shape={output.shape}, Type={output.dtype}")

print("\nBenchmarking completed successfully.")
