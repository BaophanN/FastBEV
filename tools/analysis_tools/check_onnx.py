import onnx

def is_onnx_model_valid(model_path):
    try:
        # Load the ONNX model
        onnx_model = onnx.load(model_path)
        
        # Check the model for validity
        onnx.checker.check_model(onnx_model)
        print("The model is valid!")
        return True
    except onnx.checker.ValidationError as e:
        print(f"The model is invalid: {e}")
        return False

# Replace 'path/to/your/model.onnx' with the actual path to your .onnx file
model_path = 'work_dirs/fastbev_trtfastbev_fp16_fuse.onnx'
is_valid = is_onnx_model_valid(model_path)
model = onnx.load(model_path)
# Print the model's structure
print(onnx.helper.printable_graph(model.graph))

# Get detailed info
print(f"IR version: {model.ir_version}")
print(f"Opset version: {model.opset_import[0].version}")
print(f"Producer name: {model.producer_name}")
print(f"Producer version: {model.producer_version}")