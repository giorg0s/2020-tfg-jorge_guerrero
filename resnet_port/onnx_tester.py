import onnx

onnx_model = onnx.load("cat.onnx")
onnx.checker.check_model(onnx_model)