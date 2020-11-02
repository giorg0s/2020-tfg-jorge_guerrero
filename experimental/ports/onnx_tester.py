# /usr/bin/python3

import os
import sys
import onnx
import onnxruntime

original_stdout = sys.stdout

filename = sys.argv[1]

model = onnx.load(filename)
onnx.checker.check_model(model)

with open(os.path.splitext(filename)[0]+"_onnx.txt", 'w') as f:
    sys.stdout = f
    print('Model :\n\n{}'.format(onnx.helper.printable_graph(model.graph)))
    sys.stdout = original_stdout



