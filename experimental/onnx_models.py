# /usr/bin/python3

import os
import time
import sys
import onnxruntime
from PIL import Image
import torchvision.models as models
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import onnx
import torch.onnx
import numpy as np

from onnx_tf.backend import prepare

EXPORT_MOBILENET = "exported-models/mobilenet_v2.onnx"
EXPORT_RESNET = "exported-models/resnet50.onnx"

img = Image.open("resources/imgs/cat.jpg")
img_tensor = ToTensor()(img).unsqueeze(0)
cat_img = Variable(img_tensor)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def export_model():
    if sys.argv[1] == 'resnet50':
        # Load PyTorch model
        original_model = models.resnet50(pretrained=True)
        original_model.eval()
        export_file = EXPORT_RESNET
    elif sys.argv[1] == 'mobilenet':
        original_model = models.mobilenet_v2(pretrained=True)
        original_model.eval()
        export_file = EXPORT_MOBILENET

    start = time.time()

    # Export the model
    torch.onnx.export(original_model,  # model being run
                      cat_img,  # model input (or a tuple for multiple inputs)
                      export_file,  # where to save the model (can be a file or file-like object)
                      verbose=False,
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    end = time.time()
    final_time = '%.3f' % (end - start)
    print("PyTorch model has been exported successfully to ONNX in " + str(final_time) + " seconds")

    print("Testing ONNX model...")

    ort_session = onnxruntime.InferenceSession(export_file)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(cat_img)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(original_model(cat_img)), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    # onnx_model = onnx.load(filename)
    # onnx.checker.check_model(onnx_model)

    original_stdout = sys.stdout
    exported_model = onnx.load(export_file)
    # Export also the graph that represents the ONNX model
    with open(os.path.splitext(export_file)[0] + "_onnx.txt", 'w') as f:
        sys.stdout = f
        print('{}'.format(onnx.helper.printable_graph(exported_model.graph)))
        sys.stdout = original_stdout

    print("Exporting ONNX model to TF-2.3.1...")
    onnx_to_tf(export_file, os.path.splitext(export_file)[0] + "-tf")

    print("\nDONE")
    sys.exit()


def onnx_to_tf(onnx_model, output):
    onnx_model = onnx.load(onnx_model)  # load onnx model
    tf_rep = prepare(onnx_model)  # prepare tf representation
    tf_rep.export_graph(output)  # export the model


if __name__ == '__main__':
    export_model()
