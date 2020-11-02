# /usr/bin/python3

import torchvision.models as models
import torch.onnx


def resnet50(image):
    model = models.resnet50(pretrained=True)
    model.eval()

    prediction = model(image)
    prediction = prediction.data.numpy().argmax()

    # Export the model
    torch.onnx.export(model,  # model being run
                      image,  # model input (or a tuple for multiple inputs)
                      "resnet50.onnx",  # where to save the model (can be a file or file-like object)
                      verbose=False,
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    return prediction


def mobilenet_v2(image):
    model = models.mobilenet_v2(pretrained=True)
    model.eval()

    prediction = model(image)
    prediction = prediction.data.numpy().argmax()

    # Export the model
    torch.onnx.export(model,  # model being run
                      image,  # model input (or a tuple for multiple inputs)
                      "mobilenet_v2.onnx",  # where to save the model (can be a file or file-like object)
                      verbose=False,
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})
    return prediction
