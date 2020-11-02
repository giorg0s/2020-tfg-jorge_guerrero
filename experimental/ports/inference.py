# /usr/bin/python3

import sys
from models import resnet50, mobilenet_v2

from torch.autograd import Variable
import torchvision.transforms as transforms

from PIL import Image
import json

# IMG = 'cat.jpg'

with open('labels.json') as f:
    fichero = json.load(f)

labels = {int(key): value for key, value in fichero.items()}


def inference(onnx_model, image):
    img = Image.open(image)
    img.show()

    # Now that we have an img, we need to preprocess it.
    # We need to:
    #       * resize the img, it is pretty big (~1200x1200px).
    #       * normalize it, as noted in the PyTorch pretrained models doc,
    #         with, mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    #       * convert it to a PyTorch Tensor.
    #
    # We can do all this preprocessing using a transform pipeline.
    min_img_size = 224  # The min size, as noted in the PyTorch pretrained models doc, is 224 px.
    transform_pipeline = transforms.Compose([transforms.Resize(min_img_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
    img = transform_pipeline(img)

    # PyTorch pretrained models expect the Tensor dims to be (num input imgs, num color channels, height, width).
    # Currently however, we have (num color channels, height, width); let's fix this by inserting a new axis.
    img = img.unsqueeze(0)  # Insert the new axis at index 0 i.e. in front of the other axes/dims.

    # Now that we have preprocessed our img, we need to convert it into a
    # Variable; PyTorch models expect inputs to be Variables. A PyTorch Variable is a
    # wrapper around a PyTorch Tensor.
    img = Variable(img)

    if onnx_model == 'resnet50':
        prediction = resnet50(img)
        print(labels[prediction])
    elif onnx_model == 'mobilenet':
        prediction = mobilenet_v2(img)
        print(labels[prediction])
    else:
        print("No existe el modelo")


def main():
    model = sys.argv[1]
    image = sys.argv[2]

    inference(model, image)


if __name__ == '__main__':
    main()
