import io

from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
import torch.onnx

from PIL import Image
import requests

# Random cat img taken from Google
IMG_URL = 'cat.jpg'
# Class labels used when training VGG as json, courtesy of the 'Example code' link above.
LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'

# Let's get our class labels.
response = requests.get(LABELS_URL)  # Make an HTTP GET request and store the response.
labels = {int(key): value for key, value in response.json().items()}

# Let's get the cat img.
img = Image.open(IMG_URL)

# Let's take a look at this cat!
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

resnet50 = models.resnet50(pretrained=True)
resnet50.eval()

prediction = resnet50(img)
prediction = prediction.data.numpy().argmax()  # Our prediction will be the index of the class label with the largest
# value.
print(labels[prediction])

# Export the model
torch.onnx.export(resnet50,  # model being run
                  img,  # model input (or a tuple for multiple inputs)
                  "resnet50.onnx",  # where to save the model (can be a file or file-like object)
                  verbose=True,
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                'output': {0: 'batch_size'}})
