# /usr/bin/python3

import numpy as np
import tensorflow as tf
from PIL import Image
from torchvision.transforms import ToTensor
from torch.autograd import Variable

img = Image.open("resources/imgs/cat.jpg")
img_tensor = ToTensor()(img).unsqueeze(0)
cat_img = Variable(img_tensor)


def tf_to_tflite():
    converter = tf.lite.TFLiteConverter.from_saved_model(
        "exported-models/resnet50-tf")  # path to the SavedModel directory
    tflite_model = converter.convert()

    # Save the model
    with open('resnet50.tflite', 'wb') as f:
        f.write(tflite_model)

    print('\nTesting the exported model...')

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)


if __name__ == '__main__':
    tf_to_tflite()
