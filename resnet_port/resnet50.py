import torch
import torch.utils.model_zoo as model_zoo
from resnet import resnet50

MODEL_URL = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

resnet_model = resnet50(pretrained=True, progress=True)

map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
resnet_model.load_state_dict(model_zoo.load_url(MODEL_URL, map_location=map_location))

resnet_model.eval()
