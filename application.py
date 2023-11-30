import torch.nn as nn
import googlenet_pytorch
import torchvision.transforms as transforms

import torch
from PIL import Image
#print(torch.__version__)
import torch.nn.functional as F

from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np

import base64
import cv2

from fastapi.responses import FileResponse
import tempfile
import shutil
import os

# GoogleNet Module
class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet,self).__init__()
        #self.model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=False, progress=True)
        self.model = googlenet_pytorch.GoogLeNet.from_pretrained('googlenet')
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.layernorm = nn.LayerNorm(1024,elementwise_affine=True)
        self._fc = nn.Linear(1024,2, bias=False)
    def forward(self, x):
        batch_size ,_,_,_ =x.shape
        x = self.model.extract_features(x)
        x = self.model.avgpool(x)
        x = x.view(-1, 1024)
        x = self.layernorm(x)
        x = self._fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# Model Evaluation
model = GoogleNet()#*args, **kwargs)
model_dir = '/app/model'
# Load the model from the saved file
model.load_state_dict(torch.load(os.path.join(model_dir, 'saved_model.pt')))

# model.load_state_dict(torch.load('model\saved_model.pt'))
model.eval()




# Lime Explanation
explainer = lime_image.LimeImageExplainer()

pil_image_transform = transforms.Compose([
    transforms.Resize((224, 224))
])

lime_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

def batch_predict(images):
    model.eval()
  
    batch = torch.stack(tuple(lime_transform(i) for i in images), dim=0)

    device = torch.device("cpu")
    model.to(device)
    batch = batch.to(device)
    
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()



def get_lime_exp(path_for_image):
    input_image = Image.open(path_for_image).convert('RGB')
 
    explanation = explainer.explain_instance(np.array(pil_image_transform(input_image)),
                                            batch_predict, batch_size=10 , num_samples=20)

    #print(explanation.local_exp)
    temp, mask = explanation.get_image_and_mask(1,positive_only=True)# 0:'Norm',1:'MI'
 
    img_boundry1 = mark_boundaries(temp/255.0, mask, color=[0.5,0,0])
    
    # results = {
    #     'message': 'Explanation generated successfully',
    #     'image': get_base64_image(explanation)  # Convert the image to base64
    # }

    # return results

    # Save the explanation image to a temporary file
    temp_file_path = save_temp_image(explanation)


    # Return the image using FileResponse
    return FileResponse(temp_file_path, media_type='image/png', filename='explanation.png')

# def get_base64_image(explanation):
#     temp, mask = explanation.get_image_and_mask(1, positive_only=True)
#     img_boundry1 = mark_boundaries(temp / 255.0, mask, color=[0.5, 0, 0])

#     # Convert the image to base64
#     _, buffer = cv2.imencode('.png', img_boundry1)
#     img_base64 = base64.b64encode(buffer).decode('utf-8')

#     return img_base64


def save_temp_image(explanation):
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, 'explanation.png')

    temp, mask = explanation.get_image_and_mask(1, positive_only=True)
    img_boundry1 = mark_boundaries(temp / 255.0, mask, color=[0.5, 0, 0])

    cv2.imwrite(temp_file_path, cv2.cvtColor((img_boundry1 * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    return temp_file_path
