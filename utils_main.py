import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import pickle
import io
import base64
import TLCNN
from collections import Counter

#defining pre-trained models path
cnn_path = "./models/scratch_v2.pt"


# defining class labels 
classes = ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold', 'Septoria Leaf Spot',
           'Spider Mites', 'Target Spot', 'Yellow Leaf Curl Virus',
           'Mosaic Virus', 'Healthy']


###################### IMAGE AUGMENTATION & TRANSFORMATION ######################
# data transform-> RGB to HSV, resize, to tensor, normalization and batch shape
def img_transformation(input_img):
  
  # defining mean and standard deviation for normalization
  mean = [0.5, 0.5, 0.5]
  std = [0.5, 0.5, 0.5]

  # defining transformation
  data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
                        ])
  
  img_bytes = input_img.file.read()
  img = Image.open(io.BytesIO(img_bytes))  # it needs then convert into numpy to work with opencv
  cv_img = np.array(img)

  # encoded image 
  encoded_string = base64.b64encode(img_bytes)
  bs64 = encoded_string.decode('utf-8')
  encoded_img = f'data:image/jpeg;base64,{bs64}'
  
  # convert that image into PIL format for image transformation
  pil_img = Image.fromarray(cv_img)
  tensor_img = data_transforms(pil_img).unsqueeze(0)
  return cv_img, encoded_img, tensor_img


# Function to generate grad-cam

def generate_grad_cam(model, target_layer, input_image):
    # model.eval()
    input_image.requires_grad_()

    # Forward pass
    output = model(input_image)
    model.zero_grad()
    class_idx= torch.argmax(output)
    class_loss = output[0, class_idx]
    class_loss.backward()

    # Extract feature maps and gradients
    feature_map = model.feature_maps[target_layer][0].cpu().detach().numpy()
    gradients = model.gradients[target_layer][0].cpu().detach().numpy()

    # Global average pooling to get weights
    weights = np.mean(gradients, axis=(1, 2), keepdims=True)

    # Create Grad-CAM heatmap
    grad_cam = np.sum(weights * feature_map, axis=0)
    grad_cam = np.maximum(grad_cam, 0)  # Apply ReLU
    grad_cam = grad_cam - np.min(grad_cam)
    grad_cam = grad_cam / np.max(grad_cam)  # Normalize to [0, 1]

    return class_idx, grad_cam

# Overlaying gradcam with input image 
def overlay_heatmap(grad_cam, cv_img):

    input_image = cv_img

    # Resize the heatmap to the input image size
    grad_cam = cv2.resize(grad_cam, (input_image.shape[1], input_image.shape[0]))
    # Convert Grad-CAM heatmap to color map
    heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam), cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.35 + input_image

    return superimposed_img


###################### FEATURE EXTRACTION & GRAD-CAM GENERATION #############################

def get_pred_n_gradcam(input_img):

  target_layer= 'block3_conv2'

  model = TLCNN.CamModel
  # image transformation
  org_img, org_encoded_img, img = img_transformation(input_img)
  
  # forward pass 
  pred_class, grad_cam = generate_grad_cam(model, target_layer, input_image= img)
  heatmap_img = overlay_heatmap(grad_cam, org_img)

  # encoded cam image 
  # Convert the OpenCV image to bytes
  retval, buffer = cv2.imencode('.jpg', heatmap_img)
  img_bytes = np.array(buffer).tobytes()

  # Encode the image bytes to base64
  encoded_string = base64.b64encode(img_bytes)
  bs64 = encoded_string.decode('utf-8')

  # Create the data URI
  cam_encoded_img = f'data:image/jpeg;base64,{bs64}'

  return org_encoded_img, cam_encoded_img, pred_class


###################### PREDICTION VIA AN INPUT IMAGE ######################
def get_prediction(input_img, is_api = False):
  
  # getting original image, grad-cam, image features 
  org_encoded_img, cam_encoded_img, pred_class =  get_pred_n_gradcam(input_img)

  # getting predicted class name 
  disease_name = classes[int(pred_class)]

  pred_results = {
            "class_name": disease_name,
        }
  
  # conditionally add image data to the result dictionary
  if not is_api:
      pred_results["org_encoded_img"] = org_encoded_img
      pred_results["cam_encoded_img"] = cam_encoded_img

  return pred_results
