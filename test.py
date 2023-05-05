# -*- coding: utf-8 -*-
import argparse
import urllib.request
import torch
import torchvision.models
import torchvision.transforms as transforms
from collections import OrderedDict
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_image(image):
    # TODO: optimize function for faster URL -> rgb conversion
    if image.mode != 'RGB':
        image = image.convert("RGB")
    Transform = transforms.Compose([
            transforms.Resize([224,224]),      
            transforms.ToTensor(),
            ])
    image = Transform(image)   
    image = image.unsqueeze(0)
    return image.to(device)

def predict(image, model):
    image = prepare_image(image)
    with torch.no_grad():
        preds = model(image)

    return preds.item()

## The function should return a list of predicted image scores, along with image urls/ids
def get_image_scores(urls):
    # Prepare Model
    model = torchvision.models.resnet50()
    # model.avgpool = nn.AdaptiveAvgPool2d(1) # for any size of the input
    model.fc = torch.nn.Linear(in_features=2048, out_features=1)
    model.load_state_dict(torch.load('intrinsic_image_popularity/model/model-resnet50.pth', map_location=device)) 
    model.eval().to(device)

    scores = OrderedDict()
    for url in urls:
        urllib.request.urlretrieve(url, 'downloaded_image.jpg')
        image = Image.open('downloaded_image.jpg')
        model.eval().to(device)
        prediction = predict(image, model)
        scores[url] = prediction
    return sort_image_scores(scores)

def sort_image_scores(scores):
    # Sorted Descending. A higher IIPA value indicates better intrinsic popularity (see IIPA paper)
    sorted_scores = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    return sorted_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='images/0.jpg')
    config = parser.parse_args()
    image = Image.open(config.image_path)
    model = torchvision.models.resnet50()
    # model.avgpool = nn.AdaptiveAvgPool2d(1) # for any size of the input
    model.fc = torch.nn.Linear(in_features=2048, out_features=1)
    model.load_state_dict(torch.load('intrinsic_image_popularity/model/model-resnet50.pth', map_location=device)) 
    model.eval().to(device)
    predict(image, model)
