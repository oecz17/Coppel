# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 18:05:20 2022

@author: omar.contreras
"""

import torch
import hnswlib
import requests
import torchvision
import torch.utils.data
import numpy as np
import torchvision.models as models

from io import BytesIO
from PIL import Image
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from flask_cors import CORS
from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
CORS(app)
api = Api(app)

num2label_fast = {1:'Lavadoras', 2:'Recamaras', 3:'MueblesparaTV', 4:'Escritorios'}

muebles = ['chair', 'motorcycle', 'refrigerator', 'couch', 'dinning table']


#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
detr = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')


fast = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 5  # 1 class (person) + background
# get number of input features for the classifier
in_features = fast.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
fast.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

fast.load_state_dict(torch.load('FASTRCNN_furn_4.pth'))
fast.eval()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
model = torch.nn.Sequential(*list(wide_resnet50_2.children())[:-1])
        
embedding_size  = 2048

def preprocess(r):
    r = requests.get(r, allow_redirects=True)
    img = BytesIO(r.content)
    img = Image.open(img)
    if img.mode != 'RGB':
        img = img.convert('RGB')
        print('Converting...')
    return img

def pretrained_pred(im):
    inputs = feature_extractor(images=im, return_tensors="pt")
    outputs = detr(**inputs)

    # keep only predictions of queries with 0.9+ confidence (excluding no-object class)
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.85
    
    labels = probas.argmax(1)[keep].cpu().detach().numpy()
    scores = probas.amax(1)[keep].cpu().detach().numpy()
    torch.cuda.empty_cache()
    
    return labels.tolist(), scores.tolist()

def trained_pred(img):
    fast.to(device)

    img =  np.array(img)/255.0
    img = img[np.newaxis,...]
    img = torch.from_numpy(img).permute(0,3,1,2)
    img = img.type(torch.FloatTensor)
    
    with torch.no_grad():
        prediction = fast(img.to(device))
    
    keep2 = prediction[0]['scores'] > 0.8

    scores = prediction[0]['scores'][keep2].cpu().detach().numpy()
    labels = prediction[0]['labels'][keep2].cpu().detach().numpy()
    torch.cuda.empty_cache()
    
    return labels.tolist(), scores.tolist()

def reduce_label(labels, scores, pretrained=True):
    new_scores = []
    new_labels = []
    for i,l in enumerate(labels):
        if pretrained:
        #print(l)
            label = detr.config.id2label[l]
            if label in muebles:
                new_labels.append(label)
                new_scores.append(scores[i])        
        else:
            label = num2label_fast[l]
            new_labels.append(label)
            new_scores.append(scores[i])   
    return new_labels, new_scores
       
def calculate_catalog_features(img, model):
    
    img = torch.from_numpy(img).permute(0,3,1,2)
    img = img.type(torch.FloatTensor)
    
    model.eval()
    output = model(img)
    output = output/np.linalg.norm(output.detach().numpy())
    features = output.detach().numpy().squeeze()
    return features

def similar(features, embedding_size, index_name):

    print("similar index")
    p = hnswlib.Index(
        space='l2',
        dim=embedding_size)  # possible options are l2, cosine or ip
    print("initializing index")

    p.load_index(index_name, max_elements = len(features))
    labels, distances = p.knn_query(features, k = 3)

    return labels, distances

    
class FurnDetection(Resource):
    def post(self):
        r = request.get_json(force=True)['url'] 
        img = preprocess(r)
        
        pre_labels, pre_scores = pretrained_pred(img)
        labels, scores = trained_pred(img)
        
        pre_labels, pre_scores = reduce_label(pre_labels, pre_scores)
        labels, scores = reduce_label(labels, scores, pretrained=False)
        #print(labels, scores)
        return {'labels_pre': pre_labels, 'scores_pre': pre_scores, 
                'labels': labels, 'scores': scores}
    
        #return {'labels': labels, 'scores': scores}


class FurnPrediction(Resource):
    def post(self):
        res = request.get_json(force=True)
        img = preprocess(res['url'])
        img = img.resize((224,224))
        img =  np.array(img)/255.0
        img = img[np.newaxis,...]
        
        features = calculate_catalog_features(img, model)
        
        indexname = 'index/'+ res['label'] + '.idx'
        #if indexname == 'chair'
        #print(indexname)
        labels, distances = similar(features, embedding_size, indexname)
        labels = labels.tolist()
        distances = distances.tolist()
        return {'labels': labels, 'distances': distances}

@app.route('/home', methods=['GET'])
def home():
    return 'Home - Visual Detection'


api.add_resource(FurnDetection, '/api/detection')

api.add_resource(FurnPrediction, '/api/similar')

if __name__ == '__main__':
    app.run(debug=False)
