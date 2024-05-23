import boto3
import os
import imutils
import cv2
import json
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN, InceptionResnetV1
from shutil import rmtree
import numpy as np
import torch
os.environ['TORCH_HOME'] = '/tmp/'

print('Inside the face recognition function')

s3 = boto3.client('s3')

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def face_recognition_function(key_path):
    print('inside the face recognition method')
    img = cv2.imread(key_path, cv2.IMREAD_COLOR)
    boxes, _ = mtcnn.detect(img)

    key = os.path.splitext(os.path.basename(key_path))[0].split(".")[0]
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    face, prob = mtcnn(img, return_prob=True, save_path=None)

    print('downloading data.pt from s3 bucket')
    data_file = s3.get_object(Bucket='pytorch-data-cv', Key='data.pt')
    data_path = "/tmp/data.pt"
    with open(data_path, 'wb') as f:
        f.write(data_file['Body'].read())
        
    saved_data = torch.load('/tmp/data.pt')
    # saved_data = torch.load('/Users/adrijanag/Downloads/data.pt')
    if face != None:
        emb = resnet(face.unsqueeze(0)).detach()
        embedding_list = saved_data[0]
        name_list = saved_data[1]
        dist_list = []
        for idx, emb_db in enumerate(embedding_list):
            dist = torch.dist(emb, emb_db).item()
            dist_list.append(dist)
        idx_min = dist_list.index(min(dist_list))

        with open(f'/tmp/{key}.txt', 'w+') as f:
        # with open(f'/Users/adrijanag/Downloads/{key}.txt', 'w+') as f:
            f.write(name_list[idx_min])
        print(f'text file {key}.txt saved to tmp folder')
        return name_list[idx_min]
    else:
        print(f"No face is detected")
    return


def lambda_handler(event, context):
    bucket_name = event['bucket_name']
    image_file_name = event['image_file_name']
    output_bucket = '1225464032-output'

    try:
        print(f"fetching image from stage 1 bucket: {image_file_name}")
        response = s3.get_object(Bucket=bucket_name, Key=image_file_name)
        file_path = f"/tmp/{image_file_name}"
        with open(file_path, 'wb') as f:
            f.write(response['Body'].read())

        print(f'image {image_file_name} stored at {file_path}')
        print(f'calling the face recognition method on {file_path}')
        face_recognition_function(file_path)

        txt_name = image_file_name.split('.')[0]
        s3_key = f'{txt_name}.txt'
        print(f"s3 key: {s3_key}")
        txt_file_path = f'/tmp/{s3_key}'
        s3.upload_file(txt_file_path, output_bucket, s3_key)
        print(f"Uploaded {txt_file_path} to s3://{output_bucket}/{s3_key}")

    except Exception as e:
        print(e)
        print('Error running face-recognition lambda function')
        raise e