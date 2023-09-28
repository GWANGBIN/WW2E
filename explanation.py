from collections import deque
from multiprocessing.pool import ThreadPool
import pandas as pd
import cv2 as cv
import glob
import argparse
import cv2
import numpy as np
import models
import torch
import torch.nn.functional as F
import time
import datetime
import tensorflow as tf
import csv
import base64
import os
import shutil
import time
import zmq
from scipy import ndimage
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession

# Configurations
VIDEO_SOURCE = 'recorded_driving1.mp4'
frameWidth = 1200
frameHeight = 450

count = 0
fpsTotal = 0
prevTimeStamp = time.time()
currTimeStamp = time.time()

frameWidth = 1200
frameHeight = 450

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
print("[INFO]  ZeroMQ Active. Listening...")

target_depth = 8
target_width = 128
target_height = 128

participantID = 1
condition = 1

visFlag = 0
attenFlag = 0
riskFlag = 0

depth_factor = target_depth / 100
width_factor = target_width / (frameWidth/2)
height_factor = target_height / frameHeight

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

color_map = [
    (32, 142, 111),   
    (32, 142, 111),   
    (32, 142, 111),   
    (32, 142, 111),   
    (32, 142, 111),    
    (32, 142, 111),   
    (32, 142, 111), 
    (32, 142, 111),    
    (32, 142, 111),    
    (32, 142, 111),   
    (32, 142, 111),    
    (255, 255, 0),     
    (32, 142, 111),    
    (255, 255, 0),    
    (255, 255, 0),    
    (255, 255, 0),    
    (32, 142, 111),  
    (255, 255, 0),   
    (255, 255, 0)      
]

blank_image1 = np.zeros((320, 480, 3), np.uint8)
blank_image1[:] = (144, 0, 0)
empty_without_attention = blank_image1

blank_image2 = np.zeros((320, 480, 3), np.uint8)
blank_image2[:] = (255, 142, 32)
empty_for_pause = blank_image2
_, np2bgd = cv2.imencode('.jpg', empty_for_pause*0.5)

blank_image_grad1 = np.zeros((320, 120), np.uint8)
blank_image_grad2 = np.zeros((320, 120), np.uint8)

def configure_tf():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    session = tf.compat.v1.InteractiveSession(config=config)

def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')
    parser.add_argument('--a', default='pidnet-s', type=str)
    parser.add_argument('--c', default=True, type=bool)
    parser.add_argument('--p', default='pretrained_models/cityscapes/PIDNet_S_Cityscapes_test.pt', type=str)
    parser.add_argument('--r', default='samples/', type=str)
    parser.add_argument('--t', default='.png', type=str)
    return parser.parse_args()

def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cuda:0')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if
                       (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    return model

def segmentation_input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]
    return image



def process_segmentation(frame):
    temp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    temp = cv2.resize(temp, (480, 320))
    sv_img = np.zeros_like(temp).astype(np.uint8)
    temp = segmentation_input_transform(temp)
    temp = temp.transpose((2, 0, 1)).copy()
    temp = torch.from_numpy(temp).unsqueeze(0).cuda()
    pred = segmentation_model(temp)
    pred = F.interpolate(pred, size=temp.size()[-2:], mode='bilinear', align_corners=True)
    pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
    for i, color in enumerate(color_map):
        sv_img[pred == i] = color
    return sv_img

def process_risk_prediction(frame, var_image_3d):
    gray_buff = np.empty((1, frameHeight, int(frameWidth/2)), np.dtype('uint8'))
    buff = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    buff = buff[:,300:900]
    gray_buff[0] = buff
    gray = ndimage.zoom(gray_buff, (1, height_factor, width_factor), order=1)
    gray = gray.astype(np.float32)
    gray /= 255.
    image_3d = shifting_each_frame(var_image_3d, gray[0])
    risk_input_image = ndimage.zoom(image_3d, (depth_factor, 1, 1), order=1)
    prediction = risk_prediction_model(np.expand_dims(risk_input_image, axis=0), training=False)[0]
    score = prediction[0]
    return [score, risk_input_image]

def save_gramdcam(heatmapshow, gradcampath):
    unix_time = int(time.time())
    filename = os.path.join(gradcampath, f"{unix_time}.jpg")
    cv2.imwrite(filename, heatmapshow)
    return unix_time

if __name__ == '__main__':
    
    # Video codec and writer setup
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(outputvideopath+'/threaded_record.avi', fourcc, 30, (480, 320))
    
    # Participant input and directory setup
    participantID = input("Participant name: ")
    condition = int(input("experimental condition: "))

    # Directory setup for storing data and results
    parentpath = os.path.join('logged_data', str(participantID))
    xdirpath = os.path.join(parentpath, str(condition))
    gradcampath = os.path.join(dirpath, 'gradcam')
    outputvideopath = os.path.join(dirpath, 'outputvideo')

    # Ensure base directory exists
    if not os.path.exists('logged_data'):
        os.mkdir('logged_data')

    # Ensure participant directory exists
    if not os.path.exists(parentpath):
        os.mkdir(parentpath)

    # If condition directory exists, remove it to avoid clashes
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath, ignore_errors=True)

    # Create directories for the current session
    os.mkdir(dirpath)
    os.mkdir(gradcampath)
    os.mkdir(outputvideopath)   

    # Set conditions based on user input
    visFlag, attenFlag, riskFlag = 0, 0, 0
    if condition == 2:
        visFlag = 1
    elif condition == 3:
        visFlag, attenFlag = 1, 1
    elif condition == 4:
        visFlag, riskFlag = 1, 1
    elif condition == 5:
        attenFlag, riskFlag = 1, 1
    elif condition == 6:
        visFlag, attenFlag, riskFlag = 1, 1, 1

    # Thread setup for parallel processing
    thread_num = cv.getNumberOfCPUs()
    pool = ThreadPool(processes=thread_num)
    pool2 = ThreadPool(processes=thread_num)
    pending_task = deque()
    pending_task2 = deque()

    with torch.no_grad():
        last_time = 0
        while True:
            incoming = socket.recv()
            sendStr = ""

            if (len(pending_task) > 0 and pending_task[0].ready()) and (len(pending_task2) > 0 and pending_task2[0].ready()):
                segmap = pending_task.popleft().get()
                risk_list = pending_task2.popleft().get()
                risk_probability = risk_list[0]
                risk_input_image = risk_list[1]

                # GRAD-CAM processing
                #... [Your Grad-CAM processing code]

                # GRAD-CAM processing for risk analysis visualization
                grad_input = tf.expand_dims(risk_input_image, axis=-1)
                grad_input = tf.expand_dims(risk_input_image, axis=0)
                with tf.GradientTape() as tape:
                    conv_outputs, grad_predictions = grad_model(grad_input)
                    grad_loss = grad_predictions[:, class_index]

                grad_output = conv_outputs[0]
                grads = tape.gradient(grad_loss, conv_outputs)[0]
                weights = tf.reduce_mean(grads, axis=(0, 1, 2))

                cam = np.zeros(grad_output.shape[0:3], dtype=np.float32)
                for index, w in enumerate(weights):
                    cam += w * grad_output[:, :, :, index]

                # Rescaling the CAM to increase its visibility
                capi = ndimage.zoom(cam, (1, 4, 4), order=1)
                capi = np.maximum(capi, 0)
                heatmap = (capi - capi.min()) / (capi.max() - capi.min())
                heatmap = np.squeeze(heatmap[0, :, :])

                # Normalize and resize for visualization
                heatmapshow = None
                heatmapshow = cv2.normalize(heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                heatmapshow[heatmapshow < 113] = 0
                heatmapshow = cv2.resize(heatmapshow, (240, 320))
                heatmapshow = cv2.hconcat([blank_image_grad1, heatmapshow, blank_image_grad2])
                heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)


                # Update display based on conditions and risk
                if riskFlag and (risk_probability > 0.5):
                    coolTime = 1.0
                    print("Risky! CoolTime renewed to 5.0! Risk probability:"+str(risk_probability))

                # Composite frame with segment map and heatmap
                segmap = cv2.cvtColor(segmap, cv2.COLOR_BGR2RGB)
                sv_img = cv2.addWeighted(segmap, visFlag * 0.5, heatmapshow*attenFlag+(1-attenFlag)*empty_without_attention, 0.5, 0)
                _, np2img = cv2.imencode('.jpg', sv_img)

                # Frame rate computation
                currTimeStamp = time.time()
                deltaTime = currTimeStamp - prevTimeStamp
                prevTimeStamp = currTimeStamp
                fps = 1 / deltaTime

                # Display conditions
                if condition == 1:
                    sendStr = "HIDE"
                elif coolTime > 0:
                    sendStr = base64.b64encode(np2img)
                    coolTime -= deltaTime
                else:
                    sendStr = base64.b64encode(np2bgd)

                framerate = 1 / (time.perf_counter() - last_time)
                out.write(sv_img)
                last_time = time.perf_counter()
                print(framerate)

            # Fetch next frame
            if incoming and (len(pending_task) < thread_num) and (len(pending_task2) < thread_num):    
                imgBuffer1 = base64.b64decode(incoming)
                imgBuffer2 = np.frombuffer(imgBuffer1, dtype=np.uint8)
                frame = cv2.imdecode(imgBuffer2, flags=cv2.IMREAD_COLOR)    
                task = pool.apply_async(process_segmentation, (frame.copy(),))
                pending_task.append(task)
                risk_list = pool2.apply_async(process_risk_prediction, (frame.copy(), var_image_3d))
                pending_task2.append(risk_list)

            socket.send_string(str(sendStr))

    out.release()
    cv.destroyAllWindows()