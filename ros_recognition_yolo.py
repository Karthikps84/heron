#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import Float32MultiArray, String, Float32
import pyrealsense2 as rs

import numpy as np
import time
import torch
import os
from torchvision.ops import nms
# Import the necessary modules from MMdetection
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config

# Dictionary for labels
label_map = {
    0: u'__background__', 1: u'person', 2: u'bicycle', 3: u'car', 4: u'motorcycle',
    5: u'airplane', 6: u'bus', 7: u'train', 8: u'truck', 9: u'boat', 10: u'traffic light',
    11: u'fire hydrant', 12: u'stop sign', 13: u'parking meter', 14: u'bench', 15: u'bird',
    16: u'cat', 17: u'dog', 18: u'horse', 19: u'sheep', 20: u'cow', 21: u'elephant',
    22: u'bear', 23: u'zebra', 24: u'giraffe', 25: u'backpack', 26: u'umbrella',
    27: u'handbag', 28: u'tie', 29: u'suitcase', 30: u'frisbee', 31: u'skis', 32: u'snowboard',
    33: u'sports ball', 34: u'kite', 35: u'baseball bat', 36: u'baseball glove', 37: u'skateboard',
    38: u'surfboard', 39: u'tennis racket', 40: u'bottle', 41: u'wine glass', 42: u'cup',
    43: u'fork', 44: u'knife', 45: u'spoon', 46: u'bowl', 47: u'banana', 48: u'apple',
    49: u'sandwich', 50: u'orange', 51: u'broccoli', 52: u'carrot', 53: u'hot dog',
    54: u'pizza', 55: u'donut', 56: u'cake', 57: u'chair', 58: u'couch', 59: u'potted plant',
    60: u'bed', 61: u'dining table', 62: u'toilet', 63: u'tv', 64: u'laptop', 65: u'mouse',
    66: u'remote', 67: u'keyboard', 68: u'cell phone', 69: u'microwave', 70: u'oven',
    71: u'toaster', 72: u'sink', 73: u'refrigerator', 74: u'book', 75: u'clock',
    76: u'vase', 77: u'scissors', 78: u'teddy bear', 79: u'hair drier', 80: u'toothbrush'
}

bridge = CvBridge()

class CameraSubscriber(Node):
    output_dir = 'output_images'
    task_processor = None
    
    def __init__(self):
        super().__init__('camera_subscriber')

        self.imgsz = (640, 480)  # inference size (pixels)
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 10  # maximum detections per image

        self.subscription_color = self.create_subscription(
            Image,
            'rgb_cam/image_raw',
            self.camera_callback_color,
            10)

        self.publisher_bbox = self.create_publisher(Float32MultiArray, 'bounding_box', 10)
        self.publisher_label = self.create_publisher(String, 'Class_name', 10)
        
        if CameraSubscriber.task_processor is None:
            self.init_model()  # Initialize the model only if not already initialized


    def init_model(self):
        self.deploy_cfg = 'mmdeploy/configs/mmdet/detection/detection_onnxruntime_static.py'
        device = 'cuda:0'
        model_cfg = 'rtmdet_tiny_8xb32-300e_coco.py'
        self.backend_model = ['end2end.onnx']
        deploy_cfg, model_cfg = load_config(self.deploy_cfg, model_cfg)
        CameraSubscriber.task_processor = build_task_processor(model_cfg, deploy_cfg, device)
    	
    def camera_callback_color(self, color_data):
            # Process color image here
        color_img = bridge.imgmsg_to_cv2(color_data, "bgr8")
        self.process_images(color_img)

    '''def camera_callback_depth(self, depth_data):
            # Process depth image here
            self.depth_image = bridge.imgmsg_to_cv2(depth_data, desired_encoding='passthrough')'''

    def process_images(self, color_img):
            # Process both color and depth images here
            

            # Resize color image to the model's input size
            input_shape = get_input_shape(self.deploy_cfg)
            model_inputs, _ = CameraSubscriber.task_processor.create_input(color_img, input_shape)
	    #print('inside')
            model = CameraSubscriber.task_processor.build_backend_model(self.backend_model)
            # Inference
            with torch.no_grad():
                result = model.test_step(model_inputs)
            
            nms_result = nms(result[0].pred_instances.bboxes, result[0].pred_instances.scores, 0.3)

            # Process detections
            filtered_boxes = result[0].pred_instances.bboxes[nms_result].tolist()
            filtered_scores = result[0].pred_instances.scores[nms_result].tolist()
            filtered_labels = result[0].pred_instances.labels[nms_result].tolist()

            # visualize results
            CameraSubscriber.task_processor.visualize(
            image=color_img,
            model=model,
            result=result[0],
            window_name='visualize',
            output_file='detection.png')

            for box, label, score in zip(filtered_boxes, filtered_labels, filtered_scores):
                x_min, y_min, x_max, y_max = map(int, box[:4])
                # Get label name from label map
                label_name = label_map.get(label + 1, 'Unknown')

                if score > 0.50:

                    # Publish bounding box information
                    bbox_data = [x_min, y_min, x_max, y_max]  
                    bbox_msg = Float32MultiArray(data=bbox_data)
                    self.publisher_bbox.publish(bbox_msg)

                    # Publish label information
                    string_msg = String()
                    string_msg.data = label_name
                    self.publisher_label.publish(string_msg)


            # Visualize color image with bounding boxes
            for box, label, score in zip(filtered_boxes, filtered_labels, filtered_scores):
                cv2.rectangle(color_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(color_img, f'{label_name}: {score:.2f}', (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #cv2.imwrite('detected_img.png',color_img )
            #cv2.imshow('color_image_with_boxes', color_img)
            #cv2.waitKey(1)

if __name__ == '__main__':
    rclpy.init(args=None)
    camera_subscriber = CameraSubscriber()
    rclpy.spin(camera_subscriber)
    rclpy.shutdown()
