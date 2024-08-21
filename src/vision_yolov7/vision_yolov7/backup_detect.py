"""
Landmark Detection and Distance Calculation for Humanoid Soccer Robot

This script is part of a research project in Computer Vision and Robotics, developed during a Scientific Initiation.
It is designed to detect specific landmarks on a humanoid soccer field using a YOLOv7 model, and calculate the distance 
from the robot to these landmarks based on the robot's neck position and camera orientation.

Main functionalities include:
- Capturing real-time video feed from the robot's camera.
- Detecting landmarks such as the center line, penalty area, and goalposts.
- Publishing the detected positions using ROS2 topics.
- Calculating the distance to the landmarks using trigonometric functions based on the robot's neck position.
- Visualizing detected landmarks with bounding boxes on the video feed.

Developed by: Massiel Blandy Ramón
Institution: Centro Universitário FEI
2024
"""

import rclpy
from rclpy.node import Node
import math
from custom_interfaces.msg import VisionVector, VisionVector1, VisionVector2, NeckPosition
import sys
import threading
sys.path.insert(0, './src/vision_yolov7/vision_yolov7')
import numpy as np
from numpy import random
import cv2
import torch
from utils.general import check_img_size, non_max_suppression, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device
from models.experimental import attempt_load
from .ClassConfig import *

#Limite para a confiança da detecção
THRESHOLD = 0.45

class LandmarkDetection(Node):
    def __init__(self, config):
        super().__init__('landmark_detection')
        self.config = config
        self.publisher_centerlandmark = self.create_publisher(VisionVector, '/centerlandmark_position', 10)
        self.publisher_penaltilandmark = self.create_publisher(VisionVector1, '/penaltilandmark_position', 10)
        self.publisher_goalpostlandmark = self.create_publisher(VisionVector2, '/goalpostlandmark_position', 10)
        self.weights = 'src/vision_yolov7/vision_yolov7/peso_tiny/best_localization.pt'
        self.neck_subscription = self.create_subscription(NeckPosition, '/neck_position', self.topic_callback_neck, 10)
        self.neck_sides = None
        self.neck_up = None
        self.distance = None
        self.angle = None
        self.y = None
        self.x = None
        self.camera_height = 0.05
        self.robot_height = 0.59
        self.device = select_device('cpu')
        self.model, self.stride, self.imgsz, self.names, self.colors = self.load_model()
        self.cap = cv2.VideoCapture('/dev/video0')
        self.processing_thread = threading.Thread(target=self.detect_landmarks)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.iteration_count = 0  # Contador de iterações para controlar a frequência dos logs
        self.neck_update_count = 0  # Contador para controlar a frequência dos logs na função callback

    def load_model(self):
        model = attempt_load(self.weights, map_location=self.device)
        stride = int(model.stride.max())
        imgsz = check_img_size(640, s=stride)
        names = model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        return model, stride, imgsz, names, colors

    def topic_callback_neck(self, msg):
        self.neck_sides = msg.position19
        self.neck_up = msg.position20
        self.neck_update_count += 1

        if self.neck_update_count % 50 == 0:  # Logar a cada 50 atualizações de posição do pescoço
            self.get_logger().info(f"Callback - Neck Position: Sides {self.neck_sides}, Up {self.neck_up}")

    def detect_landmarks(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().error("Failed to capture image")
                continue

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).to(self.device).float() / 255.0
            if img.ndimension() == 3:
                img = img.permute(2, 0, 1).unsqueeze(0)

            pred = self.model(img)[0]
            pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=True)
            im0 = frame.copy()

            msg_centerlandmark = VisionVector()
            msg_penaltilandmark = VisionVector1()
            msg_goalpostlandmark = VisionVector2()
            msg_centerlandmark.detected = msg_penaltilandmark.detected = msg_goalpostlandmark.detected = False
            msg_centerlandmark.left = msg_penaltilandmark.left = msg_goalpostlandmark.left = False
            msg_centerlandmark.center_left = msg_penaltilandmark.center_left = msg_goalpostlandmark.center_left = False
            msg_centerlandmark.center_right = msg_penaltilandmark.center_right = msg_goalpostlandmark.center_right = False
            msg_centerlandmark.right = msg_penaltilandmark.right = msg_goalpostlandmark.right = False
            msg_centerlandmark.med = msg_penaltilandmark.med = msg_goalpostlandmark.med = False
            msg_centerlandmark.far = msg_penaltilandmark.far = msg_goalpostlandmark.far = False
            msg_centerlandmark.close = msg_penaltilandmark.close = msg_goalpostlandmark.close = False

            if pred[0] is not None:
                for *xyxy, conf, cls in reversed(pred[0]):
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    if conf > THRESHOLD:
                        c1_center = (xyxy[0] + xyxy[2]) / 2
                        c2_center = (xyxy[1] + xyxy[3]) / 2
                        if self.names[int(cls)] in ("center", "penalti", "goalpost"):
                            msg_landmark = None
                            if self.names[int(cls)] == "center":
                                msg_landmark = msg_centerlandmark
                            elif self.names[int(cls)] == "penalti":
                                msg_landmark = msg_penaltilandmark
                            else:
                                msg_landmark = msg_goalpostlandmark
                            if msg_landmark is not None:
                                msg_landmark.detected = True
                                x_pos = "left" if int(c1_center) <= self.config.x_left else \
                                        "center_left" if int(c1_center) < self.config.x_center else \
                                        "center_right" if int(c1_center) > self.config.x_center and int(c1_center) < self.config.x_right else \
                                        "right"
                                y_pos = "far" if int(c2_center) <= self.config.y_longe else \
                                        "close" if int(c2_center) >= self.config.y_chute else \
                                        "med"
                                setattr(msg_landmark, x_pos, True)
                                setattr(msg_landmark, y_pos, True)
                                getattr(self, f"publisher_{self.names[int(cls)]}landmark").publish(msg_landmark)

                                if (x_pos == "center_right" or x_pos == "center_left") and y_pos == "med":
                                    self.angle = ((self.neck_up - 1024) * 90) / 1024
                                    self.angle_rad = math.radians(self.angle)
                                    self.y = self.camera_height * math.sin(self.angle_rad)
                                    self.x = self.camera_height * math.cos(self.angle_rad)
                                    self.total_height = self.robot_height + self.y
                                    self.distance = math.tan(self.angle_rad) * self.total_height + self.x
                                    if self.iteration_count % 50 == 0:  # Logar a cada 50 iterações
                                        self.get_logger().info(f"Distância entre robô e landmark {self.names[int(cls)]}: {self.distance}")

                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=2)

            self.iteration_count += 1

            cv2.line(im0, (self.config.x_left, 0), (self.config.x_left, im0.shape[0]), (0, 0, 255), 1)
            cv2.line(im0, (self.config.x_center, 0), (self.config.x_center, im0.shape[0]), (0, 0, 255), 1)
            cv2.line(im0, (self.config.x_right, 0), (self.config.x_right, im0.shape[0]), (0, 0, 255), 1)

            cv2.line(im0, (0, self.config.y_chute), (im0.shape[1], self.config.y_chute), (0, 0, 255), 1)
            cv2.line(im0, (0, self.config.y_longe), (im0.shape[1], self.config.y_longe), (0, 0, 255), 1)

            cv2.imshow('Landmark Detection', im0)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)  #Inicializando o ROS2
    config = classConfig()
    landmark_detection = LandmarkDetection(config)
    rclpy.spin(landmark_detection)
    landmark_detection.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()