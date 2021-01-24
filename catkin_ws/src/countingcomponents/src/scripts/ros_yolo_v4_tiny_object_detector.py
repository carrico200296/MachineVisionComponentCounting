#!/usr/bin/env python
from __future__ import print_function
import random
import sys
import rospy
import cv2
import numpy as np
import random 
import time
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class image_converter:
    def __init__(self):
        self.image_pub = rospy.Publisher("image_object_detector",Image, queue_size=1)
        self.bridge = CvBridge()
        self.detected_img = np.zeros((1080,1920),dtype='uint8')
        self.detected_img = cv2.cvtColor(self.detected_img, cv2.COLOR_GRAY2BGR)
        # Load Yolo Tiny v4
        self.model_weights = "yolo_models/yolov4-tiny.weights"
        self.model_cfg = "yolo_models/yolov4-tiny_testing_nut.cfg"
        self.net = cv2.dnn_DetectionModel(self.model_cfg, self.model_weights)
        self.net.setInputSize(608,608)
        self.net.setInputScale(1.0/255)
        self.net.setInputSwapRB(True)
        self.CONFIDENCE_THRESHOLD = 0.7
        self.NMS_THRESHOLD = 0.4
        print("STEP 0: Model loaded")
        self.image_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.callback)

    def save_image(self,frame_detected,frame_original,num_objects):
        img_nr = int(time.time())
        cv2.imwrite("pilot_test/yolov4_tiny_608_07_04_{}.png".format(img_nr), frame_original)
        cv2.imwrite("pilot_test/yolov4_tiny_608_07_04_{}_predicted.png".format(img_nr), frame_detected)    
        print("nut_{} saved correctly!".format(img_nr))
        print("------------------------------------------")

    def detect_object(self, img):

        class_names = ["NUT"]
        height_original, width_original, channels_original = img.shape
        
        print("STEP 1: Processing Image...")
        start = time.time()
        classes, scores, boxes = self.net.detect(img, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
        end = time.time()
        print("\ttime step 1 = {} s".format(np.round(end-start,2)))
        num_objects = len(boxes)

        print("STEP 2: Drawing boxes...")
        font = cv2.FONT_HERSHEY_DUPLEX
        color = (0,0,255)
        start_drawing = time.time()
        for (classid, score, box) in zip(classes, scores, boxes):
            label = "%s:%.2f" % (class_names[classid[0]], score)
            cv2.rectangle(img, box, color, 2)
            cv2.putText(img, label, (box[0], box[1] - 10), font, 0.7, color, 1)
        end_drawing = time.time()
        print("\ttime step 2 = {} s)".format(np.round(end_drawing - start_drawing,2)))
        
        fps_label = "FPS: %.2f ; Time: %.2fsec (excluding drawing time of %.2fms)" % (1 / (end - start), (end - start), (end_drawing - start_drawing) * 1000)
        cv2.putText(img, fps_label, (15, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, 'Number of Components', (15,100), font, 1.5, (255,0,0), 2)
        cv2.putText(img, 'NUT = ' + str(num_objects), (15,170), font, 2, color, 2) 
        print(fps_label)
        print("Number of components detected = {}".format(num_objects))
        
        return img, num_objects

    def callback(self,data):
        try:
            bgr_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        img_to_detect = bgr_img.copy()
        num_objects = 0

        if cv2.waitKey(100) & 0xFF == ord('s'):
            time_init = time.time()
            self.detected_img, num_objects = self.detect_object(img_to_detect)
            self.save_image(self.detected_img, bgr_img, num_objects)
    
        cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Object Detection", 900, 600)
        cv2.imshow("Object Detection", self.detected_img)
        cv2.waitKey(1)

        cv2.namedWindow("Current Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Current Frame", 900, 600)
        cv2.imshow("Current Frame", bgr_img)        
        cv2.waitKey(1)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(bgr_img, "bgr8"))
        except CvBridgeError as e:
            print(e)

def main(args):
    rospy.init_node('image_components')
    ic = image_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)