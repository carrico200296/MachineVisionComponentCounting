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

        # Load Yolo v3
        self.net = cv2.dnn.readNet("yolo_models/yolov3.weights", "yolo_models/yolov3_testing_nut.cfg")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        print("model loaded")
        self.image_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.callback)

    def save_image(self,frame_detected,frame_original, num_objects):
        img_nr = int(time.time())
        cv2.imwrite("pilot_test/yolov3_416_05_04_{}.png".format(img_nr), frame_original)
        cv2.imwrite("pilot_test/yolov3_416_05_04_{}_predicted.png".format(img_nr), frame_detected)   
        print("nut_{} saved correctly! - nr.objects = {}".format(img_nr, num_objects))

    def detect_object(self,img):
        num_objects = 0 

        classes = ["nut"]
        height, width, channels = img.shape

        print("STEP 1: Processing Image...")
        t0 = time.time()
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        t1 = time.time()
        print("\ttime step 1 = {} s".format(np.round(t1-t0,2)))

        print("STEP 2: Drawing boxes...")
        t2 = time.time()
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                label = "%s:%.2f" % (classes[class_ids[i]], confidences[i])
                color = (0,0,255)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y -10), font, 0.7, color, 1)
                num_objects+=1

        t3 = time.time()
        print("\ttime step 2 = {} s)".format(np.round(t3 - t2,2)))

        fps_label = "FPS: %.2f ; Time: %.2fsec (excluding drawing time of %.2fms)" % (1 / (t1 - t0), (t1 - t0), (t3 - t2) * 1000)
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
