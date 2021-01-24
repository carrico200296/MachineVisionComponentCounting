#!/usr/bin/env python
import random, sys, rospy, time, socket
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class counting_IPCsamples:
    def __init__(self):
        self.pub_IPCsamples_image = rospy.Publisher("/IPCsamples/detected_img",Image, queue_size=1)
        self.pub_IPCsamples_components = rospy.Publisher("/IPCsamples/num_IPCsamples",String, queue_size=1)
        self.bridge = CvBridge()
        self.detection = True
        self.time_init = time.time()
        self.num_objects = 0
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
        print("Ready to count components!")
        print("------------------------------------------")
        self.image_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.callback)

    def count_IPCsamples_DL(self, img):

        class_names = ["NUT"]
        color = (0,0,255)
        font = cv2.FONT_HERSHEY_DUPLEX

        print("------------------------------------------")
        # Detecting objects
        start = time.time()
        classes, scores, boxes = self.net.detect(img, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
        end = time.time()
        print("STEP 1: Processing Image = {} sec".format(np.round(end-start,2)) )
        self.num_objects = len(boxes)

        # Drawing bounding boxes
        start_drawing = time.time()
        for (classid, score, box) in zip(classes, scores, boxes):
            label = "%s:%.2f" % (class_names[classid[0]], score)
            cv2.rectangle(img, box, color, 2)
            cv2.putText(img, label, (box[0], box[1] - 10), font, 0.7, color, 1)
        end_drawing = time.time()
        print("STEP 2: Drawing boxes = {} ms".format(np.round(end_drawing - start_drawing,2)*1000))

        fps_info = "FPS: %.2f ; Time: %.2fsec (excluding drawing time of %.2fms)" % (1 / (end - start), (end - start), (end_drawing - start_drawing) * 1000)
        cv2.putText(img, fps_info, (15, 25), font, 1, (0, 0, 0), 2)
        cv2.putText(img, 'Number of Components', (15,100), font, 1.5, (255,0,0), 2)
        cv2.putText(img, 'NUT = ' + str(self.num_objects), (15,170), font, 2, color, 2) 
        print(fps_info)
        print("Number of components detected = {}".format(self.num_objects))
        print("------------------------------------------")
        return img, self.num_objects

    def callback(self,data):
        try:
            bgr_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = bgr_img.copy()

        print(self.detection)
        if self.detection == True:
            self.time_init = time.time()
            cv2.imwrite("pilot_test/pilot_ipc_v4_tiny_{}.png".format(self.time_init), raw_img)
            self.detected_img, self.num_objects = self.count_IPCsamples_DL(raw_img)
            cv2.imwrite("pilot_test/pilot_ipc_v4_tiny_{}_predicted.png".format(self.time_init), self.detected_img)
            self.detection = False
            exit()
            
        print(self.detection)
        video_writer.write(bgr_img)

        try:
            self.pub_IPCsamples_image.publish(self.bridge.cv2_to_imgmsg(self.detected_img, "bgr8"))
            self.pub_IPCsamples_components.publish(str(self.num_objects))
        except CvBridgeError as e:
            print(e)

def inform_ur(data):
        if int(data) > 14:
            number_IPCsamples = str(data)
            rospy.loginfo("Number of IPCsamples = %s", number_IPCsamples)
            number_IPCsamples = number_IPCsamples + "\n"
            clientsocket.send(number_IPCsamples)
            clientsocket.close()

def callback_components(msg):
    inform_ur(msg.data)

def shutting_down():
    s.close()
    clientsocket.close()
    video_writer.release()
    print("\nShutdown DONE!")

def main(args):

    global video_writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('pilot_test/pilot_ipc_video.avi', fourcc , 5.0, (1920, 1080))
    
    rospy.init_node('ipcSamples_counting')

    # Create the socket
    global s
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((socket.gethostname(), 50000))
    s.listen(5)
    print("Server waiting for a client...")
    
    while True:
        global clientsocket
        clientsocket, address = s.accept()
        print("Connection from {} has been established!".format(address))

        ip_urRobot, port_urRobot = address  
        if ip_urRobot == '192.168.12.245':
            print("Component Counting Task requested!\n")
            ipc_samples_counting = counting_IPCsamples()
            sub_IPCsamples_components = rospy.Subscriber("/IPCsamples/num_IPCsamples",String,callback_components)
        try:
            rospy.spin()
            rospy.on_shutdown(shutting_down)
        except KeyboardInterrupt:
            print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)