#!/usr/bin/env python
from __future__ import print_function
import random
import sys
import rospy
import cv2
import numpy as np
import time
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class image_converter:
    def __init__(self):
        self.image_pub = rospy.Publisher("image_raw_modify",Image, queue_size=1)
        self.bridge = CvBridge()
        self.counter = 0
        self.saved_frame = np.zeros((1080,1920),dtype='uint8')
        self.saved_frame = cv2.cvtColor(self.saved_frame, cv2.COLOR_GRAY2BGR)
        self.image_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.callback)

    def save_img(self,frame):
        img_nr = int(time.time())
        cv2.imwrite("dataset/nut_{}.png".format(img_nr), frame)
        self.counter +=1
        print("nut_"+str(img_nr)+" saved correctly - nr images: "+str(self.counter))

    def callback(self,data):
        try:
            bgr_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        if cv2.waitKey(100) & 0xFF == ord('s'):
            self.save_img(bgr_img)
            self.saved_frame = bgr_img.copy()
            
        image_to_show = np.hstack((bgr_img, self.saved_frame))
        cv2.namedWindow("Current Frame and Previous saved frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Current Frame and Previous saved frame", 1200, 500)
        cv2.imshow("Current Frame and Previous saved frame", image_to_show)
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
