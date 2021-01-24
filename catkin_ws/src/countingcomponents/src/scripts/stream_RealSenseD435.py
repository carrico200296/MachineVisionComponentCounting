#!/usr/bin/env python
from __future__ import print_function
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class image_converter:
   def __init__(self):
      self.image_pub = rospy.Publisher("image_raw_modify",Image, queue_size=1)
      self.bridge = CvBridge()
      self.image_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.callback)

   def callback(self,data):
      try:
         bgr_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
      except CvBridgeError as e:
         print(e)

      (rows,cols,channels) = bgr_img.shape
      if width_original > 60 and height_original > 60 :
         cv2.circle(bgr_img, (50,50), 10, 255) 

      cv2.imshow("Raw image", bgr_img)
      cv2.waitKey(1)
      
      try:
         self.image_pub.publish(self.bridge.cv2_to_imgmsg(bgr_img, "bgr8"))
      except CvBridgeError as e:
         print(e)

def main(args):

   rospy.init_node('image_components', disable_signals=True)
   ic = image_converter()
   try:
      rospy.spin()
   except KeyboardInterrupt:
      print("Shutting down")
      cv2.destroyAllWindows()

if __name__ == '__main__':
   main(sys.argv)