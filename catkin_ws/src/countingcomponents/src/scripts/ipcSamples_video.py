#!/usr/bin/env python
import random, sys, rospy, time, socket
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class record_video:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.callback)

    def callback(self,data):
        try:
            bgr_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        video_writer.write(bgr_img)
        cv2.namedWindow("Stream Video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Stream Video", 1000, 900)
        cv2.imshow("Stream Video",bgr_img)
        cv2.waitKey(1)

def shutting_down():
    video_writer.release()
    print("\nShutdown DONE!")

def main(args):

    global video_writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('pilot_test/stream_video.avi', fourcc , 5.0, (1920, 1080))
    
    rospy.init_node('ipcSamples_video')
    ipc_samples_counting = record_video()

    try:
        rospy.spin()
        rospy.on_shutdown(shutting_down)
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)