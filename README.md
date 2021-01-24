# MachineVisionComponentCounting
Machine Vision system for component counting for quality control line production.


### IPC component counting pilot test 

* (ssh into the jetson Nano from a remote computer)
* terminal1 (jetson): 
  - $ roslaunch realsense2_camera rs_camera.launch color_width:=1920 color_height:=1080
* terminal2 (jetson): 
  - $ cd catkin_ws/src/countingcomponents/src/scripts/
  - $ rosrun countingcomponents ipcSamples_counting.py
                    
**Run these commands to visualize the predicted images from the remote computer (running on the same ROS MASTER)**        
* terminal1 (remote pc):  
  - $ export ROS_MASTER_URI=192.168.12.242:11311/
  - $ export ROS_IP: 192.168.12.243
  - $ source ~/.bashrc
  - $ rosrun image_view image_view image:=/IPCsamples/detected_img
