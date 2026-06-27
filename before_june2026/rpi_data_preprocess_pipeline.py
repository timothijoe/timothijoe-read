1. bag输出成pkl文件
总体上来说，需要开5个terminal, 分别操作若干指令
(gd means guangdu)

gd1
gd2  && ls && myplay test331.bag
gd3
gd4
gd5

输入以上五个指令，即可对采集的bag进行操作pickle文件的提取
可以打开两个文件夹
cd /home/rpi/zt3_data_dealing/hunter_bag &&ls   用于观察需要播放的hunter_bag有哪些
cd /home/rpi/zt3_data_dealing/hunter_data/pickle_from_bag 存储的bag会存储到这个目录下，可以观察新生成的文件有哪些

上述五个指令的含义为：
alias gd1='roscore'
alias gd2='cd ~/2ros/catkin_spencer && source devel/setup.bash && cd /home/rpi/zt3_data_dealing/hunter_bag'
alias gd3='cd ~/2ros/catkin_spencer && source devel/setup.bash && roslaunch spencer_people_tracking_launch yolo_tracking_on_robot_bag.launch'
alias gd4='cd ~/2ros/catkin_all && source devel/setup.bash && roslaunch hunter_se_description_2 test_bag.launch'
alias gd5='cd ~/2ros/catkin_spencer && source devel/setup.bash && rosrun mdetector record_5v1_bag.py'
alias myplay='function _play(){ rosbag play $1 --topics /sub_image_pub_press/image_compressed/compressed /velodyne_points /pd/camerainfo /odom /tf /tf_static /led_status /target_waypoints /target_path /hunter_status /velodynemscan;};_play'
