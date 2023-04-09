roscore

#~/2ros/catkin_spencer
rosbag play /home/rpi/2ros/test333.bag --topics /sub_image_pub_press/image_compressed/compressed /velodyne_points /pd/camerainfo /odom /tf /tf_static /led_status /target_waypoints /target_path /hunter_status /velodynemscan -l

~/2ros/catkin_spencer
roslaunch spencer_people_tracking_launch yolo_tracking_on_robot_bag.launch

~/2ros/catkin_all
roslaunch hunter_se_description_2 test_bag.launch

~/2ros/catkin_spencer
rosrun mdetector record_5v1_bag.py 

# 数据的预先处理

文件夹： py_utils， 数据会存储到catkin_data文件夹中
1. 运行 python add_quadrant.py,会把之前存储的数据 pickle_from_bag里的数据转化为pkl_quadrant模式
2. 运行 python transfer_pickle_to_txt.py，会把pkl_quadrant转化为pkl_txt模式
3. 运行split_by_frame,需要提前把 catkin_data/pkl_txt里面的文件拷贝到py_utils/split/origindata里面，它会把数据生成到pages里面

scp test360.bag rpi@192.168.2.106:/home/rpi/hunter_data
