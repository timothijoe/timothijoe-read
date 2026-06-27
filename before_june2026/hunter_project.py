配置hunter时期的链接，文件等的呢过

1. 配置tensorrt
https://www.cnblogs.com/qsm2020/p/16311487.html


2. 配置yoloface tensorrt
https://github.com/we0091234/yolov7-face-tensorrt

3. https://zhuanlan.zhihu.com/p/344810135

需要更改的模型路径：
src/detect_yolo_RT.cpp:6:std::string mConfiguration = path_src + "/configs/yolov5-6.0/yolov5n.cfg";
src/detect_yolo_RT.cpp:7:std::string mWeights = path_src + "/configs/yolov5-6.0/yolov5n.weights";

需要添加的模型：
tensorrt生成的yolov5-4.0 文件夹复制到 mdetector/configs里面
tensorrt face需要复制到trt_model里面，注意下划线和-的区别