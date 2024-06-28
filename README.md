# landmark based vSLAM implementation

to start, read the instructions at https://github.com/anastasiapan/Landmarks_vSLAM

This system was tested using python3.8 on an ubuntu focal 20.04 machine with ROS noetic 
Pick any of the three versions.
cd into the directory and the system can be run by:
YOLOv5v1: $ python3 detect_landmarks_vslam.py --weights 'path/to/weights' --conf-thres CONFIDENCE_THRESHOLD --iou-thres IOU_THRESHOLD
YOLOv5v7: $ python3 detect_landmarks.py --weights 'path/to/weights' --conf-thres CONFIDENCE_THRESHOLD --iou-thres IOU_THRESHOLD
YOLOv9  : $ python3 detect_dual_landmarks.py --weights 'path/to/weights' --conf-thres CONFIDENCE_THRESHOLD --iou-thres IOU_THRESHOLD
