 #!/usr/bin/env python
import numpy as np
import rospy

from cartographer_ros_msgs.msg import LandmarkEntry, LandmarkList
from visualization_msgs.msg import Marker, MarkerArray
from parameters import r, T_cam_imu
frame_id = 'imu_frame'

## Landmarks publisher
def landmark_pub(lmk_obsv, lmk_id):
    landmark_list = LandmarkList()
    landmark = LandmarkEntry()

    landmark.id = lmk_id
    pose = np.asarray(lmk_obsv[1]).reshape(3,1)
    timestamp = lmk_obsv[0]
    # rospy.Time.from_sec(int(ts) / 1e9 if integer

    ## Calculate position w.r.t. imu frame
    landmark.tracking_from_landmark_transform.position.x = pose[2] + T_cam_imu[0]
    landmark.tracking_from_landmark_transform.position.y = pose[0]
    z = pose[1] + T_cam_imu[2]
    landmark.tracking_from_landmark_transform.position.z = z if z>0 else 0

    ## Rotation - not used
    landmark.tracking_from_landmark_transform.orientation.x = 0
    landmark.tracking_from_landmark_transform.orientation.y = 0
    landmark.tracking_from_landmark_transform.orientation.z = 0
    landmark.tracking_from_landmark_transform.orientation.w = 1

    landmark.translation_weight = 1.0
    landmark.rotation_weight = 0.0
    landmark_list.landmarks.append(landmark)

    landmark_list.header.stamp = timestamp
    landmark_list.header.frame_id = frame_id
        
    return landmark_list

def convert_landmarklist_to_markerarray(landmarklist: LandmarkList):
    marker_array_msg = MarkerArray()

    for landmark in landmarklist.landmarks:
        marker = Marker()
        marker.header = landmarklist.header
        marker.header.frame_id = "laser"
        marker.ns = "landmarks"
        marker.id = int(landmark.id[landmark.id.index('_')+1:])
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = landmark.tracking_from_landmark_transform.position.x
        marker.pose.position.y = landmark.tracking_from_landmark_transform.position.y
        marker.pose.position.z = landmark.tracking_from_landmark_transform.position.z
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 1
        marker.color.g = 0
        marker.color.b = 0
        marker.color.a = 1
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker_array_msg.markers.append(marker)

    return marker_array_msg