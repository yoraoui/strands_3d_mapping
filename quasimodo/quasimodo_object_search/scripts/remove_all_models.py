#!/usr/bin/env python

import roslib
import rospy
from sensor_msgs.msg import PointCloud2, PointField
#from world_modeling.srv import *
from cv_bridge import CvBridge, CvBridgeError
import cv2

# SOMA2 stuff
from soma_msgs.msg import SOMAObject
from mongodb_store.message_store import MessageStoreProxy
from soma_manager.srv import *
from geometry_msgs.msg import Pose, Transform
from observation_registration_services.srv import ProcessRegisteredViews, ProcessRegisteredViewsRequest, ProcessRegisteredViewsResponse
from soma_io.state import World, Object
from datetime import datetime, timedelta
from quasimodo_msgs.msg import fused_world_state_object, image_array
from quasimodo_msgs.srv import transform_cloud, transform_cloudRequest, transform_cloudResponse
from quasimodo_msgs.srv import index_cloud, index_cloudRequest, index_cloudResponse
from quasimodo_msgs.srv import mask_pointclouds, mask_pointcloudsRequest, mask_pointcloudsResponse
from quasimodo_msgs.srv import insert_model, insert_modelRequest, insert_modelResponse
from geometry_msgs.msg import Pose
from std_msgs.msg import Empty
import time
import json

pub = ()

def remove_models():

    msg_store = MessageStoreProxy(database='world_state', collection='quasimodo')
    # msg_store.delete(req.object_id)
    # resp = insert_modelResponse()
    # resp.object_id = req.object_id

    results = msg_store.query(type='quasimodo_msgs/fused_world_state_object')

    for message, metadata in results:
        if len(message.removed_at) == 0:
            message.removed_at = "manually removed"
            msg_store.update_id(metadata['_id'], message)

    empty_msg = Empty()
    pub.publish(empty_msg)


if __name__ == '__main__':
    rospy.init_node('remove_all_models', anonymous = False)

    pub = rospy.Publisher("/model/added_to_db", data_class=Empty, queue_size=None)

    remove_models()
