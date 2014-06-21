#!/usr/bin/env python

import rospy
import actionlib
import flir_pantilt_d46.msg
from sensor_msgs.msg import JointState
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
import scitos_ptu_sweep.msg

class PTUSweep():
    # Create feedback and result messages
    _feedback = scitos_ptu_sweep.msg.PTUSweepFeedback()
    _result   = scitos_ptu_sweep.msg.PTUSweepResult()

    def __init__(self, name):
        rospy.loginfo("Starting %s", name)
        self._action_name = name
        rospy.loginfo("Creating action server.")
        self._as = actionlib.SimpleActionServer(self._action_name, scitos_ptu_sweep.msg.PTUSweepAction, execute_cb = self.executeCallback, auto_start = False)
        self._as.register_preempt_callback(self.preemptCallback)
        rospy.loginfo(" ...starting")
        self._as.start()
        rospy.loginfo(" ...done")
        self.arrived = False
        self.published = False
        self.img_published  = False
        self.cancelled = False
        
        #Point Cloud topic to subscribe to
        self.sub_topic = rospy.get_param("~PointCloud", '/head_xtion/depth/points')
        #Point Cloud topic to publish to
        pub_topic = rospy.get_param("~SweepPointCloud", '/ptu_sweep/depth/points')

        #Image topic to subscribe to        
        self.sub_img_topic = rospy.get_param("~Image", '/head_xtion/rgb/image_color')
        #Image topic to publish to
        pub_img_topic = rospy.get_param("~SweepImage", '/ptu_sweep/rgb/image_color')
        
        #Subscribers and publishers
        rospy.Subscriber(self.sub_topic, PointCloud2, self.pointCloudCallback, None, 1)
        rospy.Subscriber(self.sub_img_topic, Image, self.imageCallback, None, 1)
        self.pub = rospy.Publisher(pub_topic, PointCloud2)
        self.pub_img = rospy.Publisher(pub_img_topic, Image)


        self.pan_speed = rospy.get_param("~ptu_pan_speed", 100)
        self.tilt_speed = rospy.get_param("~ptu_tilt_speed", 100)
        self.client = actionlib.SimpleActionClient('SetPTUState', flir_pantilt_d46.msg.PtuGotoAction)
        self.client.wait_for_server()
        rospy.loginfo(" ... Init done")


    def executeCallback(self, goal):
        if not (abs(goal.min_pan)+abs(goal.max_pan))%goal.pan_step == 0 :
            print goal.pan_step%(abs(goal.min_pan)+abs(goal.max_pan))
            rospy.logwarn("The defined pan angle is NOT a multiple of pan step")
        if not (abs(goal.min_tilt)+abs(goal.max_tilt))%goal.tilt_step == 0 :
            print goal.tilt_step%(abs(goal.min_tilt)+abs(goal.max_tilt))
            rospy.logwarn("The defined tilt angle is NOT a multiple of tilt step")
        print "New sweep requested"

        #Subscribers  
        pc2_sub = rospy.Subscriber(self.sub_topic, PointCloud2, self.pointCloudCallback, None, 1)
        img_sub = rospy.Subscriber(self.sub_img_topic, Image, self.imageCallback, None, 1)
        
        self.cancelled = False
        max_pan= 159
        min_pan= -159
        max_tilt = 30
        min_tilt= -46
        if goal.max_pan > max_pan :
            goal.max_pan=max_pan
        if goal.min_pan < min_pan :
            goal.min_pan= min_pan
        if goal.max_tilt > max_tilt :
            goal.max_tilt=max_tilt
        if goal.min_tilt < min_tilt :
            goal.min_tilt= min_tilt
        ptugoal = flir_pantilt_d46.msg.PtuGotoGoal()
        ptugoal.pan_vel = self.pan_speed
        ptugoal.tilt_vel = self.tilt_speed
        current_tilt = goal.min_tilt
        ended=False
        while current_tilt <= goal.max_tilt and not (ended or self.cancelled): 
            current_pan=goal.min_pan
            self._feedback.current_tilt=current_tilt
            while current_pan <= goal.max_pan and not (ended or self.cancelled):
                ptugoal.pan = current_pan
                ptugoal.tilt = current_tilt
                self.client.send_goal(ptugoal)
                self.client.wait_for_result()
                self.arrived = True
                while not self.published and not self.img_published :
                    pass
                self.published = False
                self.img_published = False
                self._feedback.current_pan=current_pan
                current_pan = goal.max_pan if current_pan + goal.pan_step > goal.max_pan and not current_pan >= goal.max_pan else current_pan + goal.pan_step             
                self._as.publish_feedback(self._feedback)
            current_tilt = goal.max_tilt if current_tilt + goal.tilt_step > goal.max_tilt and not current_tilt >= goal.max_tilt else current_tilt + goal.tilt_step
            if current_tilt > goal.max_tilt and current_pan > goal.max_pan:
                ended=True
            else:
                current_pan=goal.max_pan
            self._feedback.current_tilt=current_tilt
            while current_pan >= goal.min_pan and not ended or self.cancelled: 
                ptugoal.pan = current_pan
                ptugoal.tilt = current_tilt
                self.client.send_goal(ptugoal)
                self.client.wait_for_result()
                self.arrived = True
                while not self.published and not self.img_published :
                    pass
                self.published = False
                self.img_published = False
                self._feedback.current_pan=current_pan
                current_pan = goal.min_pan if current_pan - goal.pan_step < goal.min_pan and not current_pan <= goal.min_pan else current_pan - goal.pan_step
                self._as.publish_feedback(self._feedback)
            current_tilt = goal.max_tilt if current_tilt + goal.tilt_step > goal.max_tilt and not current_tilt >= goal.max_tilt else current_tilt + goal.tilt_step
      
        self.resetPTU()

        pc2_sub.unregister()
        img_sub.unregister()
        if not self.cancelled :
            self._result.success = True
            self._as.set_succeeded(self._result)


    def preemptCallback(self):
        self.cancelled = True
        self._result.success = False
        self._as.set_preempted(self._result)

    def resetPTU(self):
        ptugoal = flir_pantilt_d46.msg.PtuGotoGoal()
        ptugoal.pan_vel = 100 
        ptugoal.tilt_vel = 100 
        ptugoal.pan = 0
        ptugoal.tilt = 0
        self.client.send_goal(ptugoal)
        self.client.wait_for_result()

    def pointCloudCallback(self, msg):
        if self.arrived:
            self.pub.publish(msg)
            self.published = True
            self.arrived = False
            
    def imageCallback(self, msg) :
        if self.arrived:
            self.pub_img.publish(msg)
            self.img_published = True
            self.arrived = False       
    

if __name__ == '__main__':
    rospy.init_node("PTUSweep")
    ps = PTUSweep(rospy.get_name())
    rospy.spin()
