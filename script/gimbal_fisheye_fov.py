#!/usr/bin/env python

# Copyright (c) 2017, JSK(University of Tokyo)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Open Source Robotics Foundation, Inc.
#       nor the names of its contributors may be used to endorse or promote
#       products derived from this software without specific prior
#       written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Authors: Moju Zhao
# Maintainer: Moju Zhao <chou@jsk.imi.i.u-tokyo.ac.jp>

import time
import sys
import math
import rospy
import tf
import numpy as np
from numpy.linalg import inv
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo

class gimbal_fisheye_fov:

    def init(self):

        ## ROS Param
        self.__mode = rospy.get_param("~mode", 0)
        self.__height_offset = rospy.get_param("~height_offset", 0.0)
        self.__height_threshold1 = rospy.get_param("~height_threshold1", 1.0)
        self.__height_threshold2 = rospy.get_param("~height_threshold2", 0.0)
        self.__base_resolution = rospy.get_param("~base_resolution", 900)
        self.__base_fov = rospy.get_param("~base_fov", 60.0)
        self.__max_fov = rospy.get_param("~max_fov", 80.0)

        ### Topic Name
        self.__fov_pub_topic_name = rospy.get_param("~fov_pub_topic_name", "max_degree")
        self.__scale_pub_topic_name = rospy.get_param("~scale_pub_topic_name", "image_scale")
        self.__odom_sub_topic_name = rospy.get_param("~odom_sub_topic_name", "/odom")
        self.__raw_camera_info_sub_topic_name = rospy.get_param("~raw_camera_info_topic_name", "/camera/camera_info")

        ## Initial Process
        self.__raw_camera_info = False
        self.__change = False
        self.__raw_image_row = 1
        self.__base_scale = 1.0


        ## Subscriber
        self.__subscriber_odom = rospy.Subscriber(self.__odom_sub_topic_name, Odometry, self.__odom_callback)
        self.__subscriber_raw_camera_info = rospy.Subscriber(self.__raw_camera_info_sub_topic_name, CameraInfo, self.__raw_camera_info_callback)


        ## Publisher
        self.__publisher_fov = rospy.Publisher(self.__fov_pub_topic_name, Float32, queue_size = 1)
        self.__publisher_scale = rospy.Publisher(self.__scale_pub_topic_name, Float32, queue_size = 1)


    # Callback Func
    def __raw_camera_info_callback(self, msg):
        if self.__raw_camera_info:
            return

        self.__raw_image_row = msg.height
        tan_max_radian = math.tan(self.__base_fov * math.pi /180.0);
        self.__base_scale = self.__base_resolution / tan_max_radian / self.__raw_image_row
        rospy.loginfo("scale is %f", self.__base_scale)

        self.__raw_camera_info = True

    def __odom_callback(self, msg):
        distance  = msg.pose.pose.position.z - self.__height_offset

        fov_msg = Float32()
        fov_msg.data = self.__base_fov
        scale_msg = Float32()
        scale_msg.data = self.__base_scale

        # binary distribution
        if self.__mode == 0 and not self.__change:
            if distance < self.__height_threshold1:
                tan_max_radian = math.tan(self.__max_fov * math.pi /180.0);
                scale_msg.data = self.__base_resolution / tan_max_radian / self.__raw_image_row
                fov_msg.data = self.__max_fov

                print "change fov to max"
                self.__change = True

        # linear distribution
        if self.__mode == 1:
            if distance < self.__height_threshold1:
                if distance < self.__height_threshold2:
                    fov_msg.data = self.__max_fov
                    tan_max_radian = math.tan(self.__max_fov * math.pi /180.0);
                    scale_msg.data = self.__base_resolution / tan_max_radian / self.__raw_image_row
                else:
                    fov_msg.data = self.__base_fov + (self.__max_fov - self.__base_fov) * (self.__height_threshold1 - distance) /  (self.__height_threshold1 - self.__height_threshold2)
                    tan_max_radian = math.tan(fov_msg.data * math.pi /180.0);
                    scale_msg.data = self.__base_resolution / tan_max_radian / self.__raw_image_row

        self.__publisher_fov.publish(fov_msg)
        self.__publisher_scale.publish(scale_msg)


if __name__ == '__main__':
    try:
        rospy.init_node('gimbal_fisheye_fov', anonymous = False)
        gimbalFisheyeFov = gimbal_fisheye_fov()
        gimbalFisheyeFov.init()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

