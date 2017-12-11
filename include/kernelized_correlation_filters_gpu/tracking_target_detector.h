
#pragma once
#ifndef _TRACKING_TARGET_DETECTOR_H_
#define _TRACKING_TARGET_DETECTOR_H_

#include <ros/ros.h>
#include <ros/console.h>

#include <sensor_msgs/Image.h>
// #include <nav_msgs/Odometry.h>
#include <kernelized_correlation_filters_gpu/Detector.h>
#include <boost/foreach.hpp>
#include <opencv2/opencv.hpp>

class TrackingTargetDetector {

 private:
    sensor_msgs::ImageConstPtr image_msg_;
    nav_msgs::Odometry::ConstPtr odom_msg_;
    bool has_odom_;
   
    bool detectorSRV(cv::Rect_<int> &, const sensor_msgs::ImageConstPtr &);
    bool detectorSRV(cv::Rect_<int> &, const sensor_msgs::ImageConstPtr &,
                     const nav_msgs::Odometry::ConstPtr &);
   
 protected:
    ros::NodeHandle nh_;
    ros::ServiceClient srv_client_;
   
 public:
    TrackingTargetDetector();
    bool detect(cv::Rect_<int> &);
    bool detect(cv::Rect_<int> &, const cv::Mat);
    bool detect(cv::Rect_<int> &, const sensor_msgs::ImageConstPtr&);
    void setImageMsg(const sensor_msgs::ImageConstPtr &);
    void setROSMsg(const sensor_msgs::ImageConstPtr &,
                   const nav_msgs::Odometry::ConstPtr &);
};



#endif /* _TRACKING_TARGET_DETECTOR_H_ */
