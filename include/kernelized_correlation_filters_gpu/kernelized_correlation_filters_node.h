// Copyright (C) 2016 by Krishneel Chaudhary @ JSK Lab,
// The University of Tokyo, Japan
// krishneel@jsk.imi.i.u-tokyo.ac.jp

#pragma once
#ifndef _KERNELIZED_CORRELATION_FILTERS_NODE_H_
#define _KERNELIZED_CORRELATION_FILTERS_NODE_H_

#include <ros/ros.h>
#include <ros/console.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Odometry.h>
#include <kernelized_correlation_filters_gpu/kernelized_correlation_filters.h>

class KCFTargetTracking {
   
 private:
    typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::Image, nav_msgs::Odometry> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::Image> msub_image_;
    message_filters::Subscriber<nav_msgs::Odometry> msub_odom_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

    typedef message_filters::sync_policies::ApproximateTime<
       sensor_msgs::Image, geometry_msgs::PolygonStamped,
       nav_msgs::Odometry> InitPolicy;
    message_filters::Subscriber<sensor_msgs::Image> init_image_;
    message_filters::Subscriber<geometry_msgs::PolygonStamped> init_rect_;
    message_filters::Subscriber<nav_msgs::Odometry> init_odom_;
    boost::shared_ptr<message_filters::Synchronizer<InitPolicy> >init_sync_;
   
    cv::Rect_<int> screen_rect_;
    bool tracker_init_;
    int width_;
    int height_;
    int block_size_;
    int downsize_;
    int rectangle_resizing_scale_;
   
    bool headless_;
    bool init_via_detector_;
    bool without_uav_;
    float init_altitude_;  //! width(pixels) to uav height ratio
    float pixel_lenght_;
    float prev_scale_;
   
 protected:
    void onInit();
    void subscribe();
    void unsubscribe();

    boost::mutex lock_;
    ros::NodeHandle nh_, pnh_;
    ros::Subscriber sub_image_;
    ros::Subscriber sub_screen_pt_;
    ros::Publisher pub_image_;
    ros::Publisher pub_position_;
    int resize_factor_;

    boost::shared_ptr<KernelizedCorrelationFiltersGPU> tracker_;
   
 public:
    KCFTargetTracking(ros::NodeHandle nh, ros::NodeHandle pnh);
    void imageCB(
       const sensor_msgs::Image::ConstPtr &);
    void screenPtCB(
       const geometry_msgs::PolygonStamped &);
    void imageOdomCB(
       const sensor_msgs::Image::ConstPtr &,
       const nav_msgs::Odometry::ConstPtr &);

    cv::Mat imageMsgToCvImage(
       const sensor_msgs::Image::ConstPtr &);
    void imageAndScreenPtCB(
       const sensor_msgs::Image::ConstPtr &,
       const geometry_msgs::PolygonStamped::ConstPtr &,
       const nav_msgs::Odometry::ConstPtr &);
   
};


#endif /* _KERNELIZED_CORRELATION_FILTERS_NODE_H_ */
