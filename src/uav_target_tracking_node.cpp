// Copyright (C) 2016 by Krishneel Chaudhary @ JSK Lab,
// The University of Tokyo, Japan
// krishneel@jsk.imi.i.u-tokyo.ac.jp

#include <uav_target_tracking/uav_target_tracking.h>

UAVTargetTracking::UAVTargetTracking(
    ros::NodeHandle nh, ros::NodeHandle pnh) :
    nh_(nh), pnh_(pnh), block_size_(8), tracker_init_(false),
    run_as_srv_(false), run_on_uav_(true), prev_scale_(1.0f),
    init_via_detector_(false), activate_on_signal_(true),
    invoke_detector_(false), secondary_detection_(true),
    iter_counter_(0) {

    this->screen_rect_ = cv::Rect_<int>(-1, -1, -1, -1);

    //! platform type (M100 = TX1 or S900 = GTX750)
    this->pnh_.getParam("uav_name", this->uav_name_);
    this->pnh_.getParam("run_as_srv", this->run_as_srv_);

    std::string kcf_params;
    this->pnh_.getParam("kcf_params", kcf_params);
    this->tracker_ = boost::shared_ptr<KernelizedCorrelationFiltersGPU>(
       new KernelizedCorrelationFiltersGPU(kcf_params, this->uav_name_));
    
    std::string pretrained_weights;
    this->pnh_.getParam("pretrained_weights", pretrained_weights);
    std::cout << pretrained_weights  << "\n";
    if (pretrained_weights.empty()) {
       ROS_FATAL("PROVIDE PRETRAINED WEIGHTS");
       return;
    }
    std::string model_prototxt;
    this->pnh_.getParam("model_prototxt", model_prototxt);
    std::cout << model_prototxt << "\n";
    if (model_prototxt.empty()) {
       ROS_FATAL("PROVIDE NETWORK PROTOTXT");
       return;
    }
    std::string imagenet_mean;
    this->pnh_.getParam("imagenet_mean", imagenet_mean);
    std::cout << imagenet_mean  << "\n";
    if (imagenet_mean.empty()) {
       ROS_ERROR("PROVIDE IMAGENET MEAN VALUE");
       return;
    }

    char* caffe_path;
    caffe_path = std::getenv("CAFFE_ROOT");
    if (caffe_path == NULL) {
       ROS_FATAL("CANNOT FIND [CAFFE_ROOT] ENVIRONMENT VARIABLE.");
       ROS_INFO("SET ENV VAR: export $CAFFE_ROOT=<caffe_directory>");
       std::exit(EXIT_FAILURE);
    }
    std::string caffe_root(caffe_path);
    char last = caffe_root.back();
    caffe_root = (last != '/') ? (caffe_root + "/") : caffe_root;
    
    this->pnh_.param("downsize", downsize_, 1);
    this->pnh_.param("rectangle_resizing_scale", rectangle_resizing_scale_, 1);
    
    int device_id;
    this->pnh_.param<int>("device_id", device_id, 0);
    this->pnh_.getParam("device_id", device_id);

    this->pnh_.param("headless", headless_, false);
    
    pretrained_weights = caffe_root + pretrained_weights;
    //! model_prototxt = caffe_root + model_prototxt;
    imagenet_mean = caffe_root + imagenet_mean;
    
    // TOD0(ADD): any later
    std::vector<std::string> feat_ext_layers(1);
    feat_ext_layers[0] = "conv5";  //! currently hardcoded
    
    this->tracker_->setCaffeInfo(pretrained_weights,
                                 model_prototxt, imagenet_mean,
                                 feat_ext_layers, 0);

    //! set down_size
    this->pnh_.param<int>("downsize", this->downsize_, 1);
    this->pnh_.getParam("downsize", this->downsize_);

    //! when using manual marking
    this->pnh_.param<int>("resize_factor", this->resize_factor_, 1);
    this->pnh_.getParam("resize_factor", this->resize_factor_);

    //! gpu warm up
    float *d_warmup;
    cudaMalloc(reinterpret_cast<void**>(&d_warmup), sizeof(int) * 20);
    cudaFree(d_warmup);

    //! program run type
    this->pnh_.getParam("run_on_uav", this->run_on_uav_);
    this->pnh_.getParam("init_from_detector", this->init_via_detector_);
    this->pnh_.getParam("activate_on_signal", this->activate_on_signal_);

    this->camera_info_.height = 0;
    this->camera_info_.width = 0;
    
    this->onInit();
}

void UAVTargetTracking::onInit() {
     this->subscribe();
     this->pub_image_ = nh_.advertise<sensor_msgs::Image>(
         "target", 1);
     this->pub_position_ = pnh_.advertise<geometry_msgs::PointStamped>(
        "/object_image_center", sizeof(char));
}

void UAVTargetTracking::subscribe() {
     if (!this->run_on_uav_) {
        this->sub_image_ = this->nh_.subscribe(
           "image", 1, &UAVTargetTracking::imageCB, this);
     } else {
        this->msub_image_.subscribe(this->nh_, "image", 1);
        this->msub_odom_.subscribe(this->nh_, "odom", 1);
        this->sync_ = boost::make_shared<message_filters::Synchronizer<
                                            SyncPolicy> >(100);
        this->sync_->connectInput(this->msub_image_, this->msub_odom_);
        this->sync_->registerCallback(
           boost::bind(&UAVTargetTracking::imageOdomCB, this, _1, _2));

        this->sub_info_ = this->nh_.subscribe(
           "info", 1, &UAVTargetTracking::cameraInfoCB, this);
     }

     if (this->init_via_detector_) {
        this->init_image_.subscribe(this->nh_, "init_image", 1);
        this->init_rect_.subscribe(this->nh_, "init_rect", 1);
        this->init_odom_.subscribe(this->nh_, "init_odom", 1);
        this->init_sync_ = boost::make_shared<message_filters::Synchronizer<
                                                 InitPolicy> > (100);
        this->init_sync_->connectInput(this->init_image_, this->init_rect_,
                                       this->init_odom_);
        this->init_sync_->registerCallback(
           boost::bind(&UAVTargetTracking::imageAndScreenPtCB, this,
                       _1, _2, _3));
     } else {
        this->sub_screen_pt_ = this->nh_.subscribe(
           "input_screen", 1, &UAVTargetTracking::screenPtCB, this);
     }

     if (this->activate_on_signal_) {
        this->sub_signal_ = this->nh_.subscribe(
           "activation_signal", 1, &UAVTargetTracking::signalCB, this);
     }
}

void UAVTargetTracking::unsubscribe() {
    if (!this->run_on_uav_) {
       this->sub_image_.shutdown();
    } else {
       this->msub_image_.unsubscribe();
       this->msub_odom_.unsubscribe();
    }
}

void UAVTargetTracking::signalCB(
    const std_msgs::Empty &signal_msg) {
    ROS_INFO_ONCE("ACTIVATION SIGNAL RECEIVED");
    this->tracker_init_ = true;
}

void UAVTargetTracking::cameraInfoCB(
    const sensor_msgs::CameraInfo::ConstPtr &info_msg) {
    if (this->camera_info_.height != 0 && this->camera_info_.width != 0) {
       if (this->camera_info_.K[0] != info_msg->K[0] &&
           !this->invoke_detector_) {
          ROS_WARN("REDETECTION INVOKED");
          this->invoke_detector_ = true;
       }
    }
    this->camera_info_ = *info_msg;
}

void UAVTargetTracking::screenPtCB(
     const geometry_msgs::PolygonStamped &screen_msg) {
   
    if (screen_msg.polygon.points.size() == 0) {
       return;
    }
    int x = screen_msg.polygon.points[0].x;
    int y = screen_msg.polygon.points[0].y;
    int width = screen_msg.polygon.points[1].x - x;
    int height = screen_msg.polygon.points[1].y - y;
    
    x *= this->resize_factor_;
    y *= this->resize_factor_;
    width *= this->resize_factor_;
    height *= this->resize_factor_;
    
    this->screen_rect_ = cv::Rect_<int>(
       x * this->rectangle_resizing_scale_ / this->downsize_,
       y * this->rectangle_resizing_scale_/ this->downsize_,
       width * this->rectangle_resizing_scale_ / this->downsize_,
       height * this->rectangle_resizing_scale_ / this->downsize_);
    
    std::cout << "INIT SIZE: " << screen_rect_  << "\n";
    if (width > this->block_size_ && height > this->block_size_) {
       this->tracker_init_ = true;
       this->sub_screen_pt_.shutdown();
    } else {
       ROS_WARN("-- Selected Object Size is too small... Not init tracker");
    }
}

void UAVTargetTracking::imageAndScreenPtCB(
    const sensor_msgs::Image::ConstPtr &image_msg,
    const geometry_msgs::PolygonStamped::ConstPtr &screen_msg,
    const nav_msgs::Odometry::ConstPtr &odom_msg) {
    if (screen_msg->polygon.points.size() == 0) {
       return;
    }
    int x = screen_msg->polygon.points[0].x;
    int y = screen_msg->polygon.points[0].y;
    int width = screen_msg->polygon.points[1].x - x;
    int height = screen_msg->polygon.points[1].y - y;

    cv::Rect_<int> rect = cv::Rect_<int>(
       x * this->rectangle_resizing_scale_ / this->downsize_,
       y * this->rectangle_resizing_scale_/ this->downsize_,
       width * this->rectangle_resizing_scale_ / this->downsize_,
       height * this->rectangle_resizing_scale_ / this->downsize_);

    if (width > this->block_size_ && height > this->block_size_) {
       cv::Mat image = imageMsgToCvImage(image_msg);
       ROS_INFO("Initializing Tracker");
       this->tracker_->init(image, rect);
       this->tracker_init_ = false;
       ROS_INFO("Tracker Initialization Complete");

       //! simple scaling
       this->pixel_lenght_ = cv::norm(rect.br() - rect.tl());
       this->init_altitude_ = odom_msg->pose.pose.position.z;
       this->prev_height_ = this->init_altitude_;
       this->screen_rect_ = rect;

       //! shutdown
       this->init_image_.unsubscribe();
       this->init_rect_.unsubscribe();
       this->init_odom_.unsubscribe();
    } else {
       ROS_WARN("-- Selected Object Size is too small... Not init tracker");
    }
}

cv::Mat UAVTargetTracking::imageMsgToCvImage(
    const sensor_msgs::Image::ConstPtr &image_msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
       cv_ptr = cv_bridge::toCvCopy(
          image_msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
       ROS_ERROR("cv_bridge exception: %s", e.what());
       return cv::Mat();
    }
    cv::Mat image = cv_ptr->image.clone();
    if (this->downsize_ > 1) {
         cv::resize(image, image, cv::Size(image.cols/this->downsize_,
                                           image.rows/this->downsize_));
     }
    return image;
}


void UAVTargetTracking::imageCB(
    const sensor_msgs::Image::ConstPtr &image_msg) {
    
    std::clock_t start;
    double duration;
    start = std::clock();

    cv::Mat image = imageMsgToCvImage(image_msg);
    if (image.empty()) {
       ROS_ERROR("EMPTY INPUT IMAGE");
       return;
    }

    if (this->run_as_srv_) {
       this->tracker_->detector_->setImageMsg(image_msg);
    }
    
    if (this->tracker_init_) {
       ROS_INFO("Initializing Tracker");
       if (this->run_as_srv_) {
          this->tracker_->detector_->setImageMsg(image_msg);
          if (this->tracker_->detector_->detect(this->screen_rect_)) {
             this->tracker_->init(image, this->screen_rect_);
             this->tracker_init_ = false;
          }
       } else {
          this->tracker_->init(image, this->screen_rect_);
          this->tracker_init_ = false;
       }
       ROS_INFO("Tracker Initialization Complete");
        
       duration = (std::clock() - start) /
          static_cast<double>(CLOCKS_PER_SEC);
       ROS_INFO("INITIALIZATION TIME: %3.5f", duration);
    }

     geometry_msgs::PointStamped point_msg;
     point_msg.header = image_msg->header;
     if (this->screen_rect_.width > this->block_size_) {
        this->tracker_->track(image);
        BoundingBox bb = this->tracker_->getBBox();
        cv::Rect rect = cv::Rect(bb.cx - bb.w/2.0f,
                                 bb.cy - bb.h/2.0f, bb.w, bb.h);
        cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
        point_msg.point.x = bb.cx * downsize_;
        point_msg.point.y = bb.cy * downsize_;
        if ((point_msg.point.x > image.cols * downsize_) ||
            (point_msg.point.y > image.rows * downsize_)) {
           point_msg.point.x = -1;
           point_msg.point.y = -1;
        } else if ((point_msg.point.x < 0) ||
                   (point_msg.point.y < 0)) {
           point_msg.point.x = -1;
           point_msg.point.y = -1;
        }
        
        duration = (std::clock() - start) /
           static_cast<double>(CLOCKS_PER_SEC);
        // ROS_INFO_THROTTLE(1, "PROCESS: %3.5f", duration);
     } else {
        ROS_ERROR_ONCE("THE TRACKER IS NOT INITALIZED");
        point_msg.point.x = -1;
        point_msg.point.y = -1;
     }

     this->pub_position_.publish(point_msg);
     
     cv_bridge::CvImagePtr pub_msg(new cv_bridge::CvImage);
     pub_msg->header = image_msg->header;
     pub_msg->encoding = sensor_msgs::image_encodings::BGR8;
     pub_msg->image = image.clone();
     this->pub_image_.publish(pub_msg);

     if (!this->headless_) {
        cv::namedWindow("Tracking", cv::WINDOW_NORMAL);
        cv::imshow("Tracking", image);
        cv::waitKey(3);
     }
}

void UAVTargetTracking::imageOdomCB(
    const sensor_msgs::Image::ConstPtr &image_msg,
    const nav_msgs::Odometry::ConstPtr &odom_msg) {

    std::clock_t start;
    double duration;
    start = std::clock();

    cv::Mat image = imageMsgToCvImage(image_msg);
    if (image.empty()) {
       ROS_ERROR("EMPTY INPUT IMAGE");
       return;
    }
    
    if (this->run_as_srv_) {
       this->tracker_->detector_->setROSMsg(image_msg, odom_msg);
    }
    
    float uav_altitude = odom_msg->pose.pose.position.z;
    
    if (this->tracker_init_) {
       if (this->run_as_srv_) {
          this->tracker_->detector_->setROSMsg(image_msg, odom_msg);
          bool is_init = this->tracker_->detector_->detect(this->screen_rect_);
          if (is_init && this->screen_rect_.x > 0 && this->screen_rect_.y > 0 &&
              this->screen_rect_.width > this->block_size_ &&
              this->screen_rect_.height > this->block_size_) {
             this->tracker_->init(image, this->screen_rect_);
             this->tracker_init_ = false;
          }
       } else {
          this->tracker_->init(image, this->screen_rect_);
          this->tracker_init_ = false;
       }

       if (!this->tracker_init_) {
          ROS_INFO("Tracker Initialization Complete");

          //! simple scaling
          this->pixel_lenght_ = cv::norm(this->screen_rect_.br() -
                                         this->screen_rect_.tl());
          this->init_altitude_ = uav_altitude;
          this->prev_height_ = uav_altitude;

          duration = (std::clock() - start) /
             static_cast<double>(CLOCKS_PER_SEC);
          ROS_INFO("INITIALIZATION TIME: %3.5f", duration);
       }
    }
    
    geometry_msgs::PointStamped point_msg;
    point_msg.header = image_msg->header;
    if (this->screen_rect_.width > this->block_size_) {

       //! second detection to catch fast moving target
       if (secondary_detection_ && this->iter_counter_ < 2) {
          this->secondary_detection_ = this->tracker_->redetectTarget();
          if (this->secondary_detection_) {
             if (this->iter_counter_ == 1) {
                secondary_detection_ = false;
             }
             this->iter_counter_++;
          }
          this->invoke_detector_ = false;
       }
       
       if (this->invoke_detector_ && uav_altitude > 2.0f) {
          this->invoke_detector_ = this->tracker_->redetectTarget() ?
             true : false;
       }

       //! simple scaling ratio
       float factor = 1.2f;
       if (uav_altitude < 3.0f) {  //! NOT THE BEST
          this->tracker_->switchRedetection(false);
       } else {
          this->tracker_->switchRedetection(true);
       }
       
       float scaled_lenght = (std::pow(this->init_altitude_, factor) *
                              this->pixel_lenght_);
        
       scaled_lenght /= (std::pow(uav_altitude, factor));
       float scale = scaled_lenght / this->pixel_lenght_;
       scale = (std::isnan(scale)) ? this->prev_scale_ : scale;
       float scale_ratio = scale / this->prev_scale_;

       this->prev_scale_ = scale;
       this->prev_height_ = uav_altitude;
        
       this->tracker_->track(image, scale_ratio);
       BoundingBox bb = this->tracker_->getBBox();
       cv::Rect rect = cv::Rect(bb.cx - bb.w/2.0f,
                                bb.cy - bb.h/2.0f, bb.w, bb.h);
       cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);

       point_msg.point.x = bb.cx * downsize_;
       point_msg.point.y = bb.cy * downsize_;
       if ((point_msg.point.x > image.cols * downsize_) ||
           (point_msg.point.y > image.rows * downsize_)) {
          point_msg.point.x = -1;
          point_msg.point.y = -1;
       } else if ((point_msg.point.x < 0) ||
                  (point_msg.point.y < 0)) {
          point_msg.point.x = -1;
          point_msg.point.y = -1;
       }
       
        duration = (std::clock() - start) /
           static_cast<double>(CLOCKS_PER_SEC);
        // std::cout << "PROCESS: " << duration  << "\n";
        // ROS_INFO_THROTTLE(1, "PROCESS: %3.5f", duration);
     } else {
        ROS_ERROR_ONCE("THE TRACKER IS NOT INITALIZED");
        point_msg.point.x = -1;
        point_msg.point.y = -1;
     }

    this->pub_position_.publish(point_msg);
     
    cv_bridge::CvImagePtr pub_msg(new cv_bridge::CvImage);
    pub_msg->header = image_msg->header;
    pub_msg->encoding = sensor_msgs::image_encodings::BGR8;
    pub_msg->image = image.clone();
    this->pub_image_.publish(pub_msg);

    if (!this->headless_) {
       cv::namedWindow("Tracking", cv::WINDOW_NORMAL);
       cv::imshow("Tracking", image);
       cv::waitKey(3);
    }
}

int main(int argc, char *argv[]) {
   
    ros::init(argc, argv, "kernelized_correlation_filters");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    UAVTargetTracking kcf(nh, pnh);
    ros::spin();
    return 0;
}

