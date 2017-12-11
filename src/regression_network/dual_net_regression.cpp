// Copyright (C) 2016, 2017 by Krishneel Chaudhary @ JSK Lab,
// The University of Tokyo, Japan
// krishneel@jsk.imi.i.u-tokyo.ac.jp

#include <kernelized_correlation_filters_gpu/dual_net_regression.h>

DualNetRegression::DualNetRegression(
    const std::string model_proto, const std::string pretrained_weights,
    const std::string mean_file, const int device_id) :
    is_net_setup_(false), headless_(!true) {
    if (model_proto.empty() || pretrained_weights.empty()) {
       ROS_ERROR("[::DualNetRegression]: EMPTY FILENAMES");
       return;
    }
    this->feature_name_ = "fc8";
    this->setupNetwork(model_proto, pretrained_weights, mean_file, device_id);
}

void DualNetRegression::setupNetwork(
    const std::string model_proto, const std::string pretrained_weights,
    const std::string mean_file, const int device_id) {
    std::vector<std::string> b_names;
    this->feature_extractor_ = new FeatureExtractor(
       pretrained_weights, model_proto, mean_file, b_names, device_id);
    this->is_net_setup_ = true;
}

bool DualNetRegression::correspondance(
    cv::Rect_<int> &bounding_box, const cv::Mat curr_image,
    const cv::Rect_<int> curr_rect, const cv::Mat prev_image,
    const cv::Rect_<int> prev_rect) {
    if (prev_image.empty() || curr_image.empty()) {
       ROS_ERROR("[::regression]: EMPTY INPUTS");
       return false;
    }

    //! convert rect to boundingbox
    BoundingBox2D bbox_prev;
    bbox_prev.x1_ = prev_rect.x;
    bbox_prev.y1_ = prev_rect.y;
    bbox_prev.x2_ = prev_rect.br().x;
    bbox_prev.y2_ = prev_rect.br().y;

    cv::Mat templ_roi;
    CropPadImage(bbox_prev, prev_image, &templ_roi);

    BoundingBox2D bbox_curr;
    bbox_curr.x1_ = curr_rect.x;
    bbox_curr.y1_ = curr_rect.y;
    bbox_curr.x2_ = curr_rect.br().x;
    bbox_curr.y2_ = curr_rect.br().y;
    
    cv::Mat search_roi;
    BoundingBox2D search_location;
    double edge_spacing_x, edge_spacing_y;
    CropPadImage(bbox_curr, curr_image, &search_roi, &search_location,
                 &edge_spacing_x, &edge_spacing_y);

    ROS_WARN("SEARCH LOC");
    search_location.Print();
    std::cout << curr_rect  << "\n";

    if (!this->headless_) {
       cv::Mat target_resize;
       cv::resize(templ_roi, target_resize, cv::Size(227, 227));
       cv::imshow("template_roi", target_resize);

       cv::Mat c_resize;
       cv::resize(search_roi, c_resize, cv::Size(227, 227));
       cv::imshow("current_roi", c_resize);
    }
    
    //! regress
    BoundingBox2D bbox_estimate;
    estimate(bbox_estimate, search_roi, templ_roi);

    BoundingBox2D bbox_estimate_unscaled;
    bbox_estimate.Unscale(search_roi, &bbox_estimate_unscaled);
    
    bbox_estimate_unscaled.Uncenter(curr_image, search_location, edge_spacing_x,
                                    edge_spacing_y, &bbox_estimate);

    bounding_box.x = bbox_estimate.x1_;
    bounding_box.y = bbox_estimate.y1_;
    bounding_box.width = bbox_estimate.x2_ - bbox_estimate.x1_;
    bounding_box.height = bbox_estimate.y2_ - bbox_estimate.y1_;
}

void DualNetRegression::estimate(
    BoundingBox2D &bbox, const cv::Mat search_roi, const cv::Mat templ_roi) {
    if (search_roi.empty() || templ_roi.empty()) {
       ROS_ERROR("[::estimate]: EMPTY INPUTS");
       return;
    }
    this->feature_extractor_->getFeatures(templ_roi, search_roi);
    boost::shared_ptr<caffe::Blob<float> > blob_info(new caffe::Blob<float>);
    this->feature_extractor_->getNamedBlob(blob_info,
                                           this->feature_name_.c_str());

    int num_elements = 1;
    for (int i = 0; i < blob_info->num_axes(); ++i) {
       const int elements_in_dim = blob_info->shape(i);
       num_elements *= elements_in_dim;
    }
    const float* begin = blob_info->cpu_data();
    const float* end = begin + num_elements;
    bbox = BoundingBox2D(std::vector<float>(begin, end));
}
