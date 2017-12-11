// Copyright (C) 2016, 2017 by Krishneel Chaudhary @ JSK Lab,
// The University of Tokyo, Japan
// krishneel@jsk.imi.i.u-tokyo.ac.jp

#pragma once
#ifndef _DUAL_NET_REGRESSION_H_
#define _DUAL_NET_REGRESSION_H_

#include <kernelized_correlation_filters_gpu/deep_feature_extraction.h>

//! temp include ONLY
#include "../../src/regression_network/helper.h"
#include "../../src/regression_network/bounding_box.h"
#include "../../src/regression_network/image_proc.h"

class DualNetRegression {

 private:
    bool is_net_setup_;
    std::string feature_name_;
    bool headless_;
    FeatureExtractor *feature_extractor_;
   
 public:
    DualNetRegression(const std::string, const std::string,
                      const std::string, const int = 0);
    void setupNetwork(const std::string, const std::string,
                      const std::string, const int = 0);
    bool correspondance(cv::Rect_<int> &, const cv::Mat, const cv::Rect_<int>,
                        const cv::Mat, const cv::Rect_<int>);
    void estimate(BoundingBox2D &, const cv::Mat, const cv::Mat);
};



#endif /* _DUAL_NET_REGRESSION_H_ */
