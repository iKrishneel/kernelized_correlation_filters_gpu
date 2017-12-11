
#ifndef _DEEP_FEATURE_EXTRACTION_H_
#define _DEEP_FEATURE_EXTRACTION_H_

#include <ros/ros.h>
#include <ros/console.h>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/thread/mutex.hpp>
#include <caffe/caffe.hpp>

class FeatureExtractor {
   
 private:
    std::string pretrained_model_weights_;
    std::string deploy_proto_;
    std::vector<std::string> blob_names_;
    int num_min_batch_;

    int num_channels_;
    cv::Size input_geometry_;
    cv::Mat mean_;

 protected:
    boost::shared_ptr<caffe::Net<float> > feature_extractor_net_;
   
 public:
    FeatureExtractor(const std::string,
                     const std::string,
                     const std::string,
                     const std::vector<std::string>,
                     const int);
    bool loadPreTrainedCaffeModels(const std::string);
    void setExtractionLayers(std::vector<std::string>,
                             const int);
    void getFeatures(cv::Mat, const cv::Size = cv::Size(-1, -1));
    bool setImageNetMean(const std::string);
    void wrapInputLayer(std::vector<cv::Mat>*);
    void preProcessImage(const cv::Mat &,
                         std::vector<cv::Mat>*);
    bool getNamedBlob(boost::shared_ptr<caffe::Blob<float> > &,
                      const std::string);
};


#endif /* _DEEP_FEATURE_EXTRACTION_H_ */
