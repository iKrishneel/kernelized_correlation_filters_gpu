// Copyright (C) 2016 by Krishneel Chaudhary @ JSK Lab,
// The University of Tokyo, Japan
// krishneel@jsk.imi.i.u-tokyo.ac.jp

#include <kernelized_correlation_filters_gpu/deep_feature_extraction.h>

FeatureExtractor::FeatureExtractor(
    const std::string m_weight, const std::string d_proto,
    const std::string mean_file,
    const std::vector<std::string> b_names, const int device_id) :
    pretrained_model_weights_(m_weight), deploy_proto_(d_proto) {
    if (this->pretrained_model_weights_.empty()) {
       ROS_FATAL("CAFFE MODEL WEIGHTS NOT FOUND!");
       return;
    }
    if (this->deploy_proto_.empty()) {
       ROS_FATAL("MODEL PROTOTXT NOT FOUND!");
       return;
    }

    ROS_INFO("MODEL INFO:");
    std::cout << "\033[36m" << m_weight  << "\n";
    std::cout << d_proto  << "\n";
    std::cout << mean_file  << "\033[0m\n";
    
    ROS_INFO("\033[34m -- checking blob names ...\033[0m");
    // this->setExtractionLayers(b_names, 1);

    ROS_INFO("\033[34m -- checking completed ...\033[0m");
    
    CHECK_GE(device_id, 0);
    caffe::Caffe::SetDevice(device_id);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);

    ROS_INFO("\033[34m -- loading up ...\033[0m");
    this->loadPreTrainedCaffeModels(mean_file);
    ROS_INFO("\033[34m -- successfully loaded ...\033[0m");
}

void FeatureExtractor::setExtractionLayers(
    std::vector<std::string> b_names, const int min_batch) {
    for (int i = 0; i < b_names.size(); i++) {
       CHECK(feature_extractor_net_->has_blob(b_names[i]))
          << "Unknown feature blob name " << b_names[i]
          << " in the network " << this->deploy_proto_;
    }
}

bool FeatureExtractor::loadPreTrainedCaffeModels(
    const std::string mean_file) {
    this->feature_extractor_net_ =
       boost::shared_ptr<caffe::Net<float> >(
          new caffe::Net<float>(this->deploy_proto_, caffe::TEST));
    this->feature_extractor_net_->CopyTrainedLayersFrom(
       this->pretrained_model_weights_);
    
    caffe::Blob<float> *data_layer =
       this->feature_extractor_net_->input_blobs()[0];
    this->input_geometry_ = cv::Size(data_layer->width(),
                                     data_layer->height());

    this->num_channels_ = data_layer->channels();
    if (!mean_file.empty()) {
       this->setImageNetMean(mean_file);
    }
    
    return true;
}

void FeatureExtractor::getFeatures(
    cv::Mat image, const cv::Size filter_size) {
    if (image.empty()) {
       ROS_FATAL("IMAGE CHANNEL IS INCORRECT");
       return;
    }
    if (image.channels() < 3) {
       cv::cvtColor(image, image, CV_GRAY2BGR);
    }
    if (this->mean_.empty()) {
       ROS_WARN_ONCE("IMAGENET MEAN NOT SET");
    }
    caffe::Blob<float> *data_layer =
       this->feature_extractor_net_->input_blobs()[0];
    
    data_layer->Reshape(1, this->num_channels_,
                        this->input_geometry_.height,
                        this->input_geometry_.width);

    this->feature_extractor_net_->Reshape();
    std::vector<cv::Mat> input_channels;
    this->wrapInputLayer(&input_channels);

    this->preProcessImage(image, &input_channels);
    this->feature_extractor_net_->Forward();
    return;
}

//! for dual net
void FeatureExtractor::getFeatures(
    cv::Mat image1, cv::Mat image2) {
    if (image1.empty() || image2.empty()) {
       return;
    }
    if (image1.channels() != 3) {
       cv::cvtColor(image1, image1, CV_GRAY2BGR);
    }
    if (image2.channels() != 3) {
       cv::cvtColor(image2, image2, CV_GRAY2BGR);
    }
    assert(this->feature_extractor_net_->phase() == caffe::TEST);
    
    caffe::Blob<float> *data_layer1 =
       this->feature_extractor_net_->input_blobs()[0];
    data_layer1->Reshape(1, this->num_channels_,
                        this->input_geometry_.height,
                        this->input_geometry_.width);

    caffe::Blob<float> *data_layer2 =
       this->feature_extractor_net_->input_blobs()[1];
    data_layer2->Reshape(1, this->num_channels_,
                         this->input_geometry_.height,
                         this->input_geometry_.width);

    this->feature_extractor_net_->Reshape();

    std::vector<cv::Mat> image1_channels;
    this->wrapInputLayer(&image1_channels, 0);

    std::vector<cv::Mat> image2_channels;
    this->wrapInputLayer(&image2_channels, 1);

    this->preProcessImage(image1, &image1_channels, false);
    this->preProcessImage(image2, &image2_channels, false);
    this->feature_extractor_net_->Forward();
}

bool FeatureExtractor::getNamedBlob(
    boost::shared_ptr<caffe::Blob<float> > &blob_info1,
    const std::string blob_name) {
    if (blob_name.empty()) {
       ROS_ERROR("[::getNamedBlob]: BLOB NAME NOT PROVIDED");
       return false;
    }
    boost::shared_ptr<caffe::Blob<float> > blob_info =
       this->feature_extractor_net_->blob_by_name(blob_name);

    blob_info1.reset(new caffe::Blob<float>);
    blob_info1 = blob_info;
    return true;
}

void FeatureExtractor::preProcessImage(
    const cv::Mat& img, std::vector<cv::Mat>* input_channels,
    bool check) {
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1) {
       cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 4 && num_channels_ == 1) {
       cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    } else if (img.channels() == 4 && num_channels_ == 3) {
       cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    } else if (img.channels() == 1 && num_channels_ == 3) {
       cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    } else {
       sample = img;
    }
    
    cv::Mat sample_resized;
    if (sample.size() != this->input_geometry_) {
       cv::resize(sample, sample_resized, this->input_geometry_);
    } else {
       sample_resized = sample;
    }

    cv::Mat sample_float;
    if (num_channels_ == 3) {
       sample_resized.convertTo(sample_float, CV_32FC3);
    } else {
       sample_resized.convertTo(sample_float, CV_32FC1);
    }

    cv::Mat sample_normalized = sample_float;
    if (!this->mean_.empty()) {
       cv::subtract(sample_float, this->mean_, sample_normalized);
    }

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);

    if (check) {
       CHECK(reinterpret_cast<float*>(input_channels->at(0).data) ==
             this->feature_extractor_net_->input_blobs()[0]->cpu_data()) <<
          "Input channels are not wrapping the input layer of the network.";
    }
}

void FeatureExtractor::wrapInputLayer(
    std::vector<cv::Mat>* input_channels, const int index) {
    caffe::Blob<float>* input_layer =
       this->feature_extractor_net_->input_blobs()[index];
    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
       cv::Mat channel(height, width, CV_32FC1, input_data);
       input_channels->push_back(channel);
       input_data += width * height;
    }
}


bool FeatureExtractor::setImageNetMean(
    const std::string mean_file) {
    if (mean_file.empty()) {
       ROS_WARN("MEAN FILE NOT SET. IMAGE WONT BE DE-MEANED");
       return false;
    }

    std::cout << mean_file  << "\n";
    
    caffe::BlobProto blob_proto;
    caffe::ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    
    /* Convert from BlobProto to Blob<float> */
    caffe::Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), this->num_channels_)
       << "Number of channels of mean file doesn't match input layer.";

    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
       /* Extract an individual channel. */
       cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
       channels.push_back(channel);
       data += mean_blob.height() * mean_blob.width();
    }

    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
          * filled with this value. */
    cv::Scalar channel_mean = cv::mean(mean);
    this->mean_ = cv::Mat(this->input_geometry_, mean.type(), channel_mean);
}
