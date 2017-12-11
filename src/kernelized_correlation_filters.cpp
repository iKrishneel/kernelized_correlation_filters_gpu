// Copyright (C) 2016, 2017 by Krishneel Chaudhary @ JSK Lab,
// The University of Tokyo, Japan

#include <kernelized_correlation_filters_gpu/kernelized_correlation_filters.h>

KernelizedCorrelationFiltersGPU::KernelizedCorrelationFiltersGPU(
    const std::string param_filename) : resize_image_(false),
                                        is_cnn_set_(false), use_scale_(false),
                                        use_subgrid_scale_(false),
                                        use_subpixel_localization_(true),
                                        is_update_model_(true),
                                        use_max_boxes_(false),
                                        detect_lost_target_(true),
                                        is_drn_set_(false), use_drn_(false) {
    padding_ = 1.0;
    output_sigma_factor_ = 0.1;
    kernel_sigma_ = 0.5;    // def = 0.5
    lambda_ = 1e-4;         // regularization in learning step
    interp_factor_ = 0.02;
    cell_size_ = 4;
    scale_step_ = 1.1;
    current_scale_ = 1.0f;
    num_scales_ = 3;
    psr_update_thresh_ = 10.0;
    psr_detect_thresh_ = 2.0;
    num_proposals_ = 5;
    FILTER_SIZE_ = 0;
    FILTER_BATCH_ = 256;  //! change to read from caffe

    if (!param_filename.empty()) {
       this->parseParamsFromFile(param_filename);
    } else {
       ROS_WARN("HYPERPARAMETER TUNING FILE NOT FOUND!");
       ROS_WARN("TRACKER WILL BE INIT WILL DEFAULT PARAMETERS!");
    }
    
    this->blob_info_ = boost::shared_ptr<caffe::Blob<float> >(
       new caffe::Blob<float>);
    this->cublas_status_ = cublasCreate(&cublas_handle_);
}

void KernelizedCorrelationFiltersGPU::setCaffeInfo(
    const std::string pretrained_weights, const std::string model_prototxt,
    const std::string mean_file,
    std::vector<std::string> &feature_layers, const int device_id) {
    this->feature_extractor_ = boost::shared_ptr<FeatureExtractor>(
       new FeatureExtractor(pretrained_weights, model_prototxt, mean_file,
                            feature_layers, device_id));
    
    this->is_cnn_set_ = true;
}

void KernelizedCorrelationFiltersGPU::setRegressionNet(
    const std::string pretrained_weights, const std::string model_prototxt,
    const std::string mean_file, const int device_id) {
    if (!this->use_drn_) {
       ROS_WARN("REGRESSION NET WILL NOT BE LOADED!");
       this->is_drn_set_ = false;
       return;
    }
    if (pretrained_weights.empty() || model_prototxt.empty()) {
       ROS_WARN("REGRESSION NET MODEL INFO NOT PROVIDED!");
       ROS_WARN("TRACKER WILL RUN WITHOUT VISUAL SCALE ESTIMATION!");
       this->is_drn_set_ = false;
       return;
    }
    this->regression_net_ = boost::shared_ptr<DualNetRegression>(
       new DualNetRegression(model_prototxt, pretrained_weights,
                             mean_file, device_id));
    this->is_drn_set_ = true;
}

void KernelizedCorrelationFiltersGPU::init(
    cv::Mat &img, const cv::Rect & bbox) {
    if (!this->is_cnn_set_) {
       ROS_FATAL("CAFFE CNN INFO NOT SET");
       return;
    }
    double x1 = bbox.x;
    double x2 = bbox.x + bbox.width;
    double y1 = bbox.y;
    double y2 = bbox.y + bbox.height;
    
    x1 = (x1 < 0.0) ? 0.0 : x1;
    x2 = (x2 > img.cols - 1) ? img.cols - 1 : x2;
    y1 = (y1 < 0.0) ? 0.0 : y1;
    y2 = (y2 > img.rows - 1) ? img.rows - 1 : y2;
    
    if (x2 - x1 < 2 * cell_size_) {
        double diff = (2 * cell_size_ - x2 + x1) / 2.0;
        if (x1 - diff >= 0 && x2 + diff < img.cols) {
            x1 -= diff;
            x2 += diff;
        } else if (x1 - 2*diff >= 0) {
            x1 -= 2*diff;
        } else {
            x2 += 2*diff;
        }
    }
    if (y2 - y1 < 2 * cell_size_) {
        double diff = (2*cell_size_ -y2+y1)/2.;
        if (y1 - diff >= 0 && y2 + diff < img.rows) {
            y1 -= diff;
            y2 += diff;
        } else if (y1 - 2*diff >= 0) {
            y1 -= 2*diff;
        } else {
            y2 += 2*diff;
        }
    }

    this->pose_.w = x2-x1;
    this->pose_.h = y2-y1;
    this->pose_.cx = x1 + this->pose_.w/2.;
    this->pose_.cy = y1 + this->pose_.h/2.;
    
    // cv::Mat input_gray;
    cv::Mat input_rgb = img.clone();

    // don't need too large image
    if (this->pose_.w * this->pose_.h > 100.0 * 100.0) {
        std::cout << "resizing image by factor of 2" << std::endl;
        resize_image_ = true;
        this->pose_.scale(0.5);
        cv::resize(input_rgb, input_rgb, cv::Size(0, 0),
                   0.5, 0.5, cv::INTER_AREA);
    }

    this->windows_size_[0] = round(pose_.w * (1. + padding_) /
                              cell_size_) * cell_size_;
    this->windows_size_[1] = round(pose_.h * (1. + padding_) /
                                   cell_size_) * cell_size_;

    this->scales_.clear();
    if (this->use_scale_) {
       for (int i = -num_scales_/2; i <= num_scales_/2; ++i) {
           this->scales_.push_back(std::pow(scale_step_, i));
       }
    } else {
       this->scales_.push_back(1.);
    }

    this->current_scale_ = 1.;

    double min_size_ratio = std::max(5.0f * cell_size_/ windows_size_[0],
                                     5.0f * cell_size_/windows_size_[1]);
    double max_size_ratio = std::min(
       floor((img.cols + windows_size_[0]/3)/cell_size_)*
       cell_size_/windows_size_[0],
       floor((img.rows + windows_size_[1]/3)/cell_size_)*
       cell_size_/windows_size_[1]);
    min_max_scale_[0] = std::pow(scale_step_,
                                 std::ceil(std::log(min_size_ratio) /
                                           log(scale_step_)));
    min_max_scale_[1] = std::pow(scale_step_,
                                  std::floor(std::log(max_size_ratio) /
                                             log(scale_step_)));
    // min_max_scale_[1] = std::min(img.rows, img.cols) /
    //    std::max(bbox.width, bbox.height);

    ROS_INFO("TRACKER INIT INFO");
    std::cout << "init: img size " << img.cols << " " << img.rows << std::endl;
    std::cout << "init: win size. " << windows_size_[0] << " "
              << windows_size_[1] << std::endl;
    std::cout << "init: min max scales factors: " << min_max_scale_[0]
              << " " << min_max_scale_[1] << std::endl;
    std::cout << "init: window: " << bbox << std::endl;
    
    output_sigma_ = std::sqrt(pose_.w*pose_.h) *
       output_sigma_factor_ / static_cast<double>(cell_size_);

    
    // window weights, i.e. labels
    cv::Mat gsl = gaussianShapedLabels(output_sigma_,
                                       windows_size_[0]/cell_size_,
                                       windows_size_[1]/cell_size_);

    this->FILTER_SIZE_ = gsl.rows * gsl.cols;
    this->filter_size_ = gsl.size();
    
    //! setup cuffthandles
    cufftResult cufft_status;
    cufft_status = cufftPlan1d(&cufft_handle1_, FILTER_SIZE_,
                               CUFFT_C2C, 1);
    if (cufft_status != cudaSuccess) {
       ROS_FATAL("CUDA FFT HANDLE CREATION FAILED");
       std::exit(1);
    }

    cufft_status = cufftPlan1d(
       &handle_, FILTER_SIZE_, CUFFT_C2C, FILTER_BATCH_);
    if (cufft_status != cudaSuccess) {
       ROS_ERROR("CUDAFFT PLAN [C2C] ALLOC FAILED");
       std::exit(-1);  //! change to shutdown
    }

    ROS_INFO("\n\033[35m ALL CUFFT PLAN SETUP DONE \033[0m\n");

    float *dev_data;
    int IN_BYTE = FILTER_SIZE_ * sizeof(float);
    cudaMalloc(reinterpret_cast<void**>(&dev_data), IN_BYTE);
    cudaMemcpy(dev_data, reinterpret_cast<float*>(gsl.data), IN_BYTE,
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    dev_p_yf_ = this->cuDFT(dev_data, cufft_handle1_, 1, FILTER_SIZE_);

    cos_window_ = cosineWindowFunction(gsl.cols, gsl.rows);
    
    this->BYTE_ = FILTER_BATCH_ * cos_window_.rows *
       cos_window_.cols * sizeof(float);

    float *cosine_window_1D = reinterpret_cast<float*>(std::malloc(BYTE_));
    int icounter = 0;
    for (int i = 0; i < FILTER_BATCH_; i++) {
       for (int j = 0; j < cos_window_.rows; j++) {
          for (int k = 0; k < cos_window_.cols; k++) {
             cosine_window_1D[icounter] = cos_window_.at<float>(j, k);
             icounter++;
          }
       }
    }
    cudaMalloc(reinterpret_cast<void**>(&d_cos_window_), BYTE_);
    cudaMemcpy(d_cos_window_, cosine_window_1D, BYTE_, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    /**
     * GPU PROCESSING
     */
    
    //! allocate reusable memory
    //! reuse memories in other functions
    const int data_lenght = FILTER_BATCH_ * FILTER_SIZE_;
    cudaMalloc(reinterpret_cast<void**>(&d_cos_conv_), BYTE_);
    cudaMalloc(reinterpret_cast<void**>(&d_squared_norm_),
               data_lenght * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_summation_),
               data_lenght * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_conj_mxf),
               data_lenght * sizeof(cufftComplex));
    cudaMalloc(reinterpret_cast<void**>(&d_xyf_),
               data_lenght * sizeof(cufftComplex));
    cudaMalloc(reinterpret_cast<void**>(&d_kzf_),
               FILTER_SIZE_ * sizeof(cufftComplex));
    cudaMalloc(reinterpret_cast<void**>(&d_xysum_),
               FILTER_SIZE_ * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_f2c1_),
               FILTER_SIZE_ * sizeof(cufftComplex));
    cudaMalloc(reinterpret_cast<void**>(&d_kf_),
               FILTER_SIZE_ * sizeof(cufftComplex));
    cudaMalloc(reinterpret_cast<void**>(&d_f2c_),
               data_lenght * sizeof(cufftComplex));
    cudaMalloc(reinterpret_cast<void**>(&d_ifft_),
               data_lenght * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_ifft1_),
               FILTER_SIZE_ * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_complex_),
               data_lenght * sizeof(cufftComplex));
    
    float *dev_feat = getFeaturesGPU(input_rgb, input_rgb, pose_.cx, pose_.cy,
                                     windows_size_[0], windows_size_[1], 1.0f);
    cosineConvolutionGPU(&d_cos_conv_, dev_feat, this->d_cos_window_,
                         FILTER_BATCH_ * FILTER_SIZE_);
    dev_model_xf_ = this->cuDFT(d_cos_conv_, handle_,
                                FILTER_BATCH_, FILTER_SIZE_);
    
    float kf_xf_norm = 0.0f;
    float *dev_kxyf = squaredNormAndMagGPU(kf_xf_norm, dev_model_xf_,
                                           FILTER_BATCH_, FILTER_SIZE_);

    ROS_INFO("\033[35m INIT NORM: %3.3f \033[0m", kf_xf_norm);

    float kf_yf_norm = kf_xf_norm;
    convertFloatToComplexGPU(&d_f2c_, dev_kxyf, FILTER_BATCH_, FILTER_SIZE_);
    this->cuInvDFT(&d_ifft_, d_f2c_, handle_, FILTER_BATCH_, FILTER_SIZE_);
    invFFTSumOverFiltersGPU(&d_xysum_, d_ifft_, FILTER_BATCH_, FILTER_SIZE_);
    
    float normalizer = 1.0f / (static_cast<float>(FILTER_SIZE_*FILTER_BATCH_));
    cuGaussianExpGPU(d_xysum_, kf_xf_norm, kf_yf_norm, kernel_sigma_,
                     normalizer, FILTER_SIZE_);
    convertFloatToComplexGPU(&d_f2c1_, d_xysum_, 1, FILTER_SIZE_);
    cuDFT(&d_kf_, d_f2c1_, cufft_handle1_, 1, FILTER_SIZE_);

    this->dev_model_alphaf_num_ = multiplyComplexGPU(
        dev_p_yf_, d_kf_, FILTER_SIZE_);
     addComplexByScalarGPU(&d_kzf_, d_kf_, static_cast<float>(lambda_),
                           FILTER_SIZE_);
     this->dev_model_alphaf_den_ = multiplyComplexGPU(
        d_kf_, d_kzf_, FILTER_SIZE_);
     this->dev_model_alphaf_ = divisionComplexGPU(
        this->dev_model_alphaf_num_, this->dev_model_alphaf_den_, FILTER_SIZE_);

     //! setup for redetection
     cv::Size nimg_size = cv::Size(img.cols/2, img.rows/2);
     // this->lostp_ = boost::shared_ptr<LostTargetPursuait>(
     //    new LostTargetPursuit(nimg_size, this->num_proposals_,
     //                          this->use_max_boxes_));
     this->prev_img_ = img.clone();
     this->prev_rect_ = bbox;
     
     cudaFree(dev_feat);
     cudaFree(dev_kxyf);
     free(cosine_window_1D);
}

void KernelizedCorrelationFiltersGPU::setTrackerPose(
    BoundingBox &bbox, cv::Mat & img) {
     init(img, bbox.getRect());
}

void KernelizedCorrelationFiltersGPU::updateTrackerPosition(
    BoundingBox &bbox) {
    if (resize_image_) {
       BoundingBox tmp = bbox;
       tmp.scale(0.5);
       pose_.cx = tmp.cx;
       pose_.cy = tmp.cy;
    } else {
       pose_.cx = bbox.cx;
       pose_.cy = bbox.cy;
    }
}

BoundingBox KernelizedCorrelationFiltersGPU::getBBox() {
    BoundingBox tmp = pose_;
    tmp.w *= current_scale_;
    tmp.h *= current_scale_;

    if (this->resize_image_) {
       tmp.scale(2);
    }
    return tmp;
}

/**
 * altitude_ratio = ((uav_height_<init> * rect_<init>) /
 * uav_height_<now>) / init
 */

void KernelizedCorrelationFiltersGPU::track(
    cv::Mat &img, const float altitude_ratio) {

    cv::Mat input_rgb = img.clone();
    // don't need too large image
    if (resize_image_) {
       cv::resize(input_rgb, input_rgb, cv::Size(0, 0),
                  0.5, 0.5, cv::INTER_AREA);
    }

     std::vector<cv::Mat> patch_feat;
     double max_response = -1.;
     cv::Mat max_response_map;
     cv::Point2i max_response_pt;
     int scale_index = 0;
     std::vector<double> scale_responses;

     int center[2];
     center[0] = std::floor(this->filter_size_.width / 2);
     center[1] = std::floor(this->filter_size_.height / 2);
     
     int sindex = static_cast<int>(this->scales_.size()) - 1;

     float *d_features = getFeaturesGPU(
        input_rgb, input_rgb, pose_.cx, pose_.cy, windows_size_[0],
        windows_size_[1], current_scale_ * scales_[sindex], true);
     
     int new_width = windows_size_[0] * current_scale_ * scales_[sindex];
     int new_height = windows_size_[1] * current_scale_ * scales_[sindex];

     for (int i = sindex; i >= 0; i--) {
        cv::Mat response = this->trackingProcessOnGPU(d_features);
        
        double min_val;
        double max_val;
        cv::Point2i min_loc;
        cv::Point2i max_loc;
        cv::minMaxLoc(response, &min_val, &max_val, &min_loc, &max_loc);

        double weight = this->scales_[i] < 1.0 ? this->scales_[i] :
           1.0/this->scales_[i];
        if (max_val * weight > max_response) {
           max_response = max_val * weight;
           max_response_map = response;
           max_response_pt = max_loc;
           scale_index = i;
        }
        
        scale_responses.push_back(max_val * weight);
        
        /* TODO: FOR VISUAL SCALE CHANGE DETECTION(INCOMPLETE) */
        bool is_process = false;
        if (i > 0 && sindex > 1 && is_process) {
           float scale_factor = scales_[i-1]/scales_[i];
           
           new_width *= scale_factor;
           new_height *= scale_factor;

           //! 1) interpolate original
           //! 2) estimate
           
           float *d_output;
           int outsize = new_width * new_height * sizeof(float) * FILTER_BATCH_;
           cudaMalloc(reinterpret_cast<void**>(&d_output), outsize);
           spatialFeaturePyramidGPU(&d_output, d_features, center,
                                    new_width, new_height,
                                    this->filter_size_.width,
                                    this->filter_size_.height, FILTER_BATCH_,
                                    FILTER_SIZE_);

           bilinearInterpolationGPU(
              &d_features, d_output, this->filter_size_.width,
              this->filter_size_.height, new_width, new_height,
              (new_width * new_height * FILTER_BATCH_), FILTER_BATCH_);
           
           cudaFree(d_output);
        }
     }
     cudaFree(d_features);
     
     int wsize = 10;
     
     //! outter outterrect
     cv::Rect_<int> out_rect;
     out_rect.x = max_response_pt.x - wsize/2;
     out_rect.x = out_rect.x < 0 ? 0 : out_rect.x;
     out_rect.y = max_response_pt.y - wsize/2;
     out_rect.y = out_rect.y < 0 ? 0 : out_rect.y;
     out_rect.width = wsize;
     out_rect.width -= out_rect.x + out_rect.width > max_response_map.cols ?
                       out_rect.x + out_rect.width - max_response_map.cols : 0;
     out_rect.height = wsize;
     out_rect.height -= out_rect.y + out_rect.height > max_response_map.rows ?
        out_rect.y + out_rect.height - max_response_map.rows : 0;
          
     cv::Mat roi = max_response_map(out_rect).clone();

     float mean = 0.0;
     int icount = 0;
     for (int j = 0; j < roi.rows; j++) {
        for (int i = 0; i < roi.cols; i++) {
           if ((i < max_response_pt.x - 1 || i > max_response_pt.x + 1) ||
               (j < max_response_pt.y - 1 || j > max_response_pt.y + 1)) {
              mean += roi.at<float>(j, i);
              icount++;
           }
        }
     }
     mean /= static_cast<float>(icount);
    
     float demean = 0.0;
     for (int j = 0; j < roi.rows; j++) {
        for (int i = 0; i < roi.cols; i++) {
           if ((i < max_response_pt.x - 1 || i > max_response_pt.x + 1) ||
               (j < max_response_pt.y - 1 || j > max_response_pt.y + 1)) {
              demean += std::pow((roi.at<float>(j, i) - mean), 2);
           }
        }
     }
     float stddev = std::sqrt(demean / static_cast<float>(icount));
     double psr_ratio = (max_response - static_cast<double>(mean)) /
        static_cast<double>(stddev);

     
     std::cout << "\033[35mPSR RATION: " << psr_ratio << "\t"
               << "\033[0m\n";
     
     // sub pixel quadratic interpolation from neighbours
     // wrap around to negative half-space of vertical axis
     if (max_response_pt.y > max_response_map.rows / 2) {
        max_response_pt.y = max_response_pt.y - max_response_map.rows;
     }
     // same for horizontal axis
     if (max_response_pt.x > max_response_map.cols / 2) {
        max_response_pt.x = max_response_pt.x - max_response_map.cols;
     }
     
     cv::Point2f new_location(max_response_pt.x, max_response_pt.y);

     if (use_subpixel_localization_) {
         new_location = subPixelPeak(max_response_pt, max_response_map);
     }
     
     this->pose_.cx += current_scale_*cell_size_*new_location.x;
     this->pose_.cy += current_scale_*cell_size_*new_location.y;
     this->pose_.cx = (pose_.cx < 0) ? 0.0 : pose_.cx;
     this->pose_.cx = (pose_.cx > img.cols - 1) ? img.cols - 1 : pose_.cx;
     this->pose_.cy = (pose_.cy < 0) ? 0.0 : pose_.cy;
     this->pose_.cy = (pose_.cy > img.rows - 1) ? img.rows - 1 : pose_.cy;

     
     // sub grid scale interpolation
     double new_scale = scales_[scale_index];
     if (use_subgrid_scale_) {
        new_scale = subGridScale(scale_responses, scale_index);
     }

     //! uav altitude based height estimation
     current_scale_ *= new_scale;
     current_scale_ *= static_cast<double>(altitude_ratio);

     current_scale_ = (current_scale_ < min_max_scale_[0]) ?
        min_max_scale_[0] : current_scale_;
     current_scale_ = (current_scale_ > min_max_scale_[1]) ?
        min_max_scale_[1] : current_scale_;

     //! DRN for scale estimation
     if (this->is_drn_set_ && this->use_drn_) {
        cv::Mat im_plot = img.clone();
        
        if (!this->prev_img_.empty() && this->scale_momentum_ > 0.0f &&
            this->prev_img_.size() == img.size()) {

           //! padd previous box slightly
           int b_pad = 0;
           cv::Rect_<int> temp_rect = getBBox().getRect();
           if (b_pad > 0) {
              temp_rect.x = (temp_rect.x - b_pad < 0) ? 0 : temp_rect.x - b_pad;
              temp_rect.y = (temp_rect.y - b_pad < 0) ? 0 : temp_rect.y - b_pad;
              temp_rect.width += (b_pad * 2);
              temp_rect.height += (b_pad * 2);
              temp_rect.width -= (temp_rect.br().x + b_pad > img.cols) ?
                 temp_rect.br().x - img.cols : 0;
              temp_rect.height -= (temp_rect.br().y + b_pad > img.rows) ?
                 temp_rect.br().y - img.rows : 0;
           }
           
           cv::Rect_<int> box_estimate;
           this->regression_net_->correspondance(
              box_estimate, img, temp_rect, this->prev_img_, this->prev_rect_);

           //! check growth rate
           float rate = static_cast<float>(box_estimate.area()) /
              static_cast<float>(getBBox().getRect().area());

           bool is_process = false;
           // if (rate > 1.0f / this->scale_momentum_ &&
           //     rate < this->scale_momentum_)
           {
              is_process = true;
              this->pose_.w = box_estimate.br().x - box_estimate.tl().x;
              this->pose_.h = box_estimate.br().y - box_estimate.tl().y;
              this->pose_.cx = box_estimate.tl().x + this->pose_.w/2.;
              this->pose_.cy = box_estimate.tl().y + this->pose_.h/2.;
           }
           
           if (this->resize_image_ && is_process) {
              // this->pose_.cx = this->pose_.cx * 2;
              // this->pose_.cy = this->pose_.cy * 2;
              this->pose_.scale(0.5);
           }
        }
        //! update previous info
        this->prev_img_ = img.clone();
        this->prev_rect_ = this->getBBox().getRect();
     }
     
     // TODO(UPDATE): update the tracker online
     if (is_update_model_ && psr_ratio > this->psr_update_thresh_) {

        multiplyComplexByScalarGPU(&d_complex_, this->interp_factor_,
                                   FILTER_SIZE_ * FILTER_BATCH_);
        multiplyComplexByScalarGPU(&dev_model_xf_, 1.0f - this->interp_factor_,
                                   FILTER_SIZE_ * FILTER_BATCH_);
        addComplexGPU(&dev_model_xf_, this->d_complex_,
                      FILTER_SIZE_ * FILTER_BATCH_);
        cudaDeviceSynchronize();

        //! using already assigned memory(d_kzf_) to save time
        multiplyComplexGPU(&d_kzf_, this->dev_p_yf_, this->d_kf_,
                           FILTER_SIZE_);  //! d_kzf = num

        multiplyComplexByScalarGPU(&d_kzf_, this->interp_factor_,
                                   FILTER_SIZE_);
        multiplyComplexByScalarGPU(&dev_model_alphaf_num_,
                                   1.0f - this->interp_factor_,
                                   FILTER_SIZE_);
        addComplexGPU(&dev_model_alphaf_num_, d_kzf_, FILTER_SIZE_);
        cudaDeviceSynchronize();

        addComplexByScalarGPU(&d_f2c1_, this->d_kf_, this->lambda_,
                              FILTER_SIZE_);
        multiplyComplexGPU(&d_f2c1_, this->d_kf_, this->d_f2c1_,
                           FILTER_SIZE_);  //! d_f2c1_ = deno
        multiplyComplexByScalarGPU(&d_f2c1_, this->interp_factor_,
                                   FILTER_SIZE_);
        multiplyComplexByScalarGPU(&dev_model_alphaf_den_,
                                   1.0f - this->interp_factor_,
                                   FILTER_SIZE_);
        addComplexGPU(&dev_model_alphaf_den_, this->d_f2c1_,
                      FILTER_SIZE_);
        cudaDeviceSynchronize();

        divisionComplexGPU(&dev_model_alphaf_, this->dev_model_alphaf_num_,
                           this->dev_model_alphaf_den_, FILTER_SIZE_);

     }

     //! for redetecting lost or drifted target
     if (psr_ratio < this->psr_detect_thresh_ && this->detect_lost_target_) {
        ROS_WARN("REDETECTING LOST TARGET");

        cv::Rect_<int> box;
        if (this->redetectTarget(box, img, this->prev_img_, this->prev_rect_)) {
           this->pose_.w = box.br().x - box.tl().x;
           this->pose_.h = box.br().y - box.tl().y;
           this->pose_.cx = box.tl().x + this->pose_.w/2.;
           this->pose_.cy = box.tl().y + this->pose_.h/2.;
        }
     }

}

float* KernelizedCorrelationFiltersGPU::getFeaturesGPU(
    cv::Mat & input_rgb, cv::Mat & input_gray, int cx, int cy,
    int size_x, int size_y, double scale, bool interpolated) {
    int size_x_scaled = std::floor(size_x * scale);
    int size_y_scaled = std::floor(size_y * scale);

    cv::Mat patch_rgb = getSubwindow(input_rgb, cx, cy,
                                     size_x_scaled, size_y_scaled);

    boost::shared_ptr<caffe::Blob<float> > blob_info(new caffe::Blob<float>);
    this->feature_extractor_->getFeatures(patch_rgb, filter_size_);
    this->feature_extractor_->getNamedBlob(blob_info, "conv5");
    
    //! formula: caffe ==>>> blob->cpu_data() + blob->offset(n);
    const float *d_data = blob_info->gpu_data();
    if (interpolated) {
       float *d_resized_data = bilinearInterpolationGPU(
          d_data, filter_size_.width, filter_size_.height, blob_info->width(),
          blob_info->height(), blob_info->count(), FILTER_BATCH_);
       return d_resized_data;
    } else {
       return const_cast<float*>(d_data);
    }
}

cv::Mat KernelizedCorrelationFiltersGPU::trackingProcessOnGPU(
    float *d_features) {
    const int data_lenght = FILTER_SIZE_ * FILTER_BATCH_;

    cosineConvolutionGPU(&d_cos_conv_, d_features, this->d_cos_window_,
                         data_lenght);

    convertFloatToComplexGPU(&d_f2c_, d_cos_conv_, FILTER_BATCH_, FILTER_SIZE_);
    cuDFT(&d_complex_, d_f2c_, handle_, FILTER_BATCH_, FILTER_SIZE_);
    
    float xf_norm_gpu = squaredNormGPU(&d_squared_norm_, &d_summation_,
                                        d_complex_, FILTER_BATCH_,
                                        FILTER_SIZE_);
    cudaDeviceSynchronize();
    
    float yf_norm_gpu = squaredNormGPU(&d_squared_norm_, &d_summation_,
                                       dev_model_xf_, FILTER_BATCH_,
                                       FILTER_SIZE_);
    cudaDeviceSynchronize();
    
    complexConjuateGPU(&d_conj_mxf, this->dev_model_xf_,
                       FILTER_BATCH_, FILTER_SIZE_);
    cudaDeviceSynchronize();
    
    multiplyComplexGPU(&d_xyf_, d_complex_, d_conj_mxf,
                       FILTER_BATCH_ * FILTER_SIZE_);
    cudaDeviceSynchronize();

    cuInvDFT(&d_ifft_, d_xyf_, handle_, FILTER_BATCH_, FILTER_SIZE_);
    cudaDeviceSynchronize();
    
    invFFTSumOverFiltersGPU(&d_xysum_, d_ifft_, FILTER_BATCH_, FILTER_SIZE_);
    cudaDeviceSynchronize();

    float normalizer = 1.0f / (static_cast<float>(data_lenght));
    cuGaussianExpGPU(d_xysum_, xf_norm_gpu, yf_norm_gpu,
                     kernel_sigma_, normalizer, FILTER_SIZE_);
    cudaDeviceSynchronize();
    
    convertFloatToComplexGPU(&d_f2c1_, d_xysum_, 1, FILTER_SIZE_);
    cuDFT(&d_kf_, d_f2c1_, cufft_handle1_, 1, FILTER_SIZE_);
    cudaDeviceSynchronize();
    
    multiplyComplexGPU(&d_kzf_, this->dev_model_alphaf_, d_kf_, FILTER_SIZE_);
    cudaDeviceSynchronize();
    
    cuInvDFT(&d_ifft1_, d_kzf_, cufft_handle1_, 1, FILTER_SIZE_);
    cudaDeviceSynchronize();

    float odata[FILTER_SIZE_];
    int OUT_BYTE = FILTER_SIZE_ * sizeof(float);
    cudaMemcpy(odata, d_ifft1_, OUT_BYTE, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    cv::Mat results = cv::Mat(filter_size_.height, filter_size_.width,
                              CV_32F);
    
    for (int i = 0; i < filter_size_.height; i++) {
       for (int j = 0; j < filter_size_.width; j++) {
          results.at<float>(i, j) = odata[j + (i * filter_size_.width)];
       }
    }
    
    return results;
}


cv::Mat KernelizedCorrelationFiltersGPU::gaussianShapedLabels(
    double sigma, int dim1, int dim2) {
    cv::Mat labels(dim2, dim1, CV_32FC1);
    int range_y[2] = {-dim2 / 2, dim2 - dim2 / 2};
    int range_x[2] = {-dim1 / 2, dim1 - dim1 / 2};

    double sigma_s = sigma*sigma;
    for (int y = range_y[0], j = 0; y < range_y[1]; ++y, ++j) {
       float * row_ptr = labels.ptr<float>(j);
       double y_s = y*y;
       for (int x = range_x[0], i = 0; x < range_x[1]; ++x, ++i) {
          row_ptr[i] = std::exp(-0.5 * (y_s + x*x) / sigma_s);
       }
    }

    // rotate so that 1 is at top-left corner (see KCF paper for explanation)
    cv::Mat rot_labels = circularShift(labels, range_x[0], range_y[0]);
    // sanity check, 1 at top left corner
    assert(rot_labels.at<float>(0, 0) >= 1.f - 1e-10f);

    return rot_labels;
}

cv::Mat KernelizedCorrelationFiltersGPU::circularShift(
     const cv::Mat &patch, int x_rot, int y_rot) {
     cv::Mat rot_patch(patch.size(), CV_32FC1);
     cv::Mat tmp_x_rot(patch.size(), CV_32FC1);

     // circular rotate x-axis
     if (x_rot < 0) {
         // move part that does not rotate over the edge
        cv::Range orig_range(-x_rot, patch.cols);
        cv::Range rot_range(0, patch.cols - (-x_rot));
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(),
                                                             rot_range));

         // rotated part
         orig_range = cv::Range(0, -x_rot);
         rot_range = cv::Range(patch.cols - (-x_rot), patch.cols);
         patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(),
                                                              rot_range));
     } else if (x_rot > 0) {
        // move part that does not rotate over the edge
        cv::Range orig_range(0, patch.cols - x_rot);
        cv::Range rot_range(x_rot, patch.cols);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(),
                                                             rot_range));

         // rotated part
        orig_range = cv::Range(patch.cols - x_rot, patch.cols);
        rot_range = cv::Range(0, x_rot);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(),
                                                             rot_range));
     } else {  // zero rotation
         // move part that does not rotate over the edge
        cv::Range orig_range(0, patch.cols);
        cv::Range rot_range(0, patch.cols);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(),
                                                             rot_range));
     }

     // circular rotate y-axis
     if (y_rot < 0) {
        // move part that does not rotate over the edge
        cv::Range orig_range(-y_rot, patch.rows);
        cv::Range rot_range(0, patch.rows - (-y_rot));
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(
           rot_patch(rot_range, cv::Range::all()));

        // rotated part
        orig_range = cv::Range(0, -y_rot);
        rot_range = cv::Range(patch.rows - (-y_rot), patch.rows);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(
           rot_patch(rot_range, cv::Range::all()));
     } else if (y_rot > 0) {
        // move part that does not rotate over the edge
        cv::Range orig_range(0, patch.rows - y_rot);
        cv::Range rot_range(y_rot, patch.rows);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(
           rot_patch(rot_range, cv::Range::all()));

         // rotated part
        orig_range = cv::Range(patch.rows - y_rot, patch.rows);
        rot_range = cv::Range(0, y_rot);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(
           rot_patch(rot_range, cv::Range::all()));
     } else {  // zero rotation
        // move part that does not rotate over the edge
        cv::Range orig_range(0, patch.rows);
        cv::Range rot_range(0, patch.rows);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(
           rot_patch(rot_range, cv::Range::all()));
     }

     return rot_patch;
}

/* hann window actually (Power-of-cosine windows)
 */
cv::Mat KernelizedCorrelationFiltersGPU::cosineWindowFunction(
    int dim1, int dim2) {
    cv::Mat m1(1, dim1, CV_32FC1), m2(dim2, 1, CV_32FC1);
    double n_inv = 1.0f / (static_cast<double>(dim1)-1.);
    for (int i = 0; i < dim1; ++i) {
       m1.at<float>(i) = 0.5f * (
          1.0f - std::cos(2.0f * CV_PI * static_cast<double>(i) * n_inv));
    }
    n_inv = 1./ (static_cast<double>(dim2)-1.);
    for (int i = 0; i < dim2; ++i) {
       m2.at<float>(i) = 0.5f * (
          1.0f - std::cos(2.0f * CV_PI * static_cast<double>(i) * n_inv));
    }
    cv::Mat ret = m2*m1;
    return ret;
}

/* Returns sub-window of image input centered at [cx, cy] coordinates),
 * with size [width, height]. If any pixels are outside of the image,
 * they will replicate the values at the borders.
 */
cv::Mat KernelizedCorrelationFiltersGPU::getSubwindow(
    const cv::Mat &input, int cx, int cy, int width, int height) {
    cv::Mat patch;
    int x1 = cx - width/2;
    int y1 = cy - height/2;
    int x2 = cx + width/2;
    int y2 = cy + height/2;

    // out of image
    if (x1 >= input.cols || y1 >= input.rows || x2 < 0 || y2 < 0) {
       patch.create(height, width, input.type());
       patch.setTo(0.f);
       return patch;
    }

    int top = 0, bottom = 0, left = 0, right = 0;

    // fit to image coordinates, set border extensions;
    if (x1 < 0) {
       left = -x1;
       x1 = 0;
    }
    if (y1 < 0) {
       top = -y1;
       y1 = 0;
    }
    if (x2 >= input.cols) {
       right = x2 - input.cols + width % 2;
       x2 = input.cols;
    } else {
       x2 += width % 2;
    }
    if (y2 >= input.rows) {
       bottom = y2 - input.rows + height % 2;
       y2 = input.rows;
    } else {
       y2 += height % 2;
    }
    if (x2 - x1 == 0 || y2 - y1 == 0) {
       patch = cv::Mat::zeros(height, width, CV_32FC1);
    } else {
       cv::copyMakeBorder(input(cv::Range(y1, y2),
                                cv::Range(x1, x2)), patch,
                          top, bottom, left, right, cv::BORDER_REPLICATE);
    }

    // sanity check
    assert(patch.cols == width && patch.rows == height);
    return patch;
}

float KernelizedCorrelationFiltersGPU::getResponseCircular(
    cv::Point2i &pt, cv::Mat & response) {
    int x = pt.x;
    int y = pt.y;
    if (x < 0)
        x = response.cols + x;
    if (y < 0)
        y = response.rows + y;
    if (x >= response.cols)
        x = x - response.cols;
    if (y >= response.rows)
        y = y - response.rows;
    return response.at<float>(y, x);
}

cv::Point2f KernelizedCorrelationFiltersGPU::subPixelPeak(
    cv::Point & max_loc, cv::Mat & response) {
    // find neighbourhood of max_loc (response is circular)
    // 1 2 3
    // 4   5
    // 6 7 8
    cv::Point2i p1(max_loc.x-1, max_loc.y-1),
       p2(max_loc.x, max_loc.y-1), p3(max_loc.x+1, max_loc.y-1);
    cv::Point2i p4(max_loc.x-1, max_loc.y),
       p5(max_loc.x+1, max_loc.y);
    cv::Point2i p6(max_loc.x-1, max_loc.y+1),
       p7(max_loc.x, max_loc.y+1), p8(max_loc.x+1, max_loc.y+1);

    // fit 2d quadratic function f(x, y) = a*x^2 + b*x*y + c*y^2 + d*x
    // + e*y + f
    cv::Mat A = (cv::Mat_<float>(9, 6) <<
                 p1.x*p1.x, p1.x*p1.y, p1.y*p1.y, p1.x, p1.y, 1.f,
                 p2.x*p2.x, p2.x*p2.y, p2.y*p2.y, p2.x, p2.y, 1.f,
                 p3.x*p3.x, p3.x*p3.y, p3.y*p3.y, p3.x, p3.y, 1.f,
                 p4.x*p4.x, p4.x*p4.y, p4.y*p4.y, p4.x, p4.y, 1.f,
                 p5.x*p5.x, p5.x*p5.y, p5.y*p5.y, p5.x, p5.y, 1.f,
                 p6.x*p6.x, p6.x*p6.y, p6.y*p6.y, p6.x, p6.y, 1.f,
                 p7.x*p7.x, p7.x*p7.y, p7.y*p7.y, p7.x, p7.y, 1.f,
                 p8.x*p8.x, p8.x*p8.y, p8.y*p8.y, p8.x, p8.y, 1.f,
                 max_loc.x*max_loc.x, max_loc.x*max_loc.y,
                 max_loc.y*max_loc.y, max_loc.x, max_loc.y, 1.f);
    cv::Mat fval = (cv::Mat_<float>(9, 1) <<
                    getResponseCircular(p1, response),
                    getResponseCircular(p2, response),
                    getResponseCircular(p3, response),
                    getResponseCircular(p4, response),
                    getResponseCircular(p5, response),
                    getResponseCircular(p6, response),
                    getResponseCircular(p7, response),
                    getResponseCircular(p8, response),
                    getResponseCircular(max_loc, response));
    cv::Mat x;
    cv::solve(A, fval, x, cv::DECOMP_SVD);

    double a = x.at<float>(0), b = x.at<float>(1), c = x.at<float>(2),
           d = x.at<float>(3), e = x.at<float>(4);

    cv::Point2f sub_peak(max_loc.x, max_loc.y);
    if (b > 0 || b < 0) {
        sub_peak.y = ((2.f * a * e) / b - d) / (b - (4 * a * c) / b);
        sub_peak.x = (-2 * c * sub_peak.y - e) / b;
    }

    return sub_peak;
}

double KernelizedCorrelationFiltersGPU::subGridScale(
    std::vector<double> & responses, int index) {
    cv::Mat A, fval;
    if (index < 0 || index > static_cast<int>(scales_.size()) - 1) {
        // interpolate from all values
        // fit 1d quadratic function f(x) = a*x^2 + b*x + c
        A.create(scales_.size(), 3, CV_32FC1);
        fval.create(scales_.size(), 1, CV_32FC1);
        for (size_t i = 0; i < scales_.size(); ++i) {
            A.at<float>(i, 0) = scales_[i] * scales_[i];
            A.at<float>(i, 1) = scales_[i];
            A.at<float>(i, 2) = 1;
            fval.at<float>(i) = responses[i];
        }
    } else {
       // only from neighbours
       if (index == 0 || index == static_cast<int>(scales_.size()) - 1) {
            return scales_[index];
       }
       A = (cv::Mat_<float>(3, 3) <<
            scales_[index-1] * scales_[index-1], scales_[index-1], 1,
            scales_[index] * scales_[index], scales_[index], 1,
            scales_[index+1] * scales_[index+1], scales_[index+1], 1);
       fval = (cv::Mat_<float>(3, 1) << responses[index-1],
               responses[index], responses[index+1]);
    }

    cv::Mat x;
    cv::solve(A, fval, x, cv::DECOMP_SVD);
    double a = x.at<float>(0), b = x.at<float>(1);
    double scale = scales_[index];
    if (a > 0 || a < 0)
        scale = -b / (2 * a);
    return scale;
}

cufftComplex* KernelizedCorrelationFiltersGPU::cuDFT(
    float *dev_data, const cufftHandle handle,
    const int FILTER_BATCH, const int FILTER_SIZE) {

    if (FILTER_BATCH == 0 || FILTER_SIZE == 0) {
       ROS_ERROR("[cuDFT]: SIZE UNDEFINED");
       cufftComplex empty[1];
       return empty;
    }
    cufftComplex *d_input = convertFloatToComplexGPU(
       dev_data, FILTER_BATCH, FILTER_SIZE);
    cufftComplex *d_output = cuFFTC2Cprocess(
       d_input, handle, FILTER_SIZE, FILTER_BATCH);
    cudaFree(d_input);
    return d_output;
}

float* KernelizedCorrelationFiltersGPU::cuInvDFT(
    cufftComplex *d_complex, const cufftHandle handle,
    const int FILTER_BATCH, const int FILTER_SIZE) {
    float *d_real_data = invcuFFTC2CProcess(d_complex, handle,
                                         FILTER_SIZE, FILTER_BATCH, true);
    return d_real_data;
}

/**
 * memory reusing for tx1
 */

void KernelizedCorrelationFiltersGPU::cuDFT(
    cufftComplex **d_output, cufftComplex *d_input,
    const cufftHandle handle,
    const int FILTER_BATCH, const int FILTER_SIZE) {

    if (FILTER_BATCH == 0 || FILTER_SIZE == 0) {
       ROS_ERROR("[cuDFT]: SIZE UNDEFINED");
       return;
    }
    cuFFTC2Cprocess(d_output, d_input, handle, FILTER_SIZE, FILTER_BATCH);
}

void KernelizedCorrelationFiltersGPU::cuInvDFT(
    float **d_real_data, cufftComplex *d_complex,
    const cufftHandle handle,
    const int FILTER_BATCH, const int FILTER_SIZE) {
    invcuFFTC2CProcess(d_real_data, d_complex, handle,
                       FILTER_SIZE, FILTER_BATCH, true);
}


bool KernelizedCorrelationFiltersGPU::redetectTarget(
    cv::Rect_<int> &object_rect, const cv::Mat in_img, const cv::Mat p_img,
    const cv::Rect_<int> p_rect) {
    if (in_img.empty() || p_img.empty()) {
       ROS_ERROR("[::redetectTarget]: EMPTY INPUTS");
       return false;
    }

    //! downsize for fast computation
    cv::Mat src_img;
    cv::Mat prev_img;
    cv::resize(in_img, src_img, cv::Size(in_img.cols/2, in_img.rows/2));
    cv::resize(p_img, prev_img, cv::Size(in_img.cols/2, in_img.rows/2));
    cv::Rect_<int> prev_rect = p_rect;
    prev_rect.x /= 2;
    prev_rect.y /= 2;
    prev_rect.width /= 2;
    prev_rect.height /= 2;
    
    const std::string blob_name = "conv1";
    
    //! get features
    cv::Size blob_size;
    int batch;
    int count;
    const float *d_features = this->vectorizedCNNCodes(
       blob_size, batch, count, src_img, blob_name);
    cudaDeviceSynchronize();
    float *d_fsum = invFFTSumOverFiltersGPU(d_features, batch,
                                           blob_size.width * blob_size.height);
    cudaDeviceSynchronize();
    
    float *d_data = bilinearInterpolationGPU(
       d_fsum, src_img.cols, src_img.rows,
       blob_size.width, blob_size.height, count/batch, 1);
    cudaDeviceSynchronize();
    
    // find the max factor on blas
    size_t lenght = src_img.cols * src_img.rows;
    int max_index = 0;
    this->cublas_status_ = cublasIsamax(this->cublas_handle_, lenght,
                                  d_data, 1, &max_index);
    max_index -= 1;
    cudaDeviceSynchronize();

    normalizeByFactorInArrayGPU(d_data, max_index, 1, lenght);
    cudaDeviceSynchronize();
    
    //! ----------------------------------
    /*
    int byte = src_img.cols * src_img.rows * sizeof(float);
    float *output = reinterpret_cast<float*>(std::malloc(byte));
    cudaMemcpy(output, d_data, byte, cudaMemcpyDeviceToHost);
    
    cv::Mat img1 = cv::Mat::zeros(src_img.size(), CV_32F);
    for (int i = 0; i < img1.rows; i++) {
       for (int j = 0; j < img1.cols; j++) {
          img1.at<float>(i, j) = output[j + (i * img1.cols)];
       }
    }
    // cv::resize(img1, img1, cv::Size(448, 448));
    cv::imshow("filter", img1);

    cv::Mat img2;
    cv::integral(img1, img2);
    std::cout << img2  << "\n";
    
    cv::Mat pimg = prev_img(prev_rect);
    cv::imshow("prev", pimg);
    return -1;
    */
    //! ----------------------------------


    bool proposal;
    // proposal = this->lostp_->lostTrargetDetection(
    //    object_rect, prev_img, prev_rect, d_data, src_img, d_data);
    if (proposal) {
       object_rect.x *= 2;
       object_rect.y *= 2;
       object_rect.width *= 2;
       object_rect.height *= 2;
    }
    
    cudaFree(d_fsum);
    cudaFree(d_data);

    return proposal;
}


const float* KernelizedCorrelationFiltersGPU::vectorizedCNNCodes(
    cv::Size &blob_size, int &batch, int &count,
    const cv::Mat in_rgb, const std::string blob_name) {
    boost::shared_ptr<caffe::Blob<float> > blob_info(new caffe::Blob<float>);
    this->feature_extractor_->getFeatures(in_rgb, in_rgb.size());
    this->feature_extractor_->getNamedBlob(blob_info, blob_name);
    const float *d_features = blob_info->gpu_data();
    blob_size = cv::Size(blob_info->width(), blob_info->height());
    batch = FILTER_BATCH_;
    count = blob_info->count();
    return d_features;
}

void KernelizedCorrelationFiltersGPU::filterVisualization(
    const float *d_features, const cv::Size filter_size) {
    float *output = reinterpret_cast<float*>(malloc(BYTE_));
    cudaMemcpy(output, d_features, BYTE_, cudaMemcpyDeviceToHost);

    const int sq_dim = std::ceil(std::sqrt(FILTER_BATCH_));
    // cv::Size filter_size = filter_size_;
    cv::Mat all_filters = cv::Mat::zeros(filter_size.height * sq_dim,
                                         filter_size.width * sq_dim, CV_8UC1);

    int y = 0;
    int x = 0;
    int sidex = 0;
    int sidey = 0;
    for (int k = 0; k < FILTER_BATCH_; k++) {
       for (int j = 0; j < filter_size.height; j++) {
          for (int i = 0; i < filter_size.width; i++) {
             int ind = i + (j * filter_size.width) +
                (k * filter_size.height * filter_size.width);
             all_filters.at<uchar>(y, x++) = output[ind];
          }
                 
          if (x > (filter_size.width * (sidex+1)) - 1) {
             x = (filter_size.width * sidex);
             y++;
             if (y > (filter_size.height * (sidey+1)) - 1) {
                y = (filter_size.height * sidey);
             }
          }
       }
       sidex++;
       if (sidex > sq_dim - 1) {
          sidex = 0;
          sidey++;
          if (sidey > sq_dim -1) {
             sidey = 0;
          }
       }
       x = (filter_size.width * sidex);
       y = (filter_size.height * sidey);
    }
    cv::applyColorMap(all_filters, all_filters, cv::COLORMAP_JET);
    cv::namedWindow("filters", CV_WINDOW_NORMAL);
    cv::imshow("filters", all_filters);
    // cv::waitKey(0);
}

bool KernelizedCorrelationFiltersGPU::parseParamsFromFile(
    const std::string filename) {
    boost::shared_ptr<cv::FileStorage> files(new cv::FileStorage);
    files->open(filename, cv::FileStorage::READ);
    if (!files->isOpened()) {
       ROS_WARN("HYPERPARAMETER TUNING FILE NOT FOUND!");
       ROS_WARN("TRACKER WILL BE INIT WILL DEFAULT PARAMETERS!");
       return false;
    }

    cv::FileNode n = files->operator[]("Learning");
    this->lambda_ = static_cast<float>(n["learning_rate"]);
    this->interp_factor_ = static_cast<float>(n["interpolation_factor"]);
    this->cell_size_ = static_cast<int>(n["cell_size"]);
    
    n = files->operator[]("Scale");
    this->scale_step_ = static_cast<double>(n["scale_step"]);
    this->num_scales_ = static_cast<int>(n["pyramid_levels"]);
    this->use_scale_ = static_cast<int>(n["use_scale"]) != 0 ? true : false;
    this->use_subgrid_scale_ = static_cast<int>(n["subgrid_scale"]) != 0 ?
       true : false;
    this->scale_momentum_ = static_cast<float>(n["scale_momentum"]);
    
    n = files->operator[]("Localization");
    this->resize_image_ = static_cast<int>(n["resize_image"]) != 0 ?
       true : false;
    this->use_subpixel_localization_ =
       static_cast<int>(n["subpixel_localization"]) != 0 ? true: false;

    n = files->operator[]("Redetection");
    this->is_update_model_ = static_cast<int>(n["update_template"]) != 0 ?
       true : false;
    this->psr_update_thresh_ = static_cast<double>(n["update_threshold"]);
    this->detect_lost_target_ = static_cast<int>(n["detect_lost_target"]) != 0 ?
       true : false;
    this->psr_detect_thresh_ = static_cast<double>(n["detection_threshold"]);
    this->num_proposals_ = static_cast<int>(n["box_proposals"]);
    this->use_max_boxes_ = static_cast<int>(n["generate_max_boxes"]) != 0 ?
       true : false;

    //! hyperparams for reg net
    n = files->operator[]("RegressionNet");
    this->use_drn_ = static_cast<int>(n["use_regression_net"]) != 0 ?
       true : false;
    
    return true;
}

KernelizedCorrelationFiltersGPU::~KernelizedCorrelationFiltersGPU() {
    cudaFree(this->d_cos_window_);
    cudaFree(this->dev_model_alphaf_);
    cudaFree(this->dev_model_alphaf_num_);
    cudaFree(this->dev_model_alphaf_den_);
    cudaFree(this->dev_model_xf_);
    cudaFree(this->dev_p_yf_);
    cufftDestroy(this->cufft_handle1_);
    cufftDestroy(this->handle_);

    cudaFree(this->d_cos_conv_);
    cudaFree(this->d_conj_mxf);
    cudaFree(this->d_summation_);
    cudaFree(this->d_squared_norm_);
    cudaFree(this->d_xyf_);
    cudaFree(this->d_kzf_);
    cudaFree(this->d_xysum_);
    cudaFree(this->d_ifft_);
    cudaFree(this->d_ifft1_);
    cudaFree(this->d_complex_);
}
