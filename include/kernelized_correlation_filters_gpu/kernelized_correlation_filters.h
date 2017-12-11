// Copyright (C) 2016 by Krishneel Chaudhary @ JSK Lab,
// The University of Tokyo, Japan

#pragma once
#ifndef _KERNELIZED_CORRELATION_FILTERS_h_
#define _KERNELIZED_CORRELATION_FILTERS_h_

#include <kernelized_correlation_filters_gpu/fast_maths_kernel.h>
#include <kernelized_correlation_filters_gpu/crop_feature_space.h>
#include <kernelized_correlation_filters_gpu/loss_target_pursuit.h>
#include <kernelized_correlation_filters_gpu/deep_feature_extraction.h>
#include <kernelized_correlation_filters_gpu/cosine_convolution_kernel.h>
#include <kernelized_correlation_filters_gpu/gaussian_correlation_kernel.h>
#include <kernelized_correlation_filters_gpu/bilinear_interpolation_kernel.h>
#include <kernelized_correlation_filters_gpu/spatial_feature_pyramid_kernel.h>
#include <kernelized_correlation_filters_gpu/discrete_fourier_transform_kernel.h>
#include <kernelized_correlation_filters_gpu/tracking_target_detector.h>

struct BoundingBox {
    double cx, cy, w, h;
    inline void scale(double factor) {
        cx *= factor;
        cy *= factor;
        w  *= factor;
        h  *= factor;
    }
    inline cv::Rect getRect() {
        return cv::Rect(cx - w/2.0, cy - h/2.0, w, h);
    }
};

class KernelizedCorrelationFiltersGPU {
   
 private:
    BoundingBox pose_;
    bool resize_image_;

    double padding_;
    double output_sigma_factor_;
    double output_sigma_;
    double kernel_sigma_;
    float lambda_;
    float interp_factor_;
    int cell_size_;
    cv::Mat cos_window_;
    int num_scales_;
    double scale_step_;
    double current_scale_;
    std::string uav_name_;

    int windows_size_[2];
    double min_max_scale_[2];
    std::vector<double> scales_;

    cufftComplex *dev_p_yf_;
    cufftComplex *dev_model_alphaf_;
    cufftComplex *dev_model_alphaf_num_;
    cufftComplex *dev_model_alphaf_den_;
    cufftComplex *dev_model_xf_;
   
    //! alloc reusable mem
    float *d_cos_conv_;
    cufftComplex *d_conj_mxf;
    float *d_summation_;
    float *d_squared_norm_;
    cufftComplex *d_kzf_;
    cufftComplex *d_xyf_;
    float *d_xysum_;
    cufftComplex *d_f2c1_;
    cufftComplex *d_f2c_;
    cufftComplex *d_kf_;
    float *d_ifft_;
    float *d_ifft1_;
    cufftComplex *d_complex_;

    bool use_scale_;
    bool use_subpixel_localization_;
    bool use_subgrid_scale_;
    bool is_cnn_set_;

    //! updating params
    bool is_update_model_;
    bool detect_lost_target_;
    double psr_update_thresh_;
    double psr_detect_thresh_;
    int num_proposals_;
    bool use_max_boxes_;

    //! redetection params
    float similarity_thresh_;
    float iou_thresh_;
    std::vector<int> psr_hist_;
    int psr_counter_;
   
    cv::Mat gaussianShapedLabels(double sigma, int dim1, int dim2);
    cv::Mat circularShift(const cv::Mat & patch, int x_rot, int y_rot);
    cv::Mat cosineWindowFunction(int dim1, int dim2);
    cv::Point2f subPixelPeak(cv::Point &, cv::Mat &);
    double subGridScale(std::vector<double> &, int index = -1);
    float getResponseCircular(cv::Point2i &, cv::Mat &);
    cv::Mat getSubwindow(const cv::Mat & input, int cx, int cy,
                         int size_x, int size_y);

    bool detector_state_;
   
 protected:

    boost::shared_ptr<FeatureExtractor> feature_extractor_;
    boost::shared_ptr<LostTargetPursuit> lostp_;
    boost::shared_ptr<caffe::Blob<float> > blob_info_;
   
    int FILTER_SIZE_;  //! size of cnn codes
    int FILTER_BATCH_;  //! batch size
    float *d_cos_window_;
    int BYTE_;
    cv::Size filter_size_;

    //! handle with batch
    cufftHandle cufft_handle1_;
    cufftHandle handle_;

    //! handle for cublas
    cublasHandle_t cublas_handle_;
    cublasStatus_t cublas_status_;
   
    cv::Mat prev_img_;
    cv::Rect_<int> prev_rect_;
   
 public:

    KernelizedCorrelationFiltersGPU(const std::string = std::string(),
                                    const std::string = "S900");
    ~KernelizedCorrelationFiltersGPU();
   
    void init(cv::Mat &, const cv::Rect &);
    void setTrackerPose(BoundingBox &, cv::Mat &);
    void updateTrackerPosition(BoundingBox &);
    bool track(cv::Mat &, const float = 1.0f);
    BoundingBox getBBox();
    void setCaffeInfo(const std::string, const std::string, const std::string,
                      std::vector<std::string> &, const int);
    cufftComplex* cuDFT(float *, const cufftHandle, const int, const int);
    float* cuInvDFT(cufftComplex *, const cufftHandle, const int, const int);
    float* getFeaturesGPU(cv::Mat &, cv::Mat &, int, int, int,
                          int, double, bool = true);
    cv::Mat trackingProcessOnGPU(float *);
    void cuDFT(cufftComplex **, cufftComplex  *, const cufftHandle,
               const int, const int);
    void cuInvDFT(float **, cufftComplex *, const cufftHandle,
                  const int, const int);
    void filterVisualization(const float*, const cv::Size);
    const float* vectorizedCNNCodes(cv::Size &, int &, int &,
                                    const cv::Mat, const std::string = "conv5");
    bool redetectTarget(cv::Rect_<int> &, const cv::Mat, const cv::Mat,
                        const cv::Rect_<int>);
    bool parseParamsFromFile(const std::string);
    bool switchRedetection(const bool);


    //! todo hide this
    boost::shared_ptr<TrackingTargetDetector> detector_;
    bool redetectTarget();
};

#endif  /*_KERNELIZED_CORRELATION_FILTERS_h_*/
