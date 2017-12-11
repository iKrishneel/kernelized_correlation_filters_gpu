
#pragma once
#ifndef _LOST_TARGET_PURSUIT_H_
#define _LOST_TARGET_PURSUIT_H_

#include <uav_target_tracking/object_proposal_generation.h>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaobjdetect.hpp>

class LostTargetPursuit {

 private:

    cv::Ptr<cv::cuda::HOG> hog_;
    cv::cuda::GpuMat d_templ_;
    cv::Size hog_size_;
   
    float iou_thresh_;
    float match_thresh_;

    int num_proposals_;
    bool max_box_only_;
   
 protected:
    boost::shared_ptr<ObjectProposalGenerator> opg_;
   
 public:
    LostTargetPursuit(const cv::Size, const int = 5, const float = 0.5f,
                      const float = 0.5f, bool = false);
    bool lostTrargetDetection(cv::Rect_<int> &, const cv::Mat,
                                 const cv::Rect_<int>, const float *,
                                 const cv::Mat, const float *);
    void lostTargetProposals(std::vector<cv::Rect_<int> > &,
                            const cv::Mat, const float*);
    cv::Mat stitchImagePatches(const cv::Mat, const cv::Size,
                               const std::vector<cv::Rect_<int> >);
    float jaccardCoeff(const cv::Rect_<int>, const cv::Rect_<int>);
    void groupBoxesBasedOnIOU(std::vector<cv::Rect_<int> > &, const int);
   

    //! hog
    float similarityScore(const cv::Mat, const cv::Mat);
    void computeSimilarity(std::vector<cv::Rect_<int> > &, const cv::Mat,
                              const cv::Mat);
    cv::cuda::GpuMat getDescriptor(cv::cuda::GpuMat);
};

#endif /* _LOST_TARGET_PURSUIT_H_ */
