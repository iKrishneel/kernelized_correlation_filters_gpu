
#pragma once
#ifndef _OBJECT_PROPOSAL_GENERATION_H_
#define _OBJECT_PROPOSAL_GENERATION_H_

#include <ros/ros.h>
#include <ros/console.h>

#include <uav_target_tracking/gslicr.h>
#include <uav_target_tracking/summed_area_table.h>

#include <boost/foreach.hpp>
#include <opencv2/opencv.hpp>

class ObjectProposalGenerator {

 private:
    typedef gSLICr::UChar4Image Image;
   
    gSLICr::objects::settings gslicr_setting_;
    gSLICr::engines::core_engine* gslicr_engine_;

    Image* input_image_;
    Image* output_image_;

    std::vector<cv::Rect_<int> > box_proposals_;
    std::map<int, std::vector<cv::Point2f> > superpixel_points_;
   
    bool is_normalize_;
    bool max_box_only_;
    int num_proposals_;
    float cutoff_thresh_;

    
 protected:
    const int MIN_CLUSTER_SIZE_;

    float *d_sweights_;
    int *d_scounts_;
   
 public:
    ObjectProposalGenerator(const int = 610, const int = 610,
                            const int = 200, const int = 32/2,
                            const int = 3, const int = 20);
    std::vector<cv::Rect_<int> > generateObjectProposal(cv::Mat, const float *,
                                                        const int, bool = true,
                                                        bool = false);
    void getBoundingRects(std::vector<cv::Rect_<int> > &, const cv::Mat,
                          const std::map<int, std::vector<cv::Point2f> > &);
    void rankBoxProposals(std::vector<cv::Rect_<int> > &,
                          const cv::Mat, const cv::Mat,
                          const std::vector<cv::Rect_<int> >);
    void warpBoxProposals(std::vector<float> &,
                          std::vector<cv::Rect_<int> > &,
                          const cv::Mat,
                          const cv::Rect_<int>, const cv::Size,
                          const int, const float = 1.10f);
    template<class T, class U>
    U getBoxScores(const cv::Mat, const cv::Rect_<int>, bool);
   
    template <typename T>
    std::vector<int> sortIndexes(const std::vector<T> &);
    void load_image(const cv::Mat &, gSLICr::UChar4Image*);
    void load_image(const gSLICr::UChar4Image*, cv::Mat &);
   
};


#endif /* _OBJECT_PROPOSAL_GENERATION_H_ */
