#include "../../include/core/DescriptorExtractor.h"


namespace reglib
{

DescriptorExtractorORB::DescriptorExtractorORB(){
    debugg_lvl = 0;
    orb = cv::ORB(500,1.2, 8, 21, 0,2, cv::ORB::HARRIS_SCORE, 21);}
DescriptorExtractorORB::~DescriptorExtractorORB(){}

std::vector<KeyPoint> DescriptorExtractorORB::extract(RGBDFrame * frame, ModelMask * mask){
    double startTime = 0;
    if(debugg_lvl > 0){startTime = getTime();}


    std::vector<KeyPoint> kps;

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb(frame->rgb, cv::Mat(), keypoints, descriptors);

    //uint64_t xordata [4];
    const uint64_t * data = (uint64_t *)(descriptors.data);

    unsigned char  * rgbdata		= (unsigned char	*)(frame->rgb.data);
    unsigned short * depthdata		= (unsigned short	*)(frame->depth.data);
    float		   * normalsdata	= (float			*)(frame->normals.data);

    const unsigned int width	= frame->camera->width;
    const unsigned int height	= frame->camera->height;
    const float idepth			= frame->camera->idepth_scale;
    const float cx				= frame->camera->cx;
    const float cy				= frame->camera->cy;
    const float ifx				= 1.0/frame->camera->fx;
    const float ify				= 1.0/frame->camera->fy;

    unsigned int nr_keypoints = keypoints.size();
    for(unsigned int i = 0; i < nr_keypoints; i++){
        double w = keypoints[i].pt.x;
        double h = keypoints[i].pt.y;

        unsigned int ind = int(h+0.5) * width + int(0.5+w);

        float z		= idepth*float(depthdata[ind]);

        if(z > 0){
            superpoint sp;

            sp.r = rgbdata[3*ind+0];
            sp.g = rgbdata[3*ind+1];
            sp.b = rgbdata[3*ind+2];

            sp.x = (w - cx) * z * ifx;
            sp.y = (h - cy) * z * ify;
            sp.z = z;

            sp.nx = normalsdata[3*ind+0];
            sp.ny = normalsdata[3*ind+1];
            sp.nz = normalsdata[3*ind+1];

            sp.point_information = getInformation(z);
            sp.normal_information = 0;
            if(sp.nx > 1){sp.normal_information = sp.point_information;}
            sp.colour_information = sp.point_information;
            kps.push_back(KeyPoint(sp,Descriptor(data[4*i+0],data[4*i+1],data[4*i+2],data[4*i+3])));
        }
    }

    if(debugg_lvl > 0){printf("Keypoint extraction time: %5.5fs -> %i keypoints found\n",getTime()-startTime,kps.size());}
    if(debugg_lvl > 1){
        cv::Mat img = frame->rgb.clone();
        for(unsigned int i = 0; i < nr_keypoints; i++){
            double w = keypoints[i].pt.x;
            double h = keypoints[i].pt.y;
            if(depthdata[int(h+0.5) * width + int(0.5+w)] > 0){ cv::circle( img,keypoints[i].pt,5,cv::Scalar( 0, 255, 0 ),2);}
        }
        cv::namedWindow( "KeyPoints"	, cv::WINDOW_AUTOSIZE );
        cv::imshow( "KeyPoints",	img );
        cv::waitKey(0);
    }

    return kps;
}

}

