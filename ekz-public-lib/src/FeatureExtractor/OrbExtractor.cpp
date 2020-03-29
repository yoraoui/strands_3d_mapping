#include "OrbExtractor.h"
#include "OrbFeatureDescriptor.h"
#include <ctime>
#include <algorithm>
#include "cv.h"
#include "highgui.h"
#include <opencv2/opencv.hpp>

#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include "/home/younes/Bureau/Pointer2Labs/PFE/catkin_semantic/src/ekz-public-lib/include/mygeometry/Point2.h"
#include "/home/younes/Bureau/Pointer2Labs/PFE/catkin_semantic/src/ekz-public-lib/include/mygeometry/KeyPoint2.h"
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

using namespace std;
//bool comparison_orb (cv::KeyPoint  *i,cv::KeyPoint  *j) { return (i->stabilety>j->stabilety); }

OrbExtractor::OrbExtractor(){ nr_features = 500;}

OrbExtractor::~OrbExtractor(){}


KeyPointSet * OrbExtractor::getKeyPointSet(FrameInput * fi){

	struct timeval start, end;
	gettimeofday(&start, NULL);

	IplImage * rgb_img = fi->get_rgb_img();
	//cv::ORB orb = cv::ORB(nr_features,1.2f, 8, 3, 0,2, cv::ORB::HARRIS_SCORE, 31);
	
	Ptr<cv::Feature2D> orb = ORB::create(nr_features);
	cv::Mat img=cv::cvarrToMat(rgb_img);
	
	cv::Mat desc1;
	vector <cv::KeyPoint> kp1;

	Ptr<FeatureDetector> detector = ORB::create();
    	detector->detect(img, kp1);
    	Ptr<DescriptorExtractor> extractor = ORB::create();
    	extractor->compute(img, kp1, desc1 );

	
	KeyPointSet * keypoints = new KeyPointSet();

	for(unsigned int i = 0; i < kp1.size();i++){
		cv::KeyPoint curr = kp1.at(i);
		int * desc = new int[32];
		for(int j = 0; j < 32; j++){desc[j] = (int)desc1.at<uchar>(i,j);}
		FeatureDescriptor * descriptor = new OrbFeatureDescriptor(desc);

		int w = curr.pt.x+0.5;
		int h = curr.pt.y+0.5;

		float r,g,b,x,y,z;
		fi->getRGB(r,g,b,curr.pt.x,curr.pt.y);
		fi->getXYZ(x,y,z,curr.pt.x,curr.pt.y);
		
		KeyPoint2 * kp = new KeyPoint2();
		kp->w = curr.pt.x;
		kp->h = curr.pt.y;
		kp->stabilety = curr.response;
		kp->descriptor = descriptor;

		kp->r = r;
		kp->g = g;
		kp->b = b;
		if(z > 0 && !isnan(z)){
			kp->valid = true;
			kp->index_number = keypoints->valid_key_points.size();
			//keypoints->valid_key_points.push_back(kp);
			kp->point = new Point2(x,y,z,curr.pt.x,curr.pt.y);
			kp->point = new Point2(x,y,z,curr.pt.x,curr.pt.y);
		}else{
			kp->valid = false;
			kp->index_number = keypoints->invalid_key_points.size();
			//keypoints->invalid_key_points.push_back(kp);
			kp->point = new Point2(-x,-y,-z,curr.pt.x,curr.pt.y);
		}
	}

	//sort(keypoints->valid_key_points.begin(),keypoints->valid_key_points.end(),comparison_orb);
	//sort(keypoints->invalid_key_points.begin(),keypoints->invalid_key_points.end(),comparison_orb);

	gettimeofday(&end, NULL);
	double time = (end.tv_sec*1000000+end.tv_usec-(start.tv_sec*1000000+start.tv_usec))/1000000.0f;
	if(verbose){printf("Extracted %i orb features in %fs\n",(int)kp1.size(),time);}
	return keypoints;
}