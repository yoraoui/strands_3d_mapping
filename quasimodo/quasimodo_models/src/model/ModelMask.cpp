#include "model/ModelMask.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "highgui.h"

namespace reglib
{

int ModelMask_id = 0;

ModelMask::ModelMask(cv::Mat mask_, std::string label_){
	mask = mask_;
	label = label_;
	sweepid = -1;

	id = ModelMask_id++;
	using namespace cv;

	unsigned char * maskdata = (unsigned char *)mask.data;

	width = 640;
	height = 480;
	maskvec = new bool[width*height];
    for(int i = 0; i < width*height; i++){maskvec[i] = maskdata[i] != 0;}



//	for(unsigned int w = 1; w < 639; w++){
//		for(unsigned int h = 1; h < 479; h++){
//			if(maskdata[h*640+w] != 0){
//				testw.push_back(w);
//				testh.push_back(h);
//			}
//		}
//	}

//	cv::Mat combined;
//	combined.create(height,width,CV_8UC3);
//	unsigned char * combineddata = (unsigned char *)combined.data;
//	for(int j = 0; j < 3*width*height; j++){combineddata[j] = 0;}
	//cv::Mat combined = mask.clone();


	int step = 0;
	for(int w = step; w < 639-step; w++){
		for(int h = step; h < 479-step; h++){
			if(maskdata[h*640+w] != 0){
//				combineddata[3*(h*640+w) + 0] = 0;
//				combineddata[3*(h*640+w) + 1] = 0;
//				combineddata[3*(h*640+w) + 2] = 255;
//				printf("%i %i-> ",w,h);
				bool isok =  true;
				for(int w2 = w-step; w2 <= w+step; w2++){
					for(int h2 = h-step; h2 <= h+step; h2++){
//						printf("(%i %i , %i -> %i) ",w2,h2,isok,maskdata[h2*640+w2] != 0);
						isok = isok && (maskdata[h2*640+w2] != 0);
					}
				}
//				printf("\n");
//				printf("\n");
//				printf("\n");
//				printf("\n");
				if(isok){
//					combineddata[3*(h*640+w) + 0] = 0;
//					combineddata[3*(h*640+w) + 1] = 255;
//					combineddata[3*(h*640+w) + 2] = 0;
					testw.push_back(w);
					testh.push_back(h);
				}
			}
		}
	}

//	cv::imshow( "combined", combined );
//	cv::waitKey(0);


//exit(0);
	for(unsigned int i = 0; i < testw.size(); i++){
		int ind = rand() % testw.size();
		int tw = testw[i];
		int th = testh[i];
		testw[i] = testw[ind];
		testh[i] = testh[ind];
		testw[ind] = tw;
		testh[ind] = th;
	}
}

cv::Mat ModelMask::getMask(){
	cv::Mat fullmask;
	fullmask.create(height,width,CV_8UC1);
	unsigned char * maskdata = (unsigned char *)fullmask.data;
    for(int j = 0; j < width*height; j++){maskdata[j] = 255*maskvec[j];}
	return fullmask;
}

ModelMask::~ModelMask(){delete[] maskvec;}


}

