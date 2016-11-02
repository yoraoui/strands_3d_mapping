#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sensor_msgs/PointCloud2.h>
#include <string.h>

#include <pcl_ros/point_cloud.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_ros/transforms.h>

#include <pcl_ros/point_cloud.h>
#include <cv_bridge/cv_bridge.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_ros/transforms.h>

#include <sys/time.h>
#include <sys/resource.h>

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <iostream>

int thickness = 3;

std::vector< cv::Mat > imgs;
int state = 0;

double corner1_x = 0;
double corner1_y = 0;

double corner2_x = 0;
double corner2_y = 0;

double corner3_x = 0;
double corner3_y = 0;

double corner4_x = 0;
double corner4_y = 0;
void CallBackFunc(int event, int x, int y, int flags, void* userdata){
	if			( event == cv::EVENT_LBUTTONDOWN ){
		std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		if(state == 0){
			state = 1;
			corner1_x = x;
			corner1_y = y;
		}else if(state == 1){
			state = 2;

			if(x > corner1_x){
				corner2_x = x;
			}else{
				corner2_x = corner1_x;
				corner1_x = x;
			}

			if(y > corner1_y){
				corner2_y = y;
			}else{
				corner2_y = corner1_y;
				corner1_y = y;
			}
		}else if(state == 2){
			state = 3;
			corner3_x = x;
			corner3_y = y;
		}else if(state == 3){
			state = 4;
			corner4_x = x;
			corner4_y = y;
		}
	}else if	( event == cv::EVENT_RBUTTONDOWN ){
		//std::cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}else if	( event == cv::EVENT_MBUTTONDOWN ){
		//std::cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}else if	( event == cv::EVENT_MOUSEMOVE ){
		//std::cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;
	}

	if(state == 5){return;}
	//printf("state: %i\n",state);
	if(state >= 1){
		cv::Mat m = imgs.front().clone();

		if(state >= 2){
			cv::Rect ROI1(cv::Point(corner1_x,corner1_y),cv::Point(corner2_x,corner2_y));
			cv::rectangle(m, ROI1 , cv::Scalar(0,0,255),thickness);

			//			cv::line(m, cv::Point(x,		y),				cv::Point(corner1_x,	corner1_y),	cv::Scalar(0,255,0),thickness);
			//			cv::line(m, cv::Point(x,		y),				cv::Point(corner2_x,	corner2_y),	cv::Scalar(0,255,0),thickness);

			if(state >= 3){

				double ratio = fabs(corner1_x-corner2_x)/fabs(corner1_y-corner2_y);
				//printf("ratio: %f\n",ratio);
				double offsets = fabs(std::max(fabs(corner3_x-x)/ratio,fabs(corner3_y-y)));
				offsets = std::max(offsets,1.0);


				double c1x = corner3_x+offsets*ratio;
				double c1y = corner3_y+offsets;
				double c2x = corner3_x-offsets*ratio;
				double c2y = corner3_y-offsets;

				double midx = 0.5*(corner1_x+corner2_x);
				double midy = 0.5*(corner1_y+corner2_y);



				cv::Point a = cv::Point(c2x,	c2y);
				cv::Point b = cv::Point(corner1_x,	corner1_y);

//				if( c2x < std::min(corner1_x,corner2_x )){// && c1y < std::min(corner1_y,corner2_y )){
//					a.x = c2x;
//					a.y = c1y;
//				}

//				if		(c2x < corner1_x){
//					a.y = cv::Point(c2x,	c1y);
////					cv::line(m, cv::Point(c2x,	c1y),	cv::Point(corner1_x,	corner1_y),	cv::Scalar(0,255,0),thickness);
////				}else if(c2x > corner1_x && c2y < corner1_y){
////					cv::line(m, cv::Point(c2x,	c2y),	cv::Point(corner1_x,	corner1_y),	cv::Scalar(255,0,0),thickness);
//				}else{
//					//cv::line(m, cv::Point(c2x,	c2y),	cv::Point(corner1_x,	corner1_y),	cv::Scalar(0,0,255),thickness);
//					a = cv::Point(c2x,	c2y);
//				}


				//cv::line(m, cv::Point(c2x,	c1y),	cv::Point(corner1_x,	corner1_y),	cv::Scalar(0,255,0),thickness);
				cv::line(m, a,	b,	cv::Scalar(0,255,0),thickness);
				cv::line(m, cv::Point(c1x,	c1y),	cv::Point(corner2_x,	corner2_y),	cv::Scalar(0,0,255),thickness);

				cv::Rect ROI2(cv::Point(c1x,c1y),cv::Point(c2x,	c2y));
				cv::rectangle(m, ROI2 , cv::Scalar(255,0,255),thickness);

				if(state >= 4){
					state = 5;
					cv::Mat localimg = imgs.front()(ROI1);

					cv::Mat outimg;
					outimg.create(ROI2.height, ROI2.width,CV_8UC3);
					cv::resize(localimg, outimg, outimg.size(),0,0,cv::INTER_AREA);

					outimg.copyTo(m(ROI2));
					cv::rectangle(m, ROI2 , cv::Scalar(0,0,255),thickness);

					cv::imshow("Image", m);
					cv::waitKey(0);
				}
			}else{
				cv::line(m, cv::Point(x,		y),				cv::Point(corner1_x,	corner1_y),	cv::Scalar(0,255,0),thickness);
				cv::line(m, cv::Point(x,		y),				cv::Point(corner2_x,	corner2_y),	cv::Scalar(0,255,0),thickness);
			}
		}else{
			cv::Rect ROI1(cv::Point(corner1_x,corner1_y),cv::Point(x,y));
			cv::rectangle(m, ROI1 , cv::Scalar(255,0,255),thickness);
		}
		cv::imshow("Image", m);
		cv::waitKey(0);
	}
}

int main(int argc, char** argv){

	for(int i = 1; i < argc;i++){
		printf("input: %s\n",argv[i]);
		cv::Mat m = cv::imread(argv[i], CV_LOAD_IMAGE_UNCHANGED);
		if(m.data ){
			cv::namedWindow( "Image", cv::WINDOW_AUTOSIZE );
			cv::imshow( "Image", m );
			cv::waitKey(30);
			imgs.push_back(m);
		}
	}

	if(imgs.size() > 0){
		cv::namedWindow("Image", 1);
		cv::setMouseCallback("Image", CallBackFunc, NULL);
		cv::imshow("Image", imgs.front());
		cv::waitKey(0);
	}

	return 0;
}
