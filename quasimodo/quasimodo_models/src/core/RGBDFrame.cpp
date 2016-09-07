#include "core/RGBDFrame.h"

#include <pcl/console/parse.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/supervoxel_clustering.h>

#include <vtkPolyLine.h>

#include <iostream>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>

#include <vector>
#include <string>

#include <cv.h>
#include <highgui.h>
#include <fstream>

#include <ctime>

#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include "organized_edge_detection.hpp"
#include <pcl/features/integral_image_normal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/PCLPointCloud2.h>


namespace reglib
{
unsigned long RGBDFrame_id_counter;
RGBDFrame::RGBDFrame(){
    id = RGBDFrame_id_counter++;
    capturetime = 0;
    pose = Eigen::Matrix4d::Identity();
    keyval = "";
}

bool updated = true;
void on_trackbar( int, void* ){updated = true;}

//void update(cv::Mat & err, cv::Mat & est, cv::Mat & dx, int dir){
//}

float pred(float targetW, float targetH, int sourceW, int sourceH, int width, float * est, float * dx, float * dy){
    int ind = sourceH * width + sourceW;
    return est[ind]+(targetW-float(sourceW))*dx[ind]+(targetH-float(sourceH))*dy[ind];
}

float weightFunc(int w0, int h0, int w1, int h1, int width, std::vector<cv::Mat> & est, std::vector<cv::Mat> & dx, std::vector<cv::Mat> & dy){
    float weight = 1;
    for(int c = 0; c < est.size(); c++){
        float error = ((float*)(est[c].data))[h0*width+w0] - pred(w0,h0,w1,h1,width, (float*)(est[c].data), (float*)(dx[c].data), (float*)(dy[c].data));
        weight *= exp(-0.5*error*error/(0.1*0.1));
    }
    return weight;
}


float weightFunc(int w0, int h0, int w1, int h1, int width, std::vector<float*> & est, std::vector<float*> & dx, std::vector<float*> & dy){
    float weight = 1;
    for(int c = 0; c < est.size(); c++){
        float error = est[c][h0*width+w0] - pred(w0,h0,w1,h1,width, est[c], dx[c], dy[c]);
        weight *= exp(-0.5*error*error/(0.1*0.1));
    }
    return weight;
}

std::vector< std::vector<double> > getImageProbs(reglib::RGBDFrame * frame, int blursize = 5){
	cv::Mat src						= frame->rgb.clone();
	unsigned short * depthdata		= (unsigned short	*)(frame->depth.data);
	float		   * normalsdata	= (float			*)(frame->normals.data);
	const float idepth				= frame->camera->idepth_scale;

	cv::GaussianBlur( src, src, cv::Size(blursize,blursize), 0, 0, cv::BORDER_DEFAULT );

	unsigned char * srcdata = (unsigned char *)src.data;
	unsigned int width = src.cols;
	unsigned int height = src.rows;

	std::vector< std::vector<double> > probs;
	unsigned int chans = 3;
	for(unsigned int c = 0; c < chans;c++){
		std::vector<double> Xvec;
		int dir;
		for(unsigned int w = 1; w < width-1;w++){
			for(unsigned int h = 1; h < height-1;h++){
				int ind = h*width+w;
				dir		= 1;
				Xvec.push_back(fabs(srcdata[chans*ind+c] - srcdata[chans*(ind-dir)+c]));

				dir		= width;
				Xvec.push_back(fabs(srcdata[chans*ind+c] - srcdata[chans*(ind-dir)+c]));
			}
		}

		Eigen::MatrixXd X = Eigen::MatrixXd::Zero(1,Xvec.size());
		for(unsigned int i = 0; i < Xvec.size();i++){X(0,i) = Xvec[i];}

		double stdval = 0;
		for(unsigned int i = 0; i < Xvec.size();i++){stdval += X(0,i)*X(0,i);}
		stdval = sqrt(stdval/double(Xvec.size()));

		DistanceWeightFunction2PPR2 * func = new DistanceWeightFunction2PPR2();
		func->zeromean				= true;
		func->maxp					= 0.99;
		func->startreg				= 0.0;
		func->debugg_print			= false;
		//func->bidir					= true;
		func->maxd					= 256.0;
		func->histogram_size		= 256;
		func->fixed_histogram_size	= true;
		func->startmaxd				= func->maxd;
		func->starthistogram_size	= func->histogram_size;
		func->blurval				= 0.005;
		func->stdval2				= stdval;
		func->maxnoise				= stdval;
		func->reset();
		func->computeModel(X);

		std::vector<double> dx;  dx.resize(width*height);	for(unsigned int i = 0; i < width*height;i++){dx[i] = 0.5;}
		std::vector<double> dy;	 dy.resize(width*height);	for(unsigned int i = 0; i < width*height;i++){dy[i] = 0.5;}
		std::vector<double> dxy; dxy.resize(width*height);	for(unsigned int i = 0; i < width*height;i++){dxy[i] = 0.5;}
		std::vector<double> dyx; dyx.resize(width*height);	for(unsigned int i = 0; i < width*height;i++){dyx[i] = 0.5;}
		for(unsigned int w = 1; w < width;w++){
			for(unsigned int h = 1; h < height-1;h++){
				int ind = h*width+w;

				dir		= 1;
				dx[ind] = func->getProb(fabs(srcdata[chans*ind+c] - srcdata[chans*(ind-dir)+c]));

				dir		= width;
				dy[ind] = func->getProb(fabs(srcdata[chans*ind+c] - srcdata[chans*(ind-dir)+c]));
			}
		}
		delete func;

		probs.push_back(dx);
		probs.push_back(dy);
	}

	{
		std::vector<double> Xvec;
		for(unsigned int w = 1; w < width-1;w++){
			for(unsigned int h = 1; h < height-1;h++){
				int ind = h*width+w;
				float z = idepth*float(depthdata[ind]);

				if(w > 1){
					int dir = -1;
					int other2 = ind+2*dir;
					int other = ind+dir;

					float z3 = idepth*float(depthdata[other2]);
					float z2 = idepth*float(depthdata[other]);
					z2 = 2*z2-z3;

					if(z2 > 0 || z > 0){Xvec.push_back((z-z2)/(z*z+z2*z2));}
				}

				if(h > 1){
					int dir = -width;
					int other2 = ind+2*dir;
					int other = ind+dir;

					float z3 = idepth*float(depthdata[other2]);
					float z2 = idepth*float(depthdata[other]);
					z2 = 2*z2-z3;

					if(z2 > 0 || z > 0){Xvec.push_back((z-z2)/(z*z+z2*z2));}
				}
			}
		}

		Eigen::MatrixXd X = Eigen::MatrixXd::Zero(1,Xvec.size());
		for(unsigned int i = 0; i < Xvec.size();i++){X(0,i) = Xvec[i];}

		double stdval = 0;
		for(unsigned int i = 0; i < Xvec.size();i++){stdval += X(0,i)*X(0,i);}
		stdval = sqrt(stdval/double(Xvec.size()));

		DistanceWeightFunction2PPR2 * funcZ = new DistanceWeightFunction2PPR2();
		funcZ->zeromean				= true;
		funcZ->startreg				= 0.002;
		funcZ->debugg_print			= false;
		funcZ->bidir				= false;
		funcZ->maxp					= 0.999999;
		funcZ->maxd					= 0.1;
		funcZ->histogram_size		= 100;
		funcZ->fixed_histogram_size	= true;
		funcZ->startmaxd			= funcZ->maxd;
		funcZ->starthistogram_size	= funcZ->histogram_size;
		funcZ->blurval				= 0.5;
		funcZ->stdval2				= stdval;
		funcZ->maxnoise				= stdval;
		funcZ->reset();
		funcZ->computeModel(X);

		for(unsigned int w = 1; w < width-1;w++){
			for(unsigned int h = 1; h < height-1;h++){
				int ind = h*width+w;
				float z = idepth*float(depthdata[ind]);

				if(w > 1){
					int dir = -1;
					int other2 = ind+2*dir;
					int other = ind+dir;

					float z3 = idepth*float(depthdata[other2]);
					float z2 = idepth*float(depthdata[other]);
					z2 = 2*z2-z3;

					if(z2 > 0 || z > 0){Xvec.push_back((z-z2)/(z*z+z2*z2));}
				}

				if(h > 1){
					int dir = -width;
					int other2 = ind+2*dir;
					int other = ind+dir;

					float z3 = idepth*float(depthdata[other2]);
					float z2 = idepth*float(depthdata[other]);
					z2 = 2*z2-z3;

					if(z2 > 0 || z > 0){Xvec.push_back((z-z2)/(z*z+z2*z2));}
				}
			}
		}

		std::vector<double> dx;  dx.resize(width*height);	for(unsigned int i = 0; i < width*height;i++){dx[i] = 0.5;}
		std::vector<double> dy;	 dy.resize(width*height);	for(unsigned int i = 0; i < width*height;i++){dy[i] = 0.5;}

		for(unsigned int w = 1; w < width-1;w++){
			for(unsigned int h = 1; h < height-1;h++){
				int ind = h*width+w;
				float z = idepth*float(depthdata[ind]);

				if(w > 1){
					int dir = -1;
					int other2 = ind+2*dir;
					int other = ind+dir;


					float z3 = idepth*float(depthdata[other2]);
					float z2 = idepth*float(depthdata[other]);

					float dz = fabs(z-z2);

					if(z3 > 0){dz = std::min(float(dz),float(fabs(z- (2*z2-z3))));}

					if(z2 > 0 || z > 0){dx[ind] = funcZ->getProb(dz/(z*z+z2*z2));}
				}

				if(h > 1){
					int dir = -width;

					int other2 = ind+2*dir;
					int other = ind+dir;

					float z3 = idepth*float(depthdata[other2]);
					float z2 = idepth*float(depthdata[other]);
					float dz = fabs(z-z2);

					if(z3 > 0){dz = std::min(float(dz),float(fabs(z- (2*z2-z3))));}
					if(z2 > 0 || z > 0){dy[ind] = funcZ->getProb(dz/(z*z+z2*z2));}
				}
			}
		}

		delete funcZ;
		probs.push_back(dx);
		probs.push_back(dy);
	}
	return probs;
}

RGBDFrame * RGBDFrame::clone(){
	return new RGBDFrame(camera->clone(), rgb.clone(),depth.clone(),capturetime, pose, true);
}

RGBDFrame::RGBDFrame(Camera * camera_, cv::Mat rgb_, cv::Mat depth_, double capturetime_, Eigen::Matrix4d pose_, bool compute_normals){
    keyval = "";

    sweepid = -1;
    id = RGBDFrame_id_counter++;
    camera = camera_;
	rgb = rgb_;
	depth = depth_;
    capturetime = capturetime_;
    pose = pose_;

    IplImage iplimg = rgb_;
    IplImage* img = &iplimg;

    //printf("%s LINE:%i\n",__FILE__,__LINE__);

    int width = img->width;
    int height = img->height;
    int sz = height*width;
    const double idepth			= camera->idepth_scale;
    const double cx				= camera->cx;
    const double cy				= camera->cy;
    const double ifx			= 1.0/camera->fx;
    const double ify			= 1.0/camera->fy;

    //
    //	printf("camera: cx %5.5f cy %5.5f",cx,cy);
    //	printf(": fx %5.5f fy %5.5f",camera->fx,camera->fy);
    //	printf(": ifx %5.5f ify %5.5f\n",ifx,ify);

    connections.resize(1);
    connections[0].resize(1);
    connections[0][0] = 0;

    intersections.resize(1);
    intersections[0].resize(1);
    intersections[0][0] = 0;

    nr_labels = 1;
    labels = new int[width*height];
    for(int i = 0; i < width*height; i++){labels[i] = 0;}


    //printf("%s LINE:%i\n",__FILE__,__LINE__);
    unsigned short * depthdata = (unsigned short *)depth.data;
	unsigned char * rgbdata = (unsigned char *)rgb.data;

//	rgbdata = 0;

//	rgbdata = new float[3*width*height];
//	for(int i = 0; i < 3*width*height; i++){
//		rgbdata[i]	=  float(img_rgbdata[i]);
//	}


	//std::vector< std::vector<double> > probs = getImageProbs(this,5);

	cv::Mat de;
	de.create(height,width,CV_32FC3);
	float * dedata = (float*)de.data;
	for(int i = 0; i < 3*width*height; i++){
		dedata[i] = 0;
	}

	cv::Mat ce;
	ce.create(height,width,CV_32FC3);
	float * cedata = (float*)ce.data;
	for(int i = 0; i < 2*width*height; i++){
		cedata[i] = 0;
	}

	std::vector<double> Xvec;

	for(unsigned int w = 1; w < width-1;w++){
		for(unsigned int h = 1; h < height-1;h++){
			int ind = h*width+w;
			float z = idepth*float(depthdata[ind]);

			if(w > 1){
				int dir = -1;
				int other2 = ind+2*dir;
				int other = ind+dir;

				float z3 = idepth*float(depthdata[other2]);
				float z2 = idepth*float(depthdata[other]);
				z2 = 2*z2-z3;

				if(z2 > 0 || z > 0){
					dedata[3*ind+1] = fabs((z-z2)/(z*z+z2*z2));
					Xvec.push_back(dedata[3*ind+1]);
				}
			}

			if(h > 1){
				int dir = -width;
				int other2 = ind+2*dir;
				int other = ind+dir;

				float z3 = idepth*float(depthdata[other2]);
				float z2 = idepth*float(depthdata[other]);
				z2 = 2*z2-z3;

				if(z2 > 0 || z > 0){
					dedata[3*ind+2] = fabs((z-z2)/(z*z+z2*z2));
					Xvec.push_back(dedata[3*ind+2]);
				}
			}
		}
	}



	Eigen::MatrixXd X = Eigen::MatrixXd::Zero(1,Xvec.size());
	for(unsigned int i = 0; i < Xvec.size();i++){X(0,i) = Xvec[i];}

	double stdval = 0;
	for(unsigned int i = 0; i < Xvec.size();i++){stdval += X(0,i)*X(0,i);}
	stdval = sqrt(stdval/double(Xvec.size()));

	DistanceWeightFunction2PPR2 * funcZ = new DistanceWeightFunction2PPR2();
	funcZ->zeromean				= true;
	funcZ->startreg				= 0.002;
	funcZ->debugg_print			= false;
	funcZ->bidir				= false;
	funcZ->maxp					= 0.999999;
	funcZ->maxd					= 0.1;
	funcZ->histogram_size		= 100;
	funcZ->fixed_histogram_size	= true;
	funcZ->startmaxd			= funcZ->maxd;
	funcZ->starthistogram_size	= funcZ->histogram_size;
	funcZ->blurval				= 0.5;
	funcZ->stdval2				= stdval;
	funcZ->maxnoise				= stdval;
	funcZ->reset();
	funcZ->computeModel(X);

	for(unsigned int w = 1; w < width-1;w++){
		for(unsigned int h = 1; h < height-1;h++){
			int ind = h*width+w;
			if(w > 1){
				dedata[3*ind+1] = 1-funcZ->getProb(dedata[3*ind+1]);
			}

			if(h > 1){
				dedata[3*ind+2] = 1-funcZ->getProb(dedata[3*ind+2]);
			}
		}
	}

	delete funcZ;

	cv::Mat det;
	det.create(height,width,CV_8UC1);
	unsigned char * detdata = (unsigned char*)det.data;
	for(int i = 0; i < width*height; i++){
		detdata[i] = 255*((dedata[3*i+1] > 0.5) || (dedata[3*i+2] > 0.5));
	}




    int dilation_size = 4;
	cv::dilate( det, det_dilate, getStructuringElement( cv::MORPH_RECT, cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ), cv::Point( dilation_size, dilation_size ) ) );
/*
	cv::Mat det2;
	det2.create(height,width,CV_8UC3);
	unsigned char * detdata2 = (unsigned char*)det2.data;
	for(int i = 0; i < width*height; i++){
		detdata2[3*i+0] = detdata2[3*i+1] = detdata2[3*i+2] = detdata[i];
	}

	int area = 5;
	for(unsigned int w = area; w < width-area;w++){
		for(unsigned int h = area; h < height-area;h++){
			bool p11 = detdata[h*width+w-0] != 0;
			if(!p11){continue;}

			int sum = 0;
			for(unsigned int x = w-area; x < w+area;x++){
				for(unsigned int y = h-area; y < h+area;y++){
					sum += detdata[y*width+x] != 0;
				}
			}

			if(sum <= area/2){
				cv::circle(det2, cv::Point(w,h), 5, cv::Scalar(0,255,0));
			}

		}
	}


	cv::namedWindow( "de", cv::WINDOW_AUTOSIZE );			cv::imshow( "de",	de);
	cv::namedWindow( "det", cv::WINDOW_AUTOSIZE );			cv::imshow( "det",	det);
	cv::namedWindow( "det2", cv::WINDOW_AUTOSIZE );			cv::imshow( "det",	det2);
	cv::namedWindow( "det_dilate", cv::WINDOW_AUTOSIZE );	cv::imshow( "det_dilate",	det_dilate);
	cv::namedWindow( "ce", cv::WINDOW_AUTOSIZE );			cv::imshow( "ce",	ce);
	cv::waitKey(0);
*/
/*
{


	for(unsigned int w = 1; w < width-1;w++){
		for(unsigned int h = 1; h < height-1;h++){
			int ind = h*width+w;
			float z = idepth*float(depthdata[ind]);

			if(w > 1){
				int dir = -1;
				int other2 = ind+2*dir;
				int other = ind+dir;

				float z3 = idepth*float(depthdata[other2]);
				float z2 = idepth*float(depthdata[other]);
				z2 = 2*z2-z3;

				if(z2 > 0 || z > 0){Xvec.push_back((z-z2)/(z*z+z2*z2));}
			}

			if(h > 1){
				int dir = -width;
				int other2 = ind+2*dir;
				int other = ind+dir;

				float z3 = idepth*float(depthdata[other2]);
				float z2 = idepth*float(depthdata[other]);
				z2 = 2*z2-z3;

				if(z2 > 0 || z > 0){Xvec.push_back((z-z2)/(z*z+z2*z2));}
			}
		}
	}

	std::vector<double> dx;  dx.resize(width*height);	for(unsigned int i = 0; i < width*height;i++){dx[i] = 0.5;}
	std::vector<double> dy;	 dy.resize(width*height);	for(unsigned int i = 0; i < width*height;i++){dy[i] = 0.5;}

	for(unsigned int w = 1; w < width-1;w++){
		for(unsigned int h = 1; h < height-1;h++){
			int ind = h*width+w;
			float z = idepth*float(depthdata[ind]);

			if(w > 1){
				int dir = -1;
				int other2 = ind+2*dir;
				int other = ind+dir;


				float z3 = idepth*float(depthdata[other2]);
				float z2 = idepth*float(depthdata[other]);

				float dz = fabs(z-z2);
				if(z3 > 0){dz = std::min(dz,float(fabs(z- (2*z2-z3))));}
				if(z2 > 0 || z > 0){dx[ind] = funcZ->getProb(dz/(z*z+z2*z2));}
			}

			if(h > 1){
				int dir = -width;

				int other2 = ind+2*dir;
				int other = ind+dir;

				float z3 = idepth*float(depthdata[other2]);
				float z2 = idepth*float(depthdata[other]);

				float dz = fabs(z-z2);
				if(z3 > 0){dz = std::min(dz,float(fabs(z- (2*z2-z3))));}
				if(z2 > 0 || z > 0){dy[ind] = funcZ->getProb(dz/(z*z+z2*z2));}
			}
		}
	}



	for(int i = 0; i < width*height; i++){
		dedata[3*i+0] = 1-dx[i];
		dedata[3*i+1] = 1-dy[i];
		dedata[2*i+2] = 0;
	}

}
*/
//	cv::namedWindow( "de", cv::WINDOW_AUTOSIZE );		cv::imshow( "de",	de);
//	cv::namedWindow( "ce", cv::WINDOW_AUTOSIZE );		cv::imshow( "ce",	ce);
//	cv::waitKey(0);



    depthedges.create(height,width,CV_8UC1);
    unsigned char * depthedgesdata = (unsigned char *)depthedges.data;
    for(int i = 0; i < width*height; i++){
        depthedgesdata[i] = 0;
    }	

	if(false){
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr	cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
        cloud->width	= width;
        cloud->height	= height;
        cloud->points.resize(width*height);

        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("3D Viewer"));
        viewer->setBackgroundColor (0, 0, 0);
        viewer->addCoordinateSystem (1.0);
        viewer->initCameraParameters ();

        float * zbase = new float[width*height];
        float * rbase = new float[width*height];
        float * gbase = new float[width*height];
        float * bbase = new float[width*height];

        float * zest = new float[width*height];
        float * rest = new float[width*height];
        float * gest = new float[width*height];
        float * best = new float[width*height];

		float * zdx = new float[width*height];
		float * rdx = new float[width*height];
		float * gdx = new float[width*height];
		float * bdx = new float[width*height];

		float * zdy = new float[width*height];
		float * rdy = new float[width*height];
		float * gdy = new float[width*height];
		float * bdy = new float[width*height];

        float * znext = new float[width*height];
        float * rnext = new float[width*height];
        float * gnext = new float[width*height];
        float * bnext = new float[width*height];

        for(int i = 0; i < width*height; i++){
			zbase[i]	= zest[i]	= idepth*double(depthdata[i]);
			bbase[i]	= best[i]	= float(rgbdata[3*i+0])/256.0;
			gbase[i]	= gest[i]	= float(rgbdata[3*i+1])/256.0;
			rbase[i]	= rest[i]	= float(rgbdata[3*i+2])/256.0;
			zdx[i]		= zdy[i]	= 0;
			rdx[i]		= rdy[i]	= 0;
			gdx[i]		= gdy[i]	= 0;
			bdx[i]		= bdy[i]	= 0;
        }

        for(int it = 0; true; it++){
			if(it % 50 == 0){



                for(int w = 0; w < width; w++){
                    for(int h = 0; h < height;h++){
                        int ind = h*width+w;
                        pcl::PointXYZRGBA & p = cloud->points[ind];
                        double z = zest[ind];
                        if(z > 0){
                            p.x = (double(w) - cx) * z * ifx;
                            p.y = (double(h) - cy) * z * ify;
                            p.z = z;
                        }else{
                            p.x = 0;
                            p.y = 0;
                            p.z = 0;
                        }
						p.b = best[ind]*256.0;
						p.g = gest[ind]*256.0;
						p.r = rest[ind]*256.0;
                    }
                }

                viewer->removeAllPointClouds();
                viewer->addPointCloud<pcl::PointXYZRGBA> (cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA>(cloud), "cloud");
                viewer->spin();

				//for(int w = 0; w < width; w++){
				for(int w = 0; w < width; w++){
					for(int h = 0; h < height;h++)
					{
						//int h= height/2;
						int ind = h*width+w;
						pcl::PointXYZRGBA & p = cloud->points[ind];

						double z1 = zest[ind];
						double x1 = (double(w) - cx) * z1 * ifx;
						double y1 = (double(h) - cy) * z1 * ify;
						Eigen::Vector3d point1 (x1,y1,z1);

						double z2 = zest[ind]+zdx[ind];
						double x2 = (double(w+1) - cx) * z2 * ifx;
						double y2 = (double(h) - cy) * z2 * ify;
						Eigen::Vector3d point2 (x2,y2,z2);

						double z3 = zest[ind]+zdy[ind];
						double x3 = (double(w) - cx) * z2 * ifx;
						double y3 = (double(h+1) - cy) * z2 * ify;
						Eigen::Vector3d point3 (x3,y3,z3);

						Eigen::Vector3d normal =  (point2-point1).cross(point3-point1);
						normal.normalize();

						float r = 0.5*(normal(0)+1.0);
						float g = 0.5*(normal(1)+1.0);
						float b = 0.5*(normal(2)+1.0);

						p.b = b*255.0;//(2.0*normal(2)-1.0)*255.0;
						p.g = g*255.0;//(2.0*normal(1)-1.0)*255.0;
						p.r = r*255.0;//(2.0*normal(0)-1.0)*255.0;

//						printf("%4.4i -> x1: %3.3f %3.3f %3.3f x2: %3.3f %3.3f %3.3f normal: %3.3f %3.3f %3.3f shifted: %3.3f %3.3f %3.3f rgb: %3.3i %3.3i %3.3i\n",w,point2(0)-point1(0),point2(1)-point1(1),point2(2)-point1(2),   point3(0)-point1(0),point3(1)-point1(1),point3(2)-point1(2),     normal(0),normal(1),normal(2),  0.5*(normal(0)+1.0),0.5*(normal(1)+1.0),0.5*(normal(2)+1.0),   p.r,p.g,p.b);

//						viewer->removeAllPointClouds();
//						viewer->addPointCloud<pcl::PointXYZRGBA> (cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA>(cloud), "cloud");
//						viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud");
//						viewer->spin();
					}
				}

				viewer->removeAllPointClouds();
				viewer->addPointCloud<pcl::PointXYZRGBA> (cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA>(cloud), "cloud");
				viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
				viewer->spin();

				for(int w = 0; w < width; w++){
					for(int h = 0; h < height;h++){
						int ind = h*width+w;
						pcl::PointXYZRGBA & p = cloud->points[ind];
						double z = zest[ind];
						if(z > 0){
							p.x = (double(w) - cx) * z * ifx;
							p.y = (double(h) - cy) * z * ify;
							p.z = z;
						}else{
							p.x = 0;
							p.y = 0;
							p.z = 0;
						}
						p.b = best[ind]*256.0;
						p.g = gest[ind]*256.0;
						p.r = rest[ind]*256.0;
					}
				}
            }


			float sumeL = 0;
			float sumeR = 0;
			float sumeU = 0;
			float sumeD = 0;

			float sumeLpred = 0;
			float sumeRpred = 0;
			float sumeUpred = 0;
			float sumeDpred = 0;
			float angleVar = 100*pow(0.97,it);
			printf("angleVar: %15.15f\n",angleVar);

            for(int w = 1; w < width-1; w++){
                for(int h = 1; h < height-1;h++){
                    int iM = h*width+w;
                    int iL = iM-1;
                    int iR = iM+1;
                    int iU = iM-width;
                    int iD = iM+width;

					//Z
					float zMbase = zbase[iM];
					float zM = zest[iM];
					float zL = zest[iL];
					float zR = zest[iR];
					float zU = zest[iU];
					float zD = zest[iD];

					float zMdx = zdx[iM];
					float zLdx = zdx[iL];
					float zRdx = zdx[iR];
					float zUdx = zdx[iU];
					float zDdx = zdx[iD];

					float zMdy = zdy[iM];
					float zLdy = zdy[iL];
					float zRdy = zdy[iR];
					float zUdy = zdy[iU];
					float zDdy = zdy[iD];

					float zLpred = zL-zLdx;
					float zRpred = zR+zRdx;
					float zUpred = zU-zUdy;
					float zDpred = zD+zDdy;

					//R
                    float rMbase = rbase[iM];
                    float rM = rest[iM];
                    float rL = rest[iL];
                    float rR = rest[iR];
                    float rU = rest[iU];
                    float rD = rest[iD];

					//G
                    float gMbase = gbase[iM];
                    float gM = gest[iM];
                    float gL = gest[iL];
                    float gR = gest[iR];
                    float gU = gest[iU];
                    float gD = gest[iD];

					//B
                    float bMbase = bbase[iM];
                    float bM = best[iM];
                    float bL = best[iL];
                    float bR = best[iR];
                    float bU = best[iU];
                    float bD = best[iD];

					//Diffs

					float dzL2 = (zM-zL)*(zM-zL)/(zM*zM*zM*zM+zL*zL*zL*zL);
					float dzL2pred = (zM-zLpred)*(zM-zLpred)/(zM*zM*zM*zM+zL*zL*zL*zL);

					float drL = rM-rL;
					float dgL = gM-gL;
					float dbL = bM-bL;

					float dzR2 = (zM-zR)*(zM-zR)/(zM*zM*zM*zM+zR*zR*zR*zR);
					float dzR2pred = (zM-zRpred)*(zM-zRpred)/(zM*zM*zM*zM+zR*zR*zR*zR);
					float drR = rM-rR;
					float dgR = gM-gR;
					float dbR = bM-bR;

					float dzU2 = (zM-zU)*(zM-zU)/(zM*zM*zM*zM+zU*zU*zU*zU);
					float dzU2pred = (zM-zUpred)*(zM-zUpred)/(zM*zM*zM*zM+zU*zU*zU*zU);
					float drU = rM-rU;
					float dgU = gM-gU;
					float dbU = bM-bU;

					float dzD2 = (zM-zD)*(zM-zD)/(zM*zM*zM*zM+zD*zD*zD*zD);
					float dzD2pred = (zM-zDpred)*(zM-zDpred)/(zM*zM*zM*zM+zD*zD*zD*zD);
					float drD = rM-rD;
					float dgD = gM-gD;
					float dbD = bM-bD;

					int itthresh = 2000;
                    if(it < itthresh || dzL2 < dzL2pred){
                        dzL2pred = dzL2;
                        zLpred = zL;
                    }

                    if(it < itthresh ||  dzR2 < dzR2pred){
                        dzR2pred = dzR2;
                        zRpred = zR;
                    }

                    if(it < itthresh ||  dzU2 < dzU2pred){
                        dzU2pred = dzU2;
                        zUpred = zU;
                    }

                    if(it < itthresh ||  dzD2 < dzD2pred){
                        dzD2pred = dzD2;
                        zDpred = zD;
                    }

					float zangleL = (zLdx-zMdx)*(zLdx-zMdx)+(zLdy-zMdy)*(zLdy-zMdy);
					float zangleR = (zRdx-zMdx)*(zRdx-zMdx)+(zRdy-zMdy)*(zRdy-zMdy);
					float zangleU = (zUdx-zMdx)*(zUdx-zMdx)+(zUdy-zMdy)*(zUdy-zMdy);
					float zangleD = (zDdx-zMdx)*(zDdx-zMdx)+(zDdy-zMdy)*(zDdy-zMdy);


                    //Weights
					float distVar = 0.01*0.01;
					float colVar = 0.1*0.1;

                    float diffZ = fabs(zM-zMbase)/(0.001*zMbase*zMbase);

					float weightM = pow(diffZ,2)*0.01*float(zMbase > 0);//exp(-0.5*(zM-zMbase)*(zM-zMbase)/distVar);//1.0*float(zMbase > 0);
                    float weightL = exp(-0.5*(dzL2/distVar + drL*drL/colVar + dgL*dgL/colVar + dbL*dbL/colVar));//*exp(-0.5*zangleL/angleVar);
                    float weightR = exp(-0.5*(dzR2/distVar + drR*drR/colVar + dgR*dgR/colVar + dbR*dbR/colVar));//*exp(-0.5*zangleR/angleVar);
                    float weightU = exp(-0.5*(dzU2/distVar + drU*drU/colVar + dgU*dgU/colVar + dbU*dbU/colVar));//*exp(-0.5*zangleU/angleVar);
                    float weightD = exp(-0.5*(dzD2/distVar + drD*drD/colVar + dgD*dgD/colVar + dbD*dbD/colVar));//*exp(-0.5*zangleD/angleVar);

					if((weightL > 0.4 && weightR < 0.2) || (weightL < 0.2 && weightR > 0.4) || (weightU > 0.4 && weightD < 0.2) || (weightU < 0.2 && weightD > 0.4)){
						pcl::PointXYZRGBA & p = cloud->points[iM];
						p.b = 0;
						p.g = 255;
						p.r = 0;
					}


                    if(w == width/2 && h == height/2){
                        printf("w: %i h:%i\n",w,h);
                        printf("M: %3.3f Base: %3.3f\n",zM,zMbase);
                        printf("L:: val: %3.3f dx: %3.3f dy: %3.3f pred: %3.3f\n",zL,zLdx,zLdy,zLpred);
                        printf("R:: val: %3.3f dx: %3.3f dy: %3.3f pred: %3.3f\n",zR,zRdx,zRdy,zRpred);
                        printf("U:: val: %3.3f dx: %3.3f dy: %3.3f pred: %3.3f\n",zU,zUdx,zUdy,zUpred);
                        printf("D:: val: %3.3f dx: %3.3f dy: %3.3f pred: %3.3f\n",zD,zDdx,zDdy,zDpred);
                    }

					float sumW = weightM+weightL+weightR+weightU+weightD;
					if(sumW > 0){
						sumeL += dzL2*weightL;
						sumeR += dzR2*weightR;
						sumeU += dzU2*weightU;
						sumeD += dzD2*weightD;

						sumeLpred += dzL2pred*weightL;
						sumeRpred += dzR2pred*weightR;
						sumeUpred += dzU2pred*weightU;
						sumeDpred += dzD2pred*weightD;

						if(isnan(sumeLpred)){
							printf("w: %i h:%i\n",w,h);
							printf("L:: val: %3.3f dx: %3.3f dy: %3.3f pred: %3.3f\n",zL,zLdx,zLdy,zLpred);
							printf("normal sums: %10.10f %10.10f %10.10f %10.10f\npredic sums: %10.10f %10.10f %10.10f %10.10f\n",sumeL,sumeR,sumeU,sumeD,sumeLpred,sumeRpred,sumeUpred,sumeDpred);
							exit(0);
						}

						float znextM = (zM*weightM+zLpred*weightL+zRpred*weightR+zUpred*weightU+zDpred*weightD)/sumW;
						znext[iM]	= znextM;
						if(weightL + weightR > 0){
							zdx[iM]		= (weightL*(zL-znextM) + weightR*(znextM-zR))/(weightL + weightR);
						}

						if(weightU + weightD > 0){
							zdy[iM]		= (weightU*(zU-znextM) + weightD*(znextM-zD))/(weightU + weightD);
						}
						/*
						if(fabs(znext[iM] - zMbase)/sqrt(zMbase*zMbase*zMbase*zMbase + znext[iM]*znext[iM]*znext[iM]*znext[iM]) > 0.01 && znext[iM] < 2.0){

							printf("new value: %3.3f\n",znext[iM]);
							printf("value   M: %3.3f L %3.3f R %3.3f U %3.3f D %3.3f\n",zM,zL,zR,zU,zD);
							printf("weight  M: %3.3f L %3.3f R %3.3f U %3.3f D %3.3f\n",weightM,weightL,weightR,weightU,weightD);


							pcl::PointXYZRGBA & p = cloud->points[iM];
							p.b = 0;
							p.g = 255;
							p.r = 0;

							viewer->removeAllShapes();
							viewer->addLine<pcl::PointXYZRGBA> (cloud->points[iM],cloud->points[iL],(1-weightL),weightL,0,"L");
							viewer->addLine<pcl::PointXYZRGBA> (cloud->points[iM],cloud->points[iR],(1-weightR),weightR,0,"R");
							viewer->addLine<pcl::PointXYZRGBA> (cloud->points[iM],cloud->points[iU],(1-weightU),weightU,0,"U");
							viewer->addLine<pcl::PointXYZRGBA> (cloud->points[iM],cloud->points[iD],(1-weightD),weightD,0,"D");

							viewer->removeAllPointClouds();
							viewer->addPointCloud<pcl::PointXYZRGBA> (cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA>(cloud), "cloud");
							viewer->spin();

							p.b = 255;//best[iM]*256.0;
							p.g = 0;//gest[iM]*256.0;
							p.r = 255;//rest[iM]*256.0;

						}
						*/
					}else{
						znext[iM] = zest[iM];
					}

                }
            }

			printf("------\n");
			printf("normal sums: %10.10f %10.10f %10.10f %10.10f\npredic sums: %10.10f %10.10f %10.10f %10.10f\n",sumeL,sumeR,sumeU,sumeD,sumeLpred,sumeRpred,sumeUpred,sumeDpred);

//			std::memcpy(zest,znext,width*height*sizeof(float));
//			std::memcpy(rest,rnext,width*height*sizeof(float));
//			std::memcpy(gest,gnext,width*height*sizeof(float));
//			std::memcpy(best,bnext,width*height*sizeof(float));

			float * ztmp = zest; zest = znext; znext = ztmp;
//			float * rtmp = rest; rest = rnext; rnext = rtmp;
//			float * gtmp = gest; gest = gnext; gnext = gtmp;
//			float * btmp = best; best = bnext; bnext = btmp;
        }
        exit(0);
    }

    if(false){


		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr	cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
		cloud->width	= width;
		cloud->height	= height;
		cloud->points.resize(width*height);

		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("3D Viewer"));
		viewer->setBackgroundColor (0, 0, 0);
		viewer->addCoordinateSystem (1.0);
		viewer->initCameraParameters ();
        //        cv::Mat bilat;
        //        cv::adaptiveBilateralFilter(rgb, bilat, cv::Size(21,21), 30);

        //        cv::namedWindow( "rgb", cv::WINDOW_AUTOSIZE );			cv::imshow( "rgb",      rgb);
        //        cv::namedWindow( "bilat", cv::WINDOW_AUTOSIZE );		cv::imshow( "bilat",	bilat);
        //        cv::waitKey(0);
        //Per channel
        //Base
        //est
        //dx
        //dy
        //errX
        //errY
        std::vector<cv::Mat> channels_base;
        std::vector<cv::Mat> channels_noise;
        std::vector<cv::Mat> channels_est;
        std::vector<cv::Mat> channels_dx;
        std::vector<cv::Mat> channels_dy;
        std::vector<cv::Mat> channels_errx;
        std::vector<cv::Mat> channels_erry;

        cv::Mat z_base;
        z_base.create(height,width,CV_32FC1);
        float * zbase = (float *)z_base.data;

        for(int i = 0; i < width*height; i++){
            zbase[i] = idepth*double(depthdata[i]);
        }

        channels_base.push_back(z_base);
        for(int c = 0; c < 3; c++){
            cv::Mat current_base;
            current_base.create(height,width,CV_32FC1);
            float * currentbase = (float *)current_base.data;

            for(int i = 0; i < width*height; i++){
                currentbase[i] = float(rgbdata[3*i+c])/256.0;
            }
            channels_base.push_back(current_base);
        }

        for(int c = 0; c < channels_base.size(); c++){
            float * data = (float *)(channels_base[c].data);
            cv::Mat tmp;
            tmp.create(height,width,CV_32FC1);
            float * tmpdata = (float *)tmp.data;
            for(int i = 0; i < width*height; i++){
                tmpdata[i] = data[i];
            }
            channels_est.push_back(tmp);
        }

        for(int c = 0; c < channels_base.size(); c++){
            cv::Mat tmp;
            tmp.create(height,width,CV_32FC1);
            float * tmpdata = (float *)tmp.data;
            for(int i = 0; i < width*height; i++){
                tmpdata[i] = 0;
            }
            channels_dx.push_back(tmp);
        }

        for(int c = 0; c < channels_base.size(); c++){
            cv::Mat tmp;
            tmp.create(height,width,CV_32FC1);
            float * tmpdata = (float *)tmp.data;
            for(int i = 0; i < width*height; i++){
                tmpdata[i] = 0;
            }
            channels_dy.push_back(tmp);
        }

        for(int c = 0; c < channels_base.size(); c++){
            cv::Mat tmp;
            tmp.create(height,width,CV_32FC1);
            float * tmpdata = (float *)tmp.data;
            for(int i = 0; i < width*height; i++){
                tmpdata[i] = 0;
            }
            channels_errx.push_back(tmp);
        }

        for(int c = 0; c < channels_base.size(); c++){
            cv::Mat tmp;
            tmp.create(height,width,CV_32FC1);
            float * tmpdata = (float *)tmp.data;
            for(int i = 0; i < width*height; i++){
                tmpdata[i] = 0;
            }
            channels_erry.push_back(tmp);
        }

        //Shared weights
        cv::Mat wximg;
        wximg.create(height,width,CV_32FC1);
        float * wx = (float *)wximg.data;
        for(int i = 0; i < width*height; i++){wx[i] = 1;}

        cv::Mat wyimg;
        wyimg.create(height,width,CV_32FC1);
        float * wy = (float *)wyimg.data;
        for(int i = 0; i < width*height; i++){wy[i] = 1;}




        for(int it = 0; true ; it++){

            if(it % 1 == 0){
                for(int c = 0; c < channels_base.size(); c++){
                    float * est		= (float *)(channels_est[c].data);
                    for(int w = 0; w < width; w++){
                        for(int h = 0; h < height;h++){
                            int ind = h*width+w;
                            pcl::PointXYZRGBA & p = cloud->points[ind];
                            double z = est[ind];
                            if(c == 0){
                                if(z > 0){
                                    p.x = (double(w) - cx) * z * ifx;
                                    p.y = (double(h) - cy) * z * ify;
                                    p.z = z;
                                }else{
                                    p.x = 0;
                                    p.y = 0;
                                    p.z = 0;
                                }
                            }
                            if(c == 1){p.b = z*256.0;}
                            if(c == 2){p.g = z*256.0;}
                            if(c == 3){p.r = z*256.0;}
                        }
                    }
                }

                viewer->removeAllPointClouds();
                viewer->addPointCloud<pcl::PointXYZRGBA> (cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA>(cloud), "cloud");
                viewer->spin();

                viewer->removeAllShapes();
                for(int w0 = 0; w0 < width-1; w0++){
                    for(int h0 = 0; h0 < height;h0+=4){
                        int ind0 = h0*width+w0;

                        int w1 = w0+1;
                        int h1 = h0;
                        int ind1 = h1*width+w1;

                        pcl::PointXYZRGBA & p0 = cloud->points[ind0];
                        pcl::PointXYZRGBA & p1 = cloud->points[ind1];
                        if(p0.z == 0 || p1.z == 0){continue;}

                        float weight = weightFunc(w0,h0,w1,h1,width, channels_est, channels_dx, channels_dy);
                        if(weight > 0.5){continue;}
                        char buf [1024];
                        sprintf(buf,"line_%i_%i_%i_%i",w0,h0,w1,h1);
                        viewer->addLine<pcl::PointXYZRGBA> (p0,p1,(1-weight),weight,0,buf);
                    }
                }

                for(int w0 = 0; w0 < width; w0+=4){
                    for(int h0 = 0; h0 < height-1;h0++){
                        int ind0 = h0*width+w0;

                        int w1 = w0;
                        int h1 = h0+1;
                        int ind1 = h1*width+w1;

                        pcl::PointXYZRGBA & p0 = cloud->points[ind0];
                        pcl::PointXYZRGBA & p1 = cloud->points[ind1];
                        if(p0.z == 0 || p1.z == 0){continue;}

                        float weight = weightFunc(w0,h0,w1,h1,width, channels_est, channels_dx, channels_dy);

                        if(weight > 0.5){continue;}
                        char buf [1024];
                        sprintf(buf,"line_%i_%i_%i_%i",w0,h0,w1,h1);
                        viewer->addLine<pcl::PointXYZRGBA> (p0,p1,(1-weight),weight,0,buf);
                    }
                }
                viewer->spin();

                //float pred(float targetW, float targetH, int sourceW, int sourceH, int width, float * est, float * dx, float * dy)
            }

            //            float mulbase = 0.1;
            //            for(int w = 1; w < width-1; w++){
            //                for(int h = 1; h < height-1;h++){
            //                    int i			= h*width+w;
            //                    float weightBase = mulbase*((float*)(channels_base[0].data))[i] > 0;
            //                    float weightL = weightFunc(w,h,w-1,h  ,width, channels_est, channels_dx, channels_dy);
            //                    float weightR = weightFunc(w,h,w+1,h  ,width, channels_est, channels_dx, channels_dy);
            //                    float weightU = weightFunc(w,h,w  ,h-1,width, channels_est, channels_dx, channels_dy);
            //                    float weightD = weightFunc(w,h,w  ,h+1,width, channels_est, channels_dx, channels_dy);
            //                    float sumW = weightBase+weightL+weightR+weightU+weightD;
            //                    if(sumW == 0){continue;}
            //                    for(int c = 0; c < channels_base.size(); c++){
            //                        float base  = ((float*)(channels_base[c].data))[i];
            //                        float L     = pred(w,h,w-1,h,width, (float*)(channels_est[c].data), (float*)(channels_dx[c].data), (float*)(channels_dy[c].data));
            //                        float R     = pred(w,h,w+1,h,width, (float*)(channels_est[c].data), (float*)(channels_dx[c].data), (float*)(channels_dy[c].data));
            //                        float U     = pred(w,h,w,h-1,width, (float*)(channels_est[c].data), (float*)(channels_dx[c].data), (float*)(channels_dy[c].data));
            //                        float D     = pred(w,h,w,h+1,width, (float*)(channels_est[c].data), (float*)(channels_dx[c].data), (float*)(channels_dy[c].data));
            //                        float estimate = (weightBase*base+weightL*L+weightR*R+weightU*U+weightL*L+weightD*D)/sumW;
            //                        ((float*)(channels_est[c].data))[i] = estimate;
            //                    }
            //                }
            //            }
        }
        //}

        /*
            //Update errors
            for(int c = 0; c < channels_base.size(); c++){
                float * estdata = (float *)(channels_est[c].data);

                float * dxdata = (float *)(channels_dx[c].data);
                float * dydata = (float *)(channels_dy[c].data);

                float * errxdata = (float *)(channels_errx[c].data);
                float * errydata = (float *)(channels_erry[c].data);
                for(int w = 0; w < width-1; w++){
                    for(int h = 0; h < height;h++){
                        int i			= h*width+w;
                        int i2			= i+1;//			cv::namedWindow( "wx", cv::WINDOW_AUTOSIZE );			cv::imshow( "wx",	wximg);
//			cv::namedWindow( "wy", cv::WINDOW_AUTOSIZE );			cv::imshow( "wy",	wyimg);
//			cv::waitKey(0);
                        float z0		= estdata[i];
                        float z1		= estdata[i2];
                        float slope0	= dxdata[i];
                        float slope1	= dxdata[i2];

                        float e0 = (z0+slope0)-z1;
                        float e1 = (z1-slope1)-z0;

                        errxdata[i] = fabs(e0)+fabs(e1);
                    }
                }

                for(int w = 0; w < width; w++){
                    for(int h = 0; h < height-1;h++){
                        int i			= h*width+w;
                        int i2			= i+width;
                        float z0		= estdata[i];
                        float z1		= estdata[i2];
                        float slope0	= dydata[i];
                        float slope1	= dydata[i2];

                        float e0 = (z0+slope0)-z1;
                        float e1 = (z1-slope1)-z0;

                        errydata[i] = fabs(e0)+fabs(e1);
                    }
                }
            }

            //Update weights
            for(int w = 0; w < width; w++){
                for(int h = 0; h < height;h++){
                    int i = h*width+w;
                    float weight = 1;
                    for(int c = 0; c < channels_base.size(); c++){
                        float e = ((float *)(channels_errx[c].data))[i];
                        if(c == 0){weight *= exp(-0.5*e*e/0.1);}
                        if(c == 1){weight *= exp(-0.5*e*e/0.005);}
                        if(c == 2){weight *= exp(-0.5*e*e/0.005);}
                        if(c == 3){weight *= exp(-0.5*e*e/0.005);}
                    }
                    wx[i] = weight;
                }
            }

            //Update weights
            for(int w = 0; w < width; w++){
                for(int h = 0; h < height;h++){
                    int i = h*width+w;
                    float weight = 1;
                    for(int c = 0; c < channels_base.size(); c++){
                        float e = ((float *)(channels_erry[c].data))[i];
                        if(c == 0){weight *= exp(-0.5*e*e/0.1);}
                        if(c == 1){weight *= exp(-0.5*e*e/0.005);}
                        if(c == 2){weight *= exp(-0.5*e*e/0.005);}
                        if(c == 3){weight *= exp(-0.5*e*e/0.005);}
                    }
                    wy[i] = weight;
                }
            }

//			cv::namedWindow( "wx", cv::WINDOW_AUTOSIZE );			cv::imshow( "wx",	wximg);
//			cv::namedWindow( "wy", cv::WINDOW_AUTOSIZE );			cv::imshow( "wy",	wyimg);
//			cv::waitKey(0);

            //Update points
            double wbase = 0.1;
            for(int c = 0; c < channels_base.size(); c++){
                float * base	= (float *)(channels_base[c].data);
                float * est		= (float *)(channels_est[c].data);
                float * dx		= (float *)(channels_dx[c].data);
                float * dy		= (float *)(channels_dy[c].data);
                for(int w = 1; w < width-1; w++){
                    for(int h = 1; h < height-1;h++){
                        int i = h*width+w;

                        float wtmp =  wbase * float(base[i] > 0);
                        if(wtmp == 0){continue;}
                        float w0 = wx[i-1] * float(base[i-1] > 0);
                        float w1 = wx[i] * float(base[i+1] > 0);
                        float w2 = wy[i-1] * float(base[i-width] > 0);
                        float w3 = wy[i] * float(base[i+width] > 0);

                        float z			= base[i];
                        float pred0		= est[i-1]		+dx[i-1];
                        float pred1		= est[i+1]		-dx[i+1];
                        float pred2		= est[i-width]	+dy[i-width];
                        float pred3		= est[i+width]	-dy[i+width];

                        double sumz = wtmp*z+w0*pred0+w1*pred1+w2*pred2+w3*pred3;
                        double sumw = wtmp+w0+w1+w2+w3;

                        if(false && w % 40 == 0 && h % 40 == 0){
                            printf("%4.4i %4.4i %4.4i -> M(%4.4f %4.4f) ",c,w,h,z,wbase);
                            printf("est: %4.4f %4.4f %4.4f ",	est[i-1],est[i],est[i+1]);
                            printf("dx: %4.4f %4.4f %4.4f ",	dx[i-1], dx[i], dx[i+1]);
                            printf("wx: %4.4f %4.4f",			wx[i-1], wx[i]);
                            printf("\n");
                        }

                        //if(w % 40 == 0 && h % 40 == 0){printf("%i %i %i -> M(%4.4f %4.4f) L(%4.4f %4.4f) R(%4.4f %4.4f) U(%4.4f %4.4f) D(%4.4f %4.4f)\n",c,w,h,z,wbase,pred0,w0,pred1,w1,pred2,w2,pred3,w3);}
                        if(sumw > 0){
                            est[i] = sumz/sumw;
                        }else{
                            //est[i] = 0;
                        }

                    }
                }
                */
        //				exit(0);
        //			}

        //			//Update slopes


        //			//}

        ////			cv::namedWindow( "wx", cv::WINDOW_AUTOSIZE );			cv::imshow( "wx",	wximg);
        ////			cv::namedWindow( "wy", cv::WINDOW_AUTOSIZE );			cv::imshow( "wy",	wyimg);
        ////			cv::waitKey(0);
        //		}

        //			cv::namedWindow( "est", cv::WINDOW_AUTOSIZE );			cv::imshow( "est",	channels_est[c] );
        //			cv::namedWindow( "dx", cv::WINDOW_AUTOSIZE );			cv::imshow( "dx",	channels_dx[c]  );
        //			cv::namedWindow( "dy", cv::WINDOW_AUTOSIZE );           cv::imshow( "dy",	channels_dy[c]  );
        //			cv::namedWindow( "errx", cv::WINDOW_AUTOSIZE );			cv::imshow( "errx",	channels_errx[c]);
        //			cv::namedWindow( "erry", cv::WINDOW_AUTOSIZE );			cv::imshow( "erry",	channels_erry[c]);
        //			cv::waitKey(0);
        exit(0);
        //Shared weights

        //Diffs
        cv::Mat diffzimg;
        diffzimg.create(height,width,CV_32FC2);
        float * diffz = (float *)diffzimg.data;

        cv::Mat diffrimg;
        diffrimg.create(height,width,CV_32FC2);
        float * diffr = (float *)diffrimg.data;

        cv::Mat diffgimg;
        diffgimg.create(height,width,CV_32FC2);
        float * diffg = (float *)diffgimg.data;

        cv::Mat diffbimg;
        diffbimg.create(height,width,CV_32FC2);
        float * diffb = (float *)diffbimg.data;



        cv::Mat colimg_est;
        colimg_est.create(height,width,CV_32FC3);
        float * cest = (float *)colimg_est.data;

        cv::Mat colimg_base;
        colimg_base.create(height,width,CV_32FC3);
        float * cbase = (float *)colimg_base.data;

        cv::Mat cn_est;
        cn_est.create(height,width,CV_32FC3);
        float * cnest = (float *)cn_est.data;

        cv::Mat z_est;
        z_est.create(height,width,CV_32FC3);
        float * zest = (float *)z_est.data;



        cv::Mat nimg;
        nimg.create(height,width,CV_32FC3);
        float * nest = (float *)nimg.data;


        cv::Mat dz_est;
        dz_est.create(height,width,CV_32FC2);
        float * dzest = (float *)dz_est.data;



        for(int i = 0; i < width*height; i++){
            zbase[i]       = zest[i] = idepth*double(depthdata[i]);
        }

        for(int i = 0; i < 3*width*height; i++){
            cbase[i]       = cest[i] = float(rgbdata[i])/256.0;
        }

        for(int w = 0; w < width; w++){
            for(int h = 0; h < height;h++){
                int i = h*width+w;
                dzest[2*i+0] = 0;
                dzest[2*i+1] = 0;
            }
        }



        for(int it = 0; true; it++){
            printf("it: %i\n",it);
            if(it % 100 == 0){
                for(int w = 0; w < width; w++){
                    for(int h = 0; h < height;h++){
                        int ind = h*width+w;
                        pcl::PointXYZRGBA & p = cloud->points[ind];
                        p.b = cest[3*ind+0]*256.0;
                        p.g = cest[3*ind+1]*256.0;
                        p.r = cest[3*ind+2]*256.0;
                        double z = zest[ind];
                        if(z > 0){
                            p.x = (double(w) - cx) * z * ifx;
                            p.y = (double(h) - cy) * z * ify;
                            p.z = z;
                        }else{
                            p.x = 0;
                            p.y = 0;
                            p.z = 0;
                        }
                    }
                }

                viewer->removeAllPointClouds();
                viewer->addPointCloud<pcl::PointXYZRGBA> (cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA>(cloud), "cloud");
                viewer->spin();
            }

            ////            Compute normals
            //            cv::namedWindow( "rgb", cv::WINDOW_AUTOSIZE );			cv::imshow( "rgb", colimg_est );
            //            cv::namedWindow( "normals", cv::WINDOW_AUTOSIZE );		cv::imshow( "normals", nimg  );
            //            cv::namedWindow( "dz", cv::WINDOW_AUTOSIZE );           cv::imshow( "dz", dz_est  );
            //            cv::namedWindow( "depth", cv::WINDOW_AUTOSIZE );		cv::imshow( "depth", z_est);
            //            cv::waitKey(0);

            //            for(int w = 0; w < width; w++){
            //                for(int h = 0; h < height;h++){
            //                    int i = h*width+w;
            //                    double z = zest[i];
            //                    if(w < width-1){    dzest[2*i+0] = z-zest[i+1];}
            //                    else{               dzest[2*i+0] = dzest[2*i-2];}

            //                    if(h < height-1){   dzest[2*i+1] = z-zest[i+width];}
            //                    else{               dzest[2*i+1] = dzest[2*(i-width)+1];}
            //                }
            //            }

            double w0 = 0.001;
            for(int w = 0; w < width; w++){
                for(int h = 0; h < height;h++){
                    int i = h*width+w;

                    if(w < 103 && w > 97 && h == 100){
                        printf("mid %i: base: %5.5f %5.5f\n",w,zbase[i],zest[i]);
                    }
                    double z = zbase[i];
                    double sumw = w0/(z*z);
                    double sumz = sumw*zbase[i];

                    if(z == 0){
                        sumz = 0;
                        sumw = 0;
                    }

                    if(w > 0){
                        int i2 = i-1;

                        double predz = zest[i2]+dzest[2*i+0];
                        double weight = fabs(z -predz) < 0.05;
                        sumz+=weight*predz;
                        sumw+=weight;

                        //                        if(w == 100 && h == 100){
                        //                            printf("left %i: base: %5.5f %5.5f\n",w-1,zbase[i2],zest[i2]);
                        //                        }
                    }

                    if(w < width-1){
                        int i2 = i+1;
                        double predz = zest[i2]-dzest[2*i+0];
                        double weight = fabs(z -predz) < 0.05;
                        sumz+=weight*predz;
                        sumw+=weight;
                    }

                    if(h > 0){
                        int i2 = i-width;
                        double predz = zest[i2]+dzest[2*i+0];
                        double weight = fabs(z -predz) < 0.05;
                        sumz+=weight*predz;
                        sumw+=weight;
                    }

                    if(h < height-1){
                        int i2 = i+width;
                        double predz = zest[i2]-dzest[2*i+0];
                        double weight = fabs(z -predz) < 0.05;
                        sumz+=weight*predz;
                        sumw+=weight;
                    }

                    if(sumw == 0){
                        zest[i] = 0;
                    }else{
                        zest[i] = sumz/sumw;
                    }
                }
            }



        }
        //        double * z_base     = new double[width*height];
        //        double * col_base   = new double[3*width*height];
        //        double * z_est      = new double[width*height];
        //        double * col_est    = new double[3*width*height];
        //        double * normal_est = new double[3*width*height];

        //        for(int i = 0; i < width*height; i++){
        //            z_base[i]       = z_est[i] = idepth*double(depthdata[ind]);
        //            col_est[3*i+0]  = rgbdata[3*i+0];
        //            col_est[3*i+1]  = rgbdata[3*i+1];
        //            col_est[3*i+2]  = rgbdata[3*i+2];
        //        }
        //        delete[] z_base;
        //        delete[] col_base;
        //        delete[] z_est;
        //        delete[] col_est;
        //        delete[] normal_est;
    }

    //printf("%s LINE:%i\n",__FILE__,__LINE__);
    if(compute_normals){
        normals.create(height,width,CV_32FC3);
        float * normalsdata = (float *)normals.data;

        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr	cloud	(new pcl::PointCloud<pcl::PointXYZRGBA>);
        pcl::PointCloud<pcl::Normal>::Ptr	normals_cloud (new pcl::PointCloud<pcl::Normal>);
        cloud->width	= width;
        cloud->height	= height;
        cloud->points.resize(width*height);
        //cloud->dense = false;

        for(int w = 0; w < width; w++){
            for(int h = 0; h < height;h++){
                int ind = h*width+w;
                pcl::PointXYZRGBA & p = cloud->points[ind];
                p.b = rgbdata[3*ind+0];
                p.g = rgbdata[3*ind+1];
                p.r = rgbdata[3*ind+2];
                double z = idepth*double(depthdata[ind]);
                if(z > 0){
                    p.x = (double(w) - cx) * z * ifx;
                    p.y = (double(h) - cy) * z * ify;
                    p.z = z;
                }else{
                    p.x = NAN;
                    p.y = NAN;
                    p.z = NAN;
                }
            }
        }

        //printf("%s LINE:%i\n",__FILE__,__LINE__);
        pcl::IntegralImageNormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne;
        ne.setInputCloud(cloud);

        bool tune = false;
        unsigned char * combidata;
        cv::Mat combined;

        int NormalEstimationMethod = 0;
        int MaxDepthChangeFactor = 20;
        int NormalSmoothingSize = 7;
        int depth_dependent_smoothing = 1;

        ne.setMaxDepthChangeFactor(0.001*double(MaxDepthChangeFactor));
        ne.setNormalSmoothingSize(NormalSmoothingSize);
        ne.setDepthDependentSmoothing (depth_dependent_smoothing);
        ne.compute(*normals_cloud);

        for(int w = 0; w < width; w++){
            for(int h = 0; h < height;h++){
                int ind = h*width+w;
                pcl::PointXYZRGBA p = cloud->points[ind];
                pcl::Normal		p2		= normals_cloud->points[ind];


                //if(w % 20 == 0 && h % 20 == 0){printf("%i %i -> point(%f %f %f) normal(%f %f %f)\n",w,h,p.x,p.y,p.z,p2.normal_x,p2.normal_y,p2.normal_z);}

                if(!isnan(p2.normal_x)){
                    normalsdata[3*ind+0]	= p2.normal_x;
                    normalsdata[3*ind+1]	= p2.normal_y;
                    normalsdata[3*ind+2]	= p2.normal_z;
                }else{
                    normalsdata[3*ind+0]	= 2;
                    normalsdata[3*ind+1]	= 2;
                    normalsdata[3*ind+2]	= 2;
                }
            }
        }

        //		cv::Mat curvature;
        //		curvature.create(height,width,CV_8UC3);
        //		unsigned char * curvaturedata = (unsigned char *)curvature.data;
        //		for(int w = 0; w < width; w++){
        //			for(int h = 0; h < height;h++){
        //				int ind = h*width+w;
        //				pcl::Normal		p2		= normals_cloud->points[ind];
        //				//if(w % 5 == 0 && h % 5 == 0){printf("%i %i -> curvature %f\n",w,h,p2.curvature);}
        //				curvaturedata[3*ind+0]	= 255.0*p2.curvature;
        //				curvaturedata[3*ind+1]	= 255.0*p2.curvature;
        //				curvaturedata[3*ind+2]	= 255.0*p2.curvature;
        //			}
        //		}
        //		cv::namedWindow( "curvature", cv::WINDOW_AUTOSIZE );
        //		cv::imshow( "curvature", curvature );
        //		cv::waitKey(0);



        //printf("%s LINE:%i\n",__FILE__,__LINE__);
        if(tune){
            combined.create(height,2*width,CV_8UC3);
            combidata = (unsigned char *)combined.data;

            cv::namedWindow( "normals", cv::WINDOW_AUTOSIZE );
            cv::namedWindow( "combined", cv::WINDOW_AUTOSIZE );

            //cv::createTrackbar( "NormalEstimationMethod", "combined", &NormalEstimationMethod, 3, on_trackbar );
            //cv::createTrackbar( "MaxDepthChangeFactor", "combined", &MaxDepthChangeFactor, 1000, on_trackbar );
            //cv::createTrackbar( "NormalSmoothingSize", "combined", &NormalSmoothingSize, 100, on_trackbar );
            //cv::createTrackbar( "depth_dependent_smoothing", "combined", &depth_dependent_smoothing, 1, on_trackbar );

            //while(true){

            if(NormalEstimationMethod == 0){ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);}
            if(NormalEstimationMethod == 1){ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);}
            if(NormalEstimationMethod == 2){ne.setNormalEstimationMethod (ne.AVERAGE_DEPTH_CHANGE);}
            if(NormalEstimationMethod == 3){ne.setNormalEstimationMethod (ne.SIMPLE_3D_GRADIENT);}

            ne.setMaxDepthChangeFactor(0.001*double(MaxDepthChangeFactor));
            ne.setNormalSmoothingSize(NormalSmoothingSize);
            ne.setDepthDependentSmoothing (depth_dependent_smoothing);
            ne.compute(*normals_cloud);
            for(int w = 0; w < width; w++){
                for(int h = 0; h < height;h++){
                    int ind = h*width+w;
                    pcl::Normal		p2		= normals_cloud->points[ind];
                    if(!isnan(p2.normal_x)){
                        normalsdata[3*ind+0]	= p2.normal_x;
                        normalsdata[3*ind+1]	= p2.normal_y;
                        normalsdata[3*ind+2]	= p2.normal_z;
                    }else{
                        normalsdata[3*ind+0]	= 2;
                        normalsdata[3*ind+1]	= 2;
                        normalsdata[3*ind+2]	= 2;
                    }
                }
            }


            for(int w = 0; w < width; w++){
                for(int h = 0; h < height;h++){
                    int ind = h*width+w;
                    int indn = h*2*width+(w+width);
                    int indc = h*2*width+(w);
                    combidata[3*indc+0]	= rgbdata[3*ind+0];
                    combidata[3*indc+1]	= rgbdata[3*ind+1];
                    combidata[3*indc+2]	= rgbdata[3*ind+2];
                    pcl::Normal		p2		= normals_cloud->points[ind];
                    if(!isnan(p2.normal_x)){
                        combidata[3*indn+0]	= 255.0*fabs(p2.normal_x);
                        combidata[3*indn+1]	= 255.0*fabs(p2.normal_y);
                        combidata[3*indn+2]	= 255.0*fabs(p2.normal_z);
                    }else{
                        combidata[3*indn+0]	= 255;
                        combidata[3*indn+1]	= 255;
                        combidata[3*indn+2]	= 255;
                    }
                }
            }
            char buf [1024];
            sprintf(buf,"combined%i.png",int(id));
            cv::imwrite( buf, combined );
            printf("saving: %s\n",buf);
            cv::imshow( "combined", combined );
            cv::waitKey(0);
            //while(!updated){cv::waitKey(50);}
            updated = false;
            //}
        }

        //pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2 = getPCLcloud();
        pcl::OrganizedEdgeFromRGBNormals<pcl::PointXYZRGBA, pcl::Normal, pcl::Label> oed;
        oed.setInputNormals (normals_cloud);
        oed.setInputCloud (cloud);
        oed.setDepthDisconThreshold (0.02); // 2cm
        //		oed.setHCCannyLowThreshold (0.4);
        //		oed.setHCCannyHighThreshold (1.1);
        oed.setMaxSearchNeighbors (50);
        pcl::PointCloud<pcl::Label> labels;
        std::vector<pcl::PointIndices> label_indices;
        oed.compute (labels, label_indices);

        //		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr occluding_edges (new pcl::PointCloud<pcl::PointXYZRGBNormal>),
        //				occluded_edges (new pcl::PointCloud<pcl::PointXYZRGBNormal>),
        //				boundary_edges (new pcl::PointCloud<pcl::PointXYZRGBNormal>),
        //				high_curvature_edges (new pcl::PointCloud<pcl::PointXYZRGBNormal>),
        //				rgb_edges (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

        //		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud3 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        //		pcl::copyPointCloud (*cloud2, label_indices[0].indices, *boundary_edges);
        //		pcl::copyPointCloud (*cloud2, label_indices[1].indices, *occluding_edges);
        //		pcl::copyPointCloud (*cloud2, label_indices[2].indices, *occluded_edges);
        //		pcl::copyPointCloud (*cloud2, label_indices[3].indices, *high_curvature_edges);
        //		pcl::copyPointCloud (*cloud2, label_indices[4].indices, *rgb_edges);

        //		cv::Mat out = rgb.clone();

        for(unsigned int i = 0; i < label_indices[1].indices.size(); i++){
            int ind = label_indices[1].indices[i];
            //depthedgesdata[ind] = 1;
            //			out.data[3*ind+0] =255;
            //			out.data[3*ind+1] =0;
            //			out.data[3*ind+2] =255;
        }

        //		for(unsigned int i = 0; i < label_indices[3].indices.size(); i++){
        //			int ind = label_indices[3].indices[i];
        //			depthedgesdata[ind] = 255;
        //			out.data[3*ind+0] =0;
        //			out.data[3*ind+1] =255;
        //			out.data[3*ind+2] =255;
        //		}


        for(unsigned int i = 0; i < label_indices[4].indices.size(); i++){
            int ind = label_indices[4].indices[i];
            depthedgesdata[ind] = 255;
            //			out.data[3*ind+0] =0;
            //			out.data[3*ind+1] =255;
            //			out.data[3*ind+2] =0;
        }

        //		cv::namedWindow( "edges", cv::WINDOW_AUTOSIZE );
        //		cv::imshow( "edges", out );
        //		cv::waitKey(0);

        //show(true);
    }

    if(true){

        for(int it = 0; it < 30; it++){
            for(int w = 0; w < width; w++){
                for(int h = 0; h < height;h++){
                    int ind = h*width+w;
                }
            }
        }
    }

    if(false){
        show(false);

        int * last = new int[width*height];
        for(int i = 0; i < width*height; i++){last[i] = 0;}
        int current = 1;

        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("3D Viewer"));
        viewer->setBackgroundColor (0, 0, 0);
        viewer->addCoordinateSystem (1.0);
        viewer->initCameraParameters ();

        for(int w = 0; w < width; w+=5){
            for(int h = 0; h < height;h+=5){
                int ind = h*width+w;

                int edgetype = depthedgesdata[ind];
                if(edgetype != 0 && depthdata[ind] != 0){
                    double xx = 0.001;
                    double xy = 0;
                    double xz = 0;
                    double yy = xx;
                    double yz = 0;
                    double zz = xx;
                    double sum = 0.001;

                    double ww = 4.0;
                    double wh = 0;
                    double hh = 4.0;
                    double sum2 = 4.0;


                    double z = idepth*double(depthdata[ind]);
                    double x = (double(w) - cx) * z * ifx;
                    double y = (double(h) - cy) * z * ify;

                    //				for(int ww = 0; ww < width; ww++){
                    //					for(int hh = 0; hh < height;hh++){
                    //						int ind = hh*width+ww;
                    //						pcl::PointXYZRGBA & p = cloud->points[ind];
                    //						p.b = 0;
                    //						p.g = 255;
                    //						p.r = 0;
                    //					}
                    //				}




                    std::vector<int> todolistw;
                    todolistw.push_back(w);
                    std::vector<int> todolisth;
                    todolisth.push_back(h);
                    current++;
                    last[ind] = current;

                    cv::Mat rgbclone = rgb.clone();
                    unsigned char * rgbclonedata = (unsigned char *)rgbclone.data;


                    cv::circle(rgbclone, cv::Point(w,h), 2, cv::Scalar(0,255,0));
                    cv::imshow( "rgbclone", rgbclone );


                    printf("ratio = [");
                    double best_ratio = 0;
                    int tmp;
                    for(int go = 0; go < 100000 && go < todolistw.size(); go++){
                        int cw = todolistw[go];
                        int ch = todolisth[go];
                        int ind = ch*width+cw;

                        int dw = cw-w;
                        int dh = ch-h;
                        ww+=dw*dw;
                        wh+=dw*dh;
                        hh+=dh*dh;
                        sum2++;

                        Eigen::Matrix2d mimg;
                        mimg(0,0) = ww/sum2;
                        mimg(0,1) = wh/sum2;
                        mimg(1,0) = wh/sum2;
                        mimg(1,1) = hh/sum2;

                        Eigen::EigenSolver<Eigen::Matrix2d> es(mimg);
                        double e1 = (es.eigenvalues())(0).real();
                        double e2 = (es.eigenvalues())(1).real();
                        double ratio = 1;
                        if(e1 > e2){ratio = e1/e2;}
                        if(e2 > e1){ratio = e2/e1;}
                        printf("%5.5f ",ratio);

                        if(ratio < best_ratio){
                            ww-=dw*dw;
                            wh-=dw*dh;
                            hh-=dh*dh;
                            sum2--;
                            //cv::circle(rgbclone, cv::Point(cw,ch), 2, cv::Scalar(255,0,255));
                            continue;
                        }

                        best_ratio = ratio;
                        //                    cout << "The mimg are:" << endl << mimg << endl;
                        //					cout << "The eigenvalues of mimg are:" << endl << es.eigenvalues() << endl;
                        //					printf("%f %f\n",e1,e2);
                        //					cout << "The matrix of eigenvectors, V, is:" << endl << es.eigenvectors() << endl << endl;

                        //					B2=(-cov[0][0]-cov[1][1])*(-cov[0][0]-cov[1][1]);
                        //				    c= (4*cov[0][0]*cov[1][1])-(4*cov[0][1]*cov[1][0]);


                        //				    sq= sqrt(B2-c);


                        //				    B=(cov[0][0])+(cov[1][1]);

                        //				    r1=(B+sq)(.5);
                        //				    r2=(B-sq)(.5);


                        //				    printf("\nThe eigenvalues are &#37;f and %f", r1, r2);


                        double z2 = idepth*double(depthdata[ind]);
                        if(z2 > 0){
                            double x2 = (double(cw) - cx) * z2 * ifx;
                            double y2 = (double(ch) - cy) * z2 * ify;

                            double dx = x-x2;
                            double dy = y-y2;
                            double dz = z-z2;

                            xx += dx*dx;
                            xy += dx*dy;
                            xz += dx*dz;

                            yy += dy*dy;
                            yz += dy*dz;

                            zz += dz*dz;
                            sum++;
                        }

                        //double z = idepth*double(depthdata[ind]);
                        rgbclonedata[3*ind+0] = 0;
                        rgbclonedata[3*ind+1] = 0;
                        rgbclonedata[3*ind+2] = 255;

                        //					cloud->points[ind].r = 255;
                        //					cloud->points[ind].g = 0;
                        //					cloud->points[ind].b = 0;

                        int startw	= std::max(int(0),cw-1);
                        int stopw	= std::min(int(width-1),cw+1);

                        int starth	= std::max(int(0),ch-1);
                        int stoph	= std::min(int(height-1),ch+1);

                        for(int tw = startw; tw <= stopw; tw++){
                            for(int th = starth; th <= stoph; th++){
                                int ind2 = th*width+tw;
                                if(depthedgesdata[ind2] == edgetype && last[ind2] != current){
                                    todolistw.push_back(tw);
                                    todolisth.push_back(th);
                                    last[ind2] = current;
                                }
                            }
                        }

                    }


                    printf("];\n");
                    ww /= sum2;
                    wh /= sum2;
                    hh /= sum2;
                    Eigen::Matrix2d mimg;
                    mimg(0,0) = ww;
                    mimg(0,1) = wh;
                    mimg(1,0) = wh;
                    mimg(1,1) = hh;

                    Eigen::EigenSolver<Eigen::Matrix2d> es(mimg);
                    cout << "The mimg are:" << endl << mimg << endl;
                    cout << "The eigenvalues of mimg are:" << endl << es.eigenvalues() << endl;
                    cout << "The matrix of eigenvectors, V, is:" << endl << es.eigenvectors() << endl << endl;

                    //cv::circle(rgbclone, cv::Point(w,h), 3, cv::Scalar(0,255,0));


                    cv::imshow( "rgbclone", rgbclone );
                    cv::waitKey(0);
                    /*

                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr	cloud2	(new pcl::PointCloud<pcl::PointXYZRGBA>);
                cloud2->width	= width;
                cloud2->height	= height;
                cloud2->points.resize(width*height);
                    //cloud->dense = false;

                unsigned short *	depthdata	= (unsigned short *)depth.data;
                unsigned char *		rgbdata		= (unsigned char *)rgb.data;
                for(int w3 = 0; w3 < width; w3++){
                    for(int h3 = 0; h3 < height;h3++){
                        int ind3 = h3*width+w3;
                        pcl::PointXYZRGBA & p = cloud2->points[ind3];
                        p.b = rgbdata[3*ind3+0];//255;
                        p.g = rgbdata[3*ind3+1];//255;
                        p.r = rgbdata[3*ind3+2];//255;
                    }
                }

                for(int go = 0; go < todolistw.size(); go++){
                    int cw = todolistw[go];
                    int ch = todolisth[go];
                    int ind3 = ch*width+cw;

                    pcl::PointXYZRGBA & p = cloud2->points[ind3];
                    p.b = 0;
                    p.g = 0;
                    p.r = 255;
                }

                for(int w3 = 0; w3 < width; w3++){
                    for(int h3 = 0; h3 < height;h3++){
                        int ind3 = h3*width+w3;
                        pcl::PointXYZRGBA & p = cloud2->points[ind3];
                        double z3 = idepth*double(depthdata[ind3]);
                        if(z3 > 0){
                            p.x = (double(w3) - cx) * z3 * ifx;
                            p.y = (double(h3) - cy) * z3 * ify;
                            p.z = z3;
                        }else{
                            p.x = 0;
                            p.y = 0;
                            p.z = 0;
                        }
                    }
                }

                viewer->removeAllPointClouds();
                viewer->addPointCloud<pcl::PointXYZRGBA> (cloud2, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA>(cloud2), "cloud");
                viewer->spin();
                */
                }
            }
        }
    }

}

RGBDFrame::~RGBDFrame(){
    rgb.release();
    normals.release();
    depth.release();
    depthedges.release();
	//if(rgbdata != 0){delete[] rgbdata; rgbdata = 0;}
    if(labels != 0){delete[] labels; labels = 0;}
}

void RGBDFrame::show(bool stop){
    cv::namedWindow( "rgb", cv::WINDOW_AUTOSIZE );			cv::imshow( "rgb", rgb );
    cv::namedWindow( "normals", cv::WINDOW_AUTOSIZE );		cv::imshow( "normals", normals );
    cv::namedWindow( "depth", cv::WINDOW_AUTOSIZE );		cv::imshow( "depth", depth );
    cv::namedWindow( "depthedges", cv::WINDOW_AUTOSIZE );	cv::imshow( "depthedges", depthedges );
    if(stop){	cv::waitKey(0);}else{					cv::waitKey(30);}
}

pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr RGBDFrame::getPCLcloud(){
    unsigned char * rgbdata = (unsigned char *)rgb.data;
    unsigned short * depthdata = (unsigned short *)depth.data;

    const unsigned int width	= camera->width; const unsigned int height	= camera->height;
    const double idepth			= camera->idepth_scale;
    const double cx				= camera->cx;		const double cy				= camera->cy;
    const double ifx			= 1.0/camera->fx;	const double ify			= 1.0/camera->fy;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr	cloud	(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::Normal>::Ptr		normals (new pcl::PointCloud<pcl::Normal>);
    cloud->width	= width;
    cloud->height	= height;
    cloud->points.resize(width*height);

    for(unsigned int w = 0; w < width; w++){
        for(unsigned int h = 0; h < height;h++){
            int ind = h*width+w;
            double z = idepth*double(depthdata[ind]);

            pcl::PointXYZRGB p;
            p.b = rgbdata[3*ind+0];
            p.g = rgbdata[3*ind+1];
            p.r = rgbdata[3*ind+2];
            if(z > 0){
                p.x = (double(w) - cx) * z * ifx;
                p.y = (double(h) - cy) * z * ify;
                p.z = z;
            }else{
                p.x = NAN;
                p.y = NAN;
                p.z = NAN;
            }
            cloud->points[ind] = p;
        }
    }

    pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
    ne.setMaxDepthChangeFactor(0.02f);
    ne.setNormalSmoothingSize(10.0f);
    ne.setInputCloud(cloud);
    ne.compute(*normals);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    cloud_ptr->width	= width;
    cloud_ptr->height	= height;
    cloud_ptr->points.resize(width*height);

    for(unsigned int w = 0; w < width; w++){
        for(unsigned int h = 0; h < height;h++){
            int ind = h*width+w;
            pcl::PointXYZRGBNormal & p0	= cloud_ptr->points[ind];
            pcl::PointXYZRGB p1			= cloud->points[ind];
            pcl::Normal p2				= normals->points[ind];
            p0.x		= p1.x;
            p0.y		= p1.y;
            p0.z		= p1.z;
            p0.r		= p1.r;
            p0.g		= p1.g;
            p0.b		= p1.b;
            p0.normal_x	= p2.normal_x;
            p0.normal_y	= p2.normal_y;
            p0.normal_z	= p2.normal_z;
        }
    }
    return cloud_ptr;
}

void RGBDFrame::savePCD(std::string path, Eigen::Matrix4d pose){
   // printf("saving pcd: %s\n",path.c_str());
    unsigned char * rgbdata = (unsigned char *)rgb.data;
    unsigned short * depthdata = (unsigned short *)depth.data;

    const unsigned int width	= camera->width; const unsigned int height	= camera->height;
    const double idepth			= camera->idepth_scale;
    const double cx				= camera->cx;		const double cy				= camera->cy;
    const double ifx			= 1.0/camera->fx;	const double ify			= 1.0/camera->fy;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr	cloud	(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud->width	= width;
    cloud->height	= height;
    cloud->points.resize(width*height);

    for(unsigned int w = 0; w < width; w++){
        for(unsigned int h = 0; h < height;h++){
            int ind = h*width+w;
            double z = idepth*double(depthdata[ind]);
            if(z > 0){
                pcl::PointXYZRGB p;
                p.x = (double(w) - cx) * z * ifx;
                p.y = (double(h) - cy) * z * ify;
                p.z = z;
                p.b = rgbdata[3*ind+0];
                p.g = rgbdata[3*ind+1];
                p.r = rgbdata[3*ind+2];
                cloud->points[ind] = p;
            }
        }
    }

    //Mat4f2RotTrans(const Eigen::Matrix4f &tf, Eigen::Quaternionf &q, Eigen::Vector4f &trans)
    Mat4f2RotTrans(pose.cast<float>(),cloud->sensor_orientation_,cloud->sensor_origin_);
    int success = pcl::io::savePCDFileBinaryCompressed(path,*cloud);
}

void RGBDFrame::save(std::string path){
    //printf("saving frame %i to %s\n",id,path.c_str());

    cv::imwrite( path+"_rgb.png", rgb );
    cv::imwrite( path+"_depth.png", depth );

    unsigned long buffersize = 19*sizeof(double);
    char* buffer = new char[buffersize];
    double * buffer_double = (double *)buffer;
    unsigned long * buffer_long = (unsigned long *)buffer;

    int counter = 0;
    buffer_double[counter++] = capturetime;
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            buffer_double[counter++] = pose(i,j);
        }
    }
    buffer_long[counter++] = sweepid;
    buffer_long[counter++] = camera->id;
    std::ofstream outfile (path+"_data.txt",std::ofstream::binary);
    outfile.write (buffer,buffersize);
    outfile.close();
    delete[] buffer;
}

RGBDFrame * RGBDFrame::load(Camera * cam, std::string path){
    printf("RGBDFrame * RGBDFrame::load(Camera * cam, std::string path)\n");

    std::streampos size;
    char * buffer;
    char buf [1024];
    std::string datapath = path+"_data.txt";
    std::ifstream file (datapath, std::ios::in | std::ios::binary | std::ios::ate);
    if (file.is_open()){
        size = file.tellg();
        buffer = new char [size];
        file.seekg (0, std::ios::beg);
        file.read (buffer, size);
        file.close();

        double * buffer_double = (double *)buffer;
        unsigned long * buffer_long = (unsigned long *)buffer;

        int counter = 0;
        double capturetime = buffer_double[counter++];
        Eigen::Matrix4d pose;
        for(int i = 0; i < 4; i++){
            for(int j = 0; j < 4; j++){
                pose(i,j) = buffer_double[counter++];
            }
        }
        int sweepid = buffer_long[counter++];
        int camera_id = buffer_long[counter++];

        cv::Mat rgb = cv::imread(path+"_rgb.png", -1);   // Read the file
        cv::Mat depth = cv::imread(path+"_depth.png", -1);   // Read the file

        RGBDFrame * frame = new RGBDFrame(cam,rgb,depth,capturetime,pose);
        frame->sweepid = sweepid;
        //printf("sweepid: %i",sweepid);

        return frame;
    }else{printf("cant open %s\n",(path+"/data.txt").c_str());}
}

}
