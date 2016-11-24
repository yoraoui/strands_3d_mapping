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

int RGBDFrame::saveId = 0;

unsigned long RGBDFrame_id_counter;
RGBDFrame::RGBDFrame(){
	id = RGBDFrame_id_counter++;
	capturetime = 0;
	pose = Eigen::Matrix4d::Identity();
	keyval = "";
	soma_id = "";
    labels = 0;
}

bool updated = true;
void on_trackbar( int, void* ){updated = true;}

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

		DistanceWeightFunction2PPR3 * func = new DistanceWeightFunction2PPR3();
		func->zeromean				= true;
		func->maxp					= 0.99;
		func->startreg				= 0.0;
		func->debugg_print			= false;
		func->maxd					= 256.0;
		func->histogram_size		= 256;
		//func->fixed_histogram_size	= true;
		func->fixed_histogram_size	= false;
		func->update_size       	= true;
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

//RGBDFrame * RGBDFrame::clone(){
//    return new RGBDFrame(camera->clone(), rgb.clone(),depth.clone(),capturetime, pose, true);
//}

RGBDFrame * RGBDFrame::clone(){
	//    printf("clone()\n");
	//    exit(0);
	RGBDFrame * frame = new RGBDFrame();
	frame->camera = camera->clone();
	frame->rgb = rgb.clone();
	frame->depth = depth.clone();
	frame->keyval = keyval;
	frame->capturetime = capturetime;
	frame->pose = pose;
	frame->sweepid = sweepid;
	frame->det_dilate = det_dilate.clone();
	frame->normals = normals.clone();
	frame->depthedges = depthedges.clone();
	frame->de = de.clone();
	frame->ce = ce.clone();
	frame->ce = ce.clone();
	frame->ce = ce.clone();

	unsigned int width = camera->width;
	unsigned int height = camera->height;
	unsigned int nr_pixels = width*height;

	frame->connections = connections;
	frame->intersections = intersections;
	frame->nr_labels = nr_labels;
	frame->labels = new int[nr_pixels];
	for(int i = 0; i < nr_pixels; i++){frame->labels[i] = labels[i];}

	return frame;//new RGBDFrame(camera->clone(), rgb.clone(),depth.clone(),capturetime, pose, true);
}

RGBDFrame::RGBDFrame(Camera * camera_, cv::Mat rgb_, cv::Mat depth_, double capturetime_, Eigen::Matrix4d pose_, bool compute_normals, std::string savePath, bool compute_imgedges){
	soma_id = "";
    labels = 0;

	//printf("savepath: %s\n",savePath.c_str());
	bool verbose = false;
	if(verbose)
		printf("------------------------------\n");
	double startTime = getTime();
	double completeStartTime = getTime();
	keyval = "";

	sweepid = -1;
	id = RGBDFrame_id_counter++;
	camera = camera_;
	rgb = rgb_;
	depth = depth_;
	capturetime = capturetime_;
	pose = pose_;

	char savestr [1024];
	sprintf(savestr,"%s/frame%5.5i_",savePath.c_str(),saveId);
	if(savePath.size() != 0){saveId++;}


	if(savePath.size() != 0){
		cv::imwrite( std::string(savestr)+"rgb.png", rgb );
		cv::imwrite( std::string(savestr)+"depth.png", depth );
	}

	IplImage iplimg = rgb_;
	IplImage* img = &iplimg;

	unsigned int width = img->width;
	unsigned int height = img->height;
	unsigned int nr_pixels = width*height;
	const double idepth			= camera->idepth_scale;
	const double cx				= camera->cx;
	const double cy				= camera->cy;
	const double ifx			= 1.0/camera->fx;
	const double ify			= 1.0/camera->fy;


	connections.resize(1);
	connections[0].resize(1);
	connections[0][0] = 0;

	intersections.resize(1);
	intersections[0].resize(1);
	intersections[0][0] = 0;

	nr_labels = 1;
	labels = new int[nr_pixels];
	for(unsigned int i = 0; i < nr_pixels; i++){labels[i] = 0;}

	unsigned short * depthdata = (unsigned short *)depth.data;
	unsigned char * rgbdata = (unsigned char *)rgb.data;

	if(verbose)
		printf("%s::%i time: %5.5fs\n",__PRETTY_FUNCTION__,__LINE__,getTime()-startTime); startTime = getTime();

	de.create(height,width,CV_32FC3);
	float * dedata = (float*)de.data;
	for(unsigned int i = 0; i < 3*nr_pixels; i++){dedata[i] = 0;}

	ce.create(height,width,CV_32FC3);
	float * cedata = (float*)ce.data;
	for(unsigned int i = 0; i < 3*nr_pixels; i++){cedata[i] = 0;}

	depthedges.create(height,width,CV_8UC1);
	unsigned char * depthedgesdata = (unsigned char *)depthedges.data;
	for(unsigned int i = 0; i < nr_pixels; i++){depthedgesdata[i] = 0;}
	if(compute_imgedges){
		std::vector<double> Xvec;
		int step = 1;
		for(unsigned int w = step; w < width-step;w++){
			for(unsigned int h = step; h < height-step;h++){
				int ind = h*width+w;
				float z = idepth*float(depthdata[ind]);

				if(w > 2 && w < width-1){
					float z0 = idepth*float(depthdata[ind-2]);
					float z1 = idepth*float(depthdata[ind-1]);
					float z2 = z;
					float z3 = idepth*float(depthdata[ind+1]);

					float pred1 = z1 + (z1-z0);
					float pred2 = z2 - (z3-z2);

					if(pred1 > 0 || pred2 > 0){
						double n0 = z0*z0;
						double n1 = z1*z1;
						double n2 = z2*z2;
						double n3 = z3*z3;

						double meanpred = 0.5*(fabs(z2-pred1) + fabs(z1-pred2));
						double noise3 = sqrt(n1*n1+n2*n2);
						float pred3 = std::min(meanpred,fabs(z1-z2));
						dedata[3*ind+1] = pred3/noise3;
						Xvec.push_back(meanpred/noise3);
					}
				}

				if(h > 2 && h < height-1){
					float z0 = idepth*float(depthdata[ind-2*width]);
					float z1 = idepth*float(depthdata[ind-1*width]);
					float z2 = z;
					float z3 = idepth*float(depthdata[ind+1*width]);

					float pred1 = z1 + (z1-z0);
					float pred2 = z2 - (z3-z2);

					if(pred1 > 0 || pred2 > 0){
						double n0 = z0*z0;
						double n1 = z1*z1;
						double n2 = z2*z2;
						double n3 = z3*z3;

						double meanpred = 0.5*(fabs(z2-pred1) + fabs(z1-pred2));
						double noise3 = sqrt(n1*n1+n2*n2);
						float pred3 = std::min(meanpred,fabs(z1-z2));
						dedata[3*ind+2] = pred3/noise3;
						Xvec.push_back(meanpred/noise3);
					}
				}
			}
		}

		//	cv::namedWindow( "de", cv::WINDOW_AUTOSIZE );		cv::imshow( "de",	de);
		//	cv::namedWindow( "rgb", cv::WINDOW_AUTOSIZE );			cv::imshow( "rgb",	rgb);
		//	cv::waitKey(0);

		if(savePath.size() != 0){cv::imwrite( std::string(savestr)+"depth_diff.png", 4.0*255.0*de );}

		if(verbose)
			printf("%s::%i time: %5.5fs\n",__PRETTY_FUNCTION__,__LINE__,getTime()-startTime); startTime = getTime();

		Eigen::MatrixXd X = Eigen::MatrixXd::Zero(1,Xvec.size());
		for(unsigned int i = 0; i < Xvec.size();i++){X(0,i) = Xvec[i];}

		double stdval = 0;
		for(unsigned int i = 0; i < Xvec.size();i++){stdval += X(0,i)*X(0,i);}
		stdval = sqrt(stdval/double(Xvec.size()));

		DistanceWeightFunction2PPR3 * funcZ = new DistanceWeightFunction2PPR3();
		if(savePath.size() != 0){
			funcZ->savePath = std::string(savestr)+"funcZ.m";
		}
		funcZ->zeromean				= true;
		//funcZ->startreg				= 0.001;
		funcZ->startreg				= 0.0005;
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
				unsigned int ind = h*width+w;
				if(w > 1){dedata[3*ind+1] = 1-funcZ->getProb(dedata[3*ind+1]);}
				if(h > 1){dedata[3*ind+2] = 1-funcZ->getProb(dedata[3*ind+2]);}
			}
		}


		//	cv::namedWindow( "de", cv::WINDOW_AUTOSIZE );		cv::imshow( "de",	de);
		//	cv::namedWindow( "rgb", cv::WINDOW_AUTOSIZE );			cv::imshow( "rgb",	rgb);
		//	cv::waitKey(0);

		//	cv::namedWindow( "rgb", cv::WINDOW_AUTOSIZE );			cv::imshow( "rgb", rgb );
		//	cv::namedWindow( "normals", cv::WINDOW_AUTOSIZE );		cv::imshow( "normals", normals );
		//	cv::namedWindow( "depth", cv::WINDOW_AUTOSIZE );		cv::imshow( "depth", depth );
		//	cv::namedWindow( "depthedges", cv::WINDOW_AUTOSIZE );	cv::imshow( "depthedges", depthedges );
		//	if(stop){	cv::waitKey(0);}else{					cv::waitKey(30);}

		if(savePath.size() != 0){cv::imwrite( std::string(savestr)+"depth_prob.png", 255.0*de );}

		delete funcZ;

		if(verbose)
			printf("%s::%i time: %5.5fs\n",__PRETTY_FUNCTION__,__LINE__,getTime()-startTime); startTime = getTime();

		int blursize = 5;
		cv::Mat blur_rgb				= rgb.clone();
		cv::GaussianBlur( blur_rgb, blur_rgb, cv::Size(blursize,blursize), 0, 0, cv::BORDER_DEFAULT );


		if(savePath.size() != 0){cv::imwrite( std::string(savestr)+"blur_rgb.png", blur_rgb );}
		unsigned char * blurdata = (unsigned char *)blur_rgb.data;

		cv::Mat re;
		re.create(height,width,CV_32FC3);
		float * redata = (float*)re.data;
		for(unsigned int i = 0; i < 3*nr_pixels; i++){redata[i] = 0;}

		cv::Mat ge;
		ge.create(height,width,CV_32FC3);
		float * gedata = (float*)ge.data;
		for(unsigned int i = 0; i < 3*nr_pixels; i++){gedata[i] = 0;}

		cv::Mat be;
		be.create(height,width,CV_32FC3);
		float * bedata = (float*)be.data;
		for(unsigned int i = 0; i < 3*nr_pixels; i++){bedata[i] = 0;}

		std::vector<double> XvecR;
		std::vector<double> XvecG;
		std::vector<double> XvecB;

		for(unsigned int w = 1; w < width-1;w++){
			for(unsigned int h = 0; h < height;h++){
				unsigned int ind = h*width+w;
				bedata[3*ind+1] = fabs(blurdata[3*ind+0] - blurdata[3*(ind-1)+0]);
				XvecB.push_back(bedata[3*ind+1]);

				gedata[3*ind+1] = fabs(blurdata[3*ind+1] - blurdata[3*(ind-1)+1]);
				XvecG.push_back(gedata[3*ind+1]);

				redata[3*ind+1] = fabs(blurdata[3*ind+2] - blurdata[3*(ind-1)+2]);
				XvecR.push_back(redata[3*ind+1]);
			}
		}
		for(unsigned int w = 0; w < width;w++){
			for(unsigned int h = 1; h < height-1;h++){
				unsigned int ind = h*width+w;
				bedata[3*ind+2] = fabs(blurdata[3*ind+0] - blurdata[3*(ind-width)+0]);
				XvecB.push_back(bedata[3*ind+1]);

				gedata[3*ind+2] = fabs(blurdata[3*ind+1] - blurdata[3*(ind-width)+1]);
				XvecG.push_back(gedata[3*ind+1]);

				redata[3*ind+2] = fabs(blurdata[3*ind+2] - blurdata[3*(ind-width)+2]);
				XvecR.push_back(redata[3*ind+1]);
			}
		}


		if(savePath.size() != 0){cv::imwrite( std::string(savestr)+"re_diff.png", 3.0*re );}
		if(savePath.size() != 0){cv::imwrite( std::string(savestr)+"ge_diff.png", 3.0*ge );}
		if(savePath.size() != 0){cv::imwrite( std::string(savestr)+"be_diff.png", 3.0*be );}

		std::vector<double> src_dxdata;
		src_dxdata.resize(nr_pixels);
		for(unsigned int i = 0; i < nr_pixels;i++){src_dxdata[i] = 0;}

		std::vector<double> src_dydata;
		src_dydata.resize(nr_pixels);
		for(unsigned int i = 0; i < nr_pixels;i++){src_dydata[i] = 0;}

		std::vector<bool> maxima_dxdata;
		maxima_dxdata.resize(nr_pixels);
		for(unsigned int i = 0; i < nr_pixels;i++){maxima_dxdata[i] = 0;}

		std::vector<bool> maxima_dydata;
		maxima_dydata.resize(nr_pixels);
		for(unsigned int i = 0; i < nr_pixels;i++){maxima_dydata[i] = 0;}

		for(unsigned int w = 0; w < width;w++){
			for(unsigned int h = 0; h < height;h++){
				unsigned int ind = h*width+w;
				src_dxdata[ind] = 0;
				src_dydata[ind] = 0;
			}
		}

		unsigned int chans = 3;
		for(unsigned int c = 0; c < chans;c++){
			for(unsigned int w = 1; w < width;w++){
				for(unsigned int h = 1; h < height;h++){
					int ind = h*width+w;
					src_dxdata[ind] += fabs(float(blurdata[chans*ind+c] - blurdata[chans*(ind-1)	+c]) / 255.0)/3.0;
					src_dydata[ind] += fabs(float(blurdata[chans*ind+c] - blurdata[chans*(ind-width)+c]) / 255.0)/3.0;
				}
			}
		}

		for(unsigned int w = 1; w < width-1;w++){
			for(unsigned int h = 1; h < height-1;h++){
				int ind = h*width+w;
				maxima_dxdata[ind] = (src_dxdata[ind] > src_dxdata[ind-1]     && src_dxdata[ind] >= src_dxdata[ind+1]);
				maxima_dydata[ind] = (src_dydata[ind] > src_dydata[ind-width] && src_dydata[ind] >= src_dydata[ind+width]);
			}
		}

		double stdvalR = 0;
		for(unsigned int i = 0; i < XvecR.size();i++){stdvalR += XvecR[i]*XvecR[i];}
		stdvalR = sqrt(stdvalR/double(XvecR.size()));

		double stdvalG = 0;
		for(unsigned int i = 0; i < XvecG.size();i++){stdvalG += XvecG[i]*XvecG[i];}
		stdvalG = sqrt(stdvalG/double(XvecG.size()));

		double stdvalB = 0;
		for(unsigned int i = 0; i < XvecB.size();i++){stdvalB += XvecB[i]*XvecB[i];}
		stdvalB = sqrt(stdvalB/double(XvecB.size()));

		if(verbose)
			printf("%s::%i time: %5.5fs\n",__PRETTY_FUNCTION__,__LINE__,getTime()-startTime); startTime = getTime();

		GeneralizedGaussianDistribution * ggdfuncR	= new GeneralizedGaussianDistribution(true,false);
		ggdfuncR->nr_refineiters					= 1;
		GeneralizedGaussianDistribution * ggdfuncG	= new GeneralizedGaussianDistribution(true,false);
		ggdfuncG->nr_refineiters					= 1;
		GeneralizedGaussianDistribution * ggdfuncB	= new GeneralizedGaussianDistribution(true,false);
		ggdfuncB->nr_refineiters					= 1;

		DistanceWeightFunction2PPR3 * funcR = new DistanceWeightFunction2PPR3(ggdfuncR);
		DistanceWeightFunction2PPR3 * funcG = new DistanceWeightFunction2PPR3(ggdfuncG);
		DistanceWeightFunction2PPR3 * funcB = new DistanceWeightFunction2PPR3(ggdfuncB);



		if(savePath.size() != 0){
			funcR->savePath = std::string(savestr)+"funcR.m";
			funcG->savePath = std::string(savestr)+"funcG.m";
			funcB->savePath = std::string(savestr)+"funcB.m";
		}


		funcR->zeromean		= funcG->zeromean	= funcB->zeromean	= true;
		funcR->maxp			= funcG->maxp		= funcB->maxp		= 0.999999999999999999;
		funcR->startreg		= funcG->startreg	= funcB->startreg	= 0.5;
		funcR->debugg_print = funcG->debugg_print = funcB->debugg_print = false;
		funcR->maxd			= funcG->maxd		= funcB->maxd		= 256.0;
		funcR->histogram_size = funcG->histogram_size = funcB->histogram_size = 256;
		funcR->fixed_histogram_size	= funcG->fixed_histogram_size	= funcB->fixed_histogram_size	= true;
		funcR->startmaxd	= funcG->startmaxd	= funcB->startmaxd	= funcR->maxd;
		funcR->starthistogram_size	= funcG->starthistogram_size	= funcB->starthistogram_size	= funcR->histogram_size;
		//funcR->blurval		= funcG->blurval	= funcB->blurval	= 0.005;
		funcR->blurval		= funcG->blurval	= funcB->blurval	= 0.5;

		funcR->stdval2		= stdvalR;
		funcR->maxnoise		= stdvalR;
		funcG->stdval2		= stdvalG;
		funcG->maxnoise		= stdvalG;
		funcB->stdval2		= stdvalB;
		funcB->maxnoise		= stdvalB;

		funcR->reset();
		((DistanceWeightFunction2*)funcR)->computeModel(XvecR);
		funcG->reset();
		((DistanceWeightFunction2*)funcG)->computeModel(XvecG);
		funcB->reset();
		((DistanceWeightFunction2*)funcB)->computeModel(XvecB);

		for(unsigned int w = 1; w < width-1;w++){
			for(unsigned int h = 1; h < height-1;h++){
				int ind = h*width+w;
				if(w > 1){
					double pr = funcR->getProb(redata[3*ind+1]);
					double pg = funcG->getProb(gedata[3*ind+1]);
					double pb = funcB->getProb(bedata[3*ind+1]);
					double pc = pr*pg*pb/(pr*pg*pb+(1-pr)*(1-pg)*(1-pb));
					redata[3*ind+1] = 1-pr;
					gedata[3*ind+1] = 1-pg;
					bedata[3*ind+1] = 1-pb;
					cedata[3*ind+1] = 1-pc;
				}

				if(h > 1){
					double pr = funcR->getProb(redata[3*ind+2]);
					double pg = funcG->getProb(gedata[3*ind+2]);
					double pb = funcB->getProb(bedata[3*ind+2]);
					double pc = pr*pg*pb/(pr*pg*pb+(1-pr)*(1-pg)*(1-pb));
					redata[3*ind+2] = 1-pr;
					gedata[3*ind+2] = 1-pg;
					bedata[3*ind+2] = 1-pb;
					cedata[3*ind+2] = 1-pc;
				}
			}
		}

		delete funcR;
		delete funcG;
		delete funcB;

		if(savePath.size() != 0){cv::imwrite( std::string(savestr)+"re_prob.png", 127.0*re );}
		if(savePath.size() != 0){cv::imwrite( std::string(savestr)+"ge_prob.png", 127.0*ge );}
		if(savePath.size() != 0){cv::imwrite( std::string(savestr)+"be_prob.png", 127.0*be );}
		if(savePath.size() != 0){cv::imwrite( std::string(savestr)+"ce_prob.png", 127.0*ce );}

		if(verbose)
			printf("%s::%i time: %5.5fs\n",__PRETTY_FUNCTION__,__LINE__,getTime()-startTime); startTime = getTime();

		cv::Mat det;
		det.create(height,width,CV_8UC1);
		unsigned char * detdata = (unsigned char*)det.data;
		for(unsigned int i = 0; i < nr_pixels; i++){
			detdata[i] = 255*((dedata[3*i+1] > 0.5) || (dedata[3*i+2] > 0.5));
		}

		cv::Mat cenms;
		cenms.create(height,width,CV_32FC3);
		float * cenmsdata = (float*)cenms.data;
		for(unsigned int i = 0; i < nr_pixels; i++){
			double edgep = std::max(cedata[3*i+1]*maxima_dxdata[i],cedata[3*i+2]*maxima_dydata[i]);
			edgep = std::max(0.000000000000000000001,edgep);
			cenmsdata[3*i+0] = edgep;
			cenmsdata[3*i+1] = edgep;
			cenmsdata[3*i+2] = edgep;
		}


		if(savePath.size() != 0){cv::imwrite( std::string(savestr)+"ce_non_maxima_supress_prob.png", 255.0*cenms );}

		//	for(int i = 0; i < nr_pixels; i++){
		//		double edgep = std::max(cedata[3*i+1]*maxima_dxdata[i],cedata[3*i+2]*maxima_dydata[i]);
		//		edgep = std::max(0.00001,edgep);
		//		cenmsdata[3*i+0] = 0;
		//		cenmsdata[3*i+1] = std::max(0.00001,cenmsdata[3*i+1]);
		//		cenmsdata[3*i+2] = std::max(0.00001,edgep);
		//	}

		//		cv::namedWindow( "colour", cv::WINDOW_AUTOSIZE );		cv::imshow( "colour",	ce);
		//		cv::namedWindow( "colour+nms", cv::WINDOW_AUTOSIZE );	cv::imshow( "colour+nms",	cenms);
		//		cv::namedWindow( "rgb", cv::WINDOW_AUTOSIZE );			cv::imshow( "rgb",	rgb);
		//		cv::waitKey(0);

		ce = cenms;

		int dilation_size = 2;
		cv::dilate( det, det_dilate, getStructuringElement( cv::MORPH_RECT, cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ), cv::Point( dilation_size, dilation_size ) ) );
		unsigned char * det_dilatedata = (unsigned char*)det_dilate.data;

		//	depthedges.create(height,width,CV_8UC1);
		//	unsigned char * depthedgesdata = (unsigned char *)depthedges.data;
		for(unsigned int i = 0; i < nr_pixels; i++){
			//depthedgesdata[i] = 255*((dedata[3*i+1] > 0.5) || (dedata[3*i+2] > 0.5) || (cedata[3*i+1]*maxima_dxdata[i] > 0.5) || (cedata[3*i+2]*maxima_dydata[i] > 0.5));
			depthedgesdata[i] = 255*(((cedata[3*i+1]*maxima_dxdata[i] > 0.5) || (cedata[3*i+2]*maxima_dydata[i] > 0.5)) && (det_dilatedata[i] == 0));
		}
	}
	//	cv::namedWindow( "depthedges", cv::WINDOW_AUTOSIZE );			cv::imshow( "depthedges",	depthedges);
	//	cv::waitKey(0);

	if(verbose)
		printf("%s::%i time: %5.5fs\n",__PRETTY_FUNCTION__,__LINE__,getTime()-startTime); startTime = getTime();
	if(compute_normals){
		normals.create(height,width,CV_32FC3);
		float * normalsdata = (float *)normals.data;

		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr	cloud	(new pcl::PointCloud<pcl::PointXYZRGBA>);
		pcl::PointCloud<pcl::Normal>::Ptr	normals_cloud (new pcl::PointCloud<pcl::Normal>);
		cloud->width	= width;
		cloud->height	= height;
		cloud->points.resize(nr_pixels);

		for(unsigned int w = 0; w < width; w++){
			for(unsigned int h = 0; h < height;h++){
				unsigned int ind = h*width+w;
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

		pcl::IntegralImageNormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne;
		ne.setInputCloud(cloud);

		int MaxDepthChangeFactor = 20;
		int NormalSmoothingSize = 7;
		int depth_dependent_smoothing = 1;

		ne.setMaxDepthChangeFactor(0.001*double(MaxDepthChangeFactor));
		ne.setNormalSmoothingSize(NormalSmoothingSize);
		ne.setDepthDependentSmoothing (depth_dependent_smoothing);
		ne.compute(*normals_cloud);

		for(unsigned int w = 0; w < width; w++){
			for(unsigned int h = 0; h < height;h++){
				unsigned int ind = h*width+w;
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

		if(savePath.size() != 0){
			cv::Mat out;
			out.create(height,width,CV_8UC3);
			unsigned char * outdata = out.data;

			for(unsigned int w = 0; w < width; w++){
				for(unsigned int h = 0; h < height;h++){
					unsigned int ind = h*width+w;
					pcl::Normal		p2		= normals_cloud->points[ind];
					if(!isnan(p2.normal_x)){
						outdata[3*ind+0]	= fabs(255.0*p2.normal_x);
						outdata[3*ind+1]	= fabs(255.0*p2.normal_y);
						outdata[3*ind+2]	= fabs(255.0*p2.normal_z);
					}else{
						outdata[3*ind+0]	= 0;
						outdata[3*ind+1]	= 0;
						outdata[3*ind+2]	= 0;
					}
				}
			}

			cv::imwrite( std::string(savestr)+"normals.png", out );
		}
	}
	if(verbose)
		printf("%s::%i time: %5.5fs\n",__PRETTY_FUNCTION__,__LINE__,getTime()-startTime); startTime = getTime();
	//printf("complete time to create RGBD image: %5.5fs\n",getTime()-completeStartTime);

	//getImageProbs();
	if(savePath.size() != 0){
		saveCombinedImages(std::string(savestr)+"Combined.png");
		saveCombinedProcessedImages(std::string(savestr)+"CombinedProcessed.png");
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

pcl::PointCloud<pcl::PointXYZRGB>::Ptr RGBDFrame::getSmallPCLcloud(){
	unsigned char * rgbdata = (unsigned char *)rgb.data;
	unsigned short * depthdata = (unsigned short *)depth.data;

	const unsigned int width	= camera->width; const unsigned int height	= camera->height;
	const double idepth			= camera->idepth_scale;
	const double cx				= camera->cx;		const double cy				= camera->cy;
	const double ifx			= 1.0/camera->fx;	const double ify			= 1.0/camera->fy;

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr	cloud	(new pcl::PointCloud<pcl::PointXYZRGB>);
	cloud->width	= 0;
	cloud->height	= 1;
	cloud->points.reserve(width*height);
	for(unsigned int w = 0; w < width; w++){
		for(unsigned int h = 0; h < height;h++){
			int ind = h*width+w;
			double z = idepth*double(depthdata[ind]);
			if(z > 0){
				pcl::PointXYZRGB p;
				p.b = rgbdata[3*ind+0];
				p.g = rgbdata[3*ind+1];
				p.r = rgbdata[3*ind+2];
				p.x = (double(w) - cx) * z * ifx;
				p.y = (double(h) - cy) * z * ify;
				p.z = z;
				cloud->points.push_back(p);
			}
		}
	}
	cloud->width	= cloud->points.size();
	return cloud;
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

			if(depthdata[ind] == 0){
				p0.x		= 0;
				p0.y		= 0;
				p0.z		= 0;
			}

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

std::string type2str(int type) {
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch ( depth ) {
		case CV_8U:  r = "8U"; break;
		case CV_8S:  r = "8S"; break;
		case CV_16U: r = "16U"; break;
		case CV_16S: r = "16S"; break;
		case CV_32S: r = "32S"; break;
		case CV_32F: r = "32F"; break;
		case CV_64F: r = "64F"; break;
		default:     r = "User"; break;
	}

	r += "C";
	r += (chans+'0');

	return r;
}

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>


void RGBDFrame::save(std::string path){
	//printf("saving frame %i to %s\n",id,path.c_str());

	cv::imwrite( path+"_rgb.png", rgb );
	cv::imwrite( path+"_depth.png", depth );
	cv::imwrite( path+"_det_dilate.png", det_dilate );
	cv::imwrite( path+"_depthedges.png", depthedges );

	const unsigned int width	= camera->width;
	const unsigned int height	= camera->height;

	std::ofstream normalsoutfile	(path+"_normalsdata.txt",std::ofstream::binary);
	normalsoutfile.write ((char*)(normals.data),width*height*3*sizeof(float));
	normalsoutfile.close();

	std::ofstream ceoutfile			(path+"_cedata.txt",std::ofstream::binary);
	ceoutfile.write ((char*)(ce.data),width*height*3*sizeof(float));
	ceoutfile.close();

	std::ofstream deoutfile			(path+"_dedata.txt",std::ofstream::binary);
	deoutfile.write ((char*)(de.data),width*height*3*sizeof(float));
	deoutfile.close();

	unsigned long buffersize = 22*sizeof(double)+keyval.length()+soma_id.length();
	char* buffer = new char[buffersize];
	double * buffer_double = (double *)buffer;
	unsigned long * buffer_long = (unsigned long *)buffer;

	int counter = 0;
	buffer_double[counter++] = capturetime; // CaptureTime
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++){
			buffer_double[counter++] = pose(i,j);
		}
	}										// Pose
	buffer_long[counter++] = sweepid;		// Sweepid
	buffer_long[counter++] = camera->id;	// Cameraid
	buffer_long[counter++] = id;	// id
	buffer_long[counter++] = keyval.length();	// keyval.length()
	buffer_long[counter++] = soma_id.length();	// soma_id.length()

	unsigned int count4 = sizeof(double)*counter;
	for(unsigned int i = 0; i < keyval.length();i++){
		buffer[count4++] = keyval[i];
	}
	for(unsigned int i = 0; i < soma_id.length();i++){
		buffer[count4++] = soma_id[i];
	}

	std::ofstream outfile (path+"_data.txt",std::ofstream::binary);
	outfile.write (buffer,buffersize);
	outfile.close();
	delete[] buffer;
}

RGBDFrame * RGBDFrame::load(Camera * cam, std::string path){
	double startTime = getTime();

	const unsigned int width	= cam->width;
	const unsigned int height	= cam->height;

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
		int id = buffer_long[counter++];
		int keyvallength = buffer_long[counter++];
		int soma_idlength = buffer_long[counter++];


		std::string keyval;
		keyval.resize(keyvallength);
		std::string soma_id;
		soma_id.resize(soma_idlength);

		unsigned int count4 = sizeof(double)*counter;
		for(unsigned int i = 0; i < keyvallength;i++){
			keyval[i] = buffer[count4++];
		}

		for(unsigned int i = 0; i < soma_idlength;i++){
			soma_id[i] = buffer[count4++];
		}

		cv::Mat rgb = cv::imread(path+"_rgb.png", -1);   // Read the file
		cv::Mat depth = cv::imread(path+"_depth.png", -1);   // Read the file
		cv::Mat det_dilate = cv::imread(path+"_det_dilate.png", -1);   // Read the file
		cv::Mat depthedges = cv::imread(path+"_depthedges.png", -1);   // Read the file

		RGBDFrame * frame = new RGBDFrame();
		frame->rgb = rgb;
		frame->depth = depth;
		frame->det_dilate = det_dilate;
		frame->depthedges = depthedges;
		frame->camera = cam;
		frame->capturetime = capturetime;
		frame->sweepid = sweepid;
		frame->pose = pose;

		frame->normals.create(height,width,CV_32FC3);
		frame->ce.create(height,width,CV_32FC3);
		frame->de.create(height,width,CV_32FC3);

		std::ifstream normalsfile (path+"_normalsdata.txt", std::ios::in | std::ios::binary | std::ios::ate);
		if (normalsfile.is_open()){
			size = normalsfile.tellg();
			normalsfile.seekg (0, std::ios::beg);
			normalsfile.read ((char*)(frame->normals.data), size);
			normalsfile.close();
		}

		std::ifstream cefile (path+"_cedata.txt", std::ios::in | std::ios::binary | std::ios::ate);
		if (cefile.is_open()){
			size = cefile.tellg();
			cefile.seekg (0, std::ios::beg);
			cefile.read ((char*)(frame->ce.data), size);
			cefile.close();
		}

		std::ifstream defile (path+"_dedata.txt", std::ios::in | std::ios::binary | std::ios::ate);
		if (defile.is_open()){
			size = defile.tellg();
			defile.seekg (0, std::ios::beg);
			defile.read ((char*)(frame->de.data), size);
			defile.close();
		}

//printf("%s::%i time: %5.5fs\n",__PRETTY_FUNCTION__,__LINE__,getTime()-startTime); startTime = getTime();
		return frame;
	}else{printf("cant open %s\n",(path+"/data.txt").c_str());}
}

std::vector<ReprojectionResult> RGBDFrame::getReprojections(std::vector<superpoint> & spvec, Eigen::Matrix4d cp, bool * maskvec, bool useDet){
	std::vector<ReprojectionResult> ret;

	unsigned char  * dst_detdata		= (unsigned char	*)(det_dilate.data);
	unsigned char  * dst_rgbdata		= (unsigned char	*)(rgb.data);
	unsigned short * dst_depthdata		= (unsigned short	*)(depth.data);
	float		   * dst_normalsdata	= (float			*)(normals.data);

	float m00 = cp(0,0); float m01 = cp(0,1); float m02 = cp(0,2); float m03 = cp(0,3);
	float m10 = cp(1,0); float m11 = cp(1,1); float m12 = cp(1,2); float m13 = cp(1,3);
	float m20 = cp(2,0); float m21 = cp(2,1); float m22 = cp(2,2); float m23 = cp(2,3);

	Camera * dst_camera				= camera;
	const unsigned int dst_width	= dst_camera->width;
	const unsigned int dst_height	= dst_camera->height;
	const float dst_idepth			= dst_camera->idepth_scale;
	const float dst_cx				= dst_camera->cx;
	const float dst_cy				= dst_camera->cy;
	const float dst_fx				= dst_camera->fx;
	const float dst_fy				= dst_camera->fy;
	const float dst_ifx				= 1.0/dst_camera->fx;
	const float dst_ify				= 1.0/dst_camera->fy;
	const unsigned int dst_width2	= dst_camera->width  - 2;
	const unsigned int dst_height2	= dst_camera->height - 2;

	unsigned long nr_data = spvec.size();
	for(unsigned long src_ind = 0; src_ind < nr_data;src_ind++){
		superpoint & sp = spvec[src_ind];
		if(sp.point_information == 0){continue;}

		float src_x = sp.point(0);
		float src_y = sp.point(1);
		float src_z = sp.point(2);

		float src_nx = sp.normal(0);
		float src_ny = sp.normal(1);
		float src_nz = sp.normal(2);

		float tx	= m00*src_x + m01*src_y + m02*src_z + m03;
		float ty	= m10*src_x + m11*src_y + m12*src_z + m13;
		float tz	= m20*src_x + m21*src_y + m22*src_z + m23;

		float itz	= 1.0/tz;
		float dst_w	= dst_fx*tx*itz + dst_cx;
		float dst_h	= dst_fy*ty*itz + dst_cy;

		if((dst_w > 0) && (dst_h > 0) && (dst_w < dst_width2) && (dst_h < dst_height2)){
			unsigned int dst_ind = unsigned(dst_h+0.5) * dst_width + unsigned(dst_w+0.5);
			if(maskvec != 0 && maskvec[dst_ind] == 0){continue;}

			float dst_z = dst_idepth*float(dst_depthdata[dst_ind]);
			float dst_nx = dst_normalsdata[3*dst_ind+0];
			if(dst_z > 0 && dst_nx != 2){
				if(useDet && dst_detdata[dst_ind] != 0){continue;}
				float dst_ny = dst_normalsdata[3*dst_ind+1];
				float dst_nz = dst_normalsdata[3*dst_ind+2];

				float dst_x = (float(dst_w) - dst_cx) * dst_z * dst_ifx;
				float dst_y = (float(dst_h) - dst_cy) * dst_z * dst_ify;

				float tnx	= m00*src_nx + m01*src_ny + m02*src_nz;
				float tny	= m10*src_nx + m11*src_ny + m12*src_nz;
				float tnz	= m20*src_nx + m21*src_ny + m22*src_nz;

				double residualZ = mysign(dst_z-tz)*fabs(tnx*(dst_x-tx) + tny*(dst_y-ty) + tnz*(dst_z-tz));
				double residualD2 = (dst_x-tx)*(dst_x-tx) + (dst_y-ty)*(dst_y-ty) + (dst_z-tz)*(dst_z-tz);
				double residualR =  dst_rgbdata[3*dst_ind + 2] - sp.feature(0);
				double residualG =  dst_rgbdata[3*dst_ind + 1] - sp.feature(1);
				double residualB =  dst_rgbdata[3*dst_ind + 0] - sp.feature(2);
				double angle = tnx*dst_nx + tny*dst_ny + tnz*dst_nz;
				ret.push_back(ReprojectionResult (src_ind, dst_ind, angle, residualZ,residualD2, residualR, residualG, residualB, getNoise(dst_z), 1.0));
			}
		}
	}
	return ret;
}

std::vector<superpoint> RGBDFrame::getSuperPoints(Eigen::Matrix4d cp, unsigned int step, bool zeroinclude){
	unsigned char  * rgbdata		= (unsigned char	*)(rgb.data);
	unsigned short * depthdata		= (unsigned short	*)(depth.data);
	float		   * normalsdata	= (float			*)(normals.data);

	float m00 = cp(0,0); float m01 = cp(0,1); float m02 = cp(0,2); float m03 = cp(0,3);
	float m10 = cp(1,0); float m11 = cp(1,1); float m12 = cp(1,2); float m13 = cp(1,3);
	float m20 = cp(2,0); float m21 = cp(2,1); float m22 = cp(2,2); float m23 = cp(2,3);

	const unsigned int width	= camera->width;
	const unsigned int height	= camera->height;
	const float idepth			= camera->idepth_scale;
	const float cx				= camera->cx;
	const float cy				= camera->cy;
	const float ifx				= 1.0/camera->fx;
	const float ify				= 1.0/camera->fy;

	std::vector<superpoint> ret;
	ret.resize((width/step)*(height/step));
	unsigned long count = 0;

	for(unsigned long h = 0; h < height;h += step){
		for(unsigned long w = 0; w < width;w += step){
			unsigned int ind = h * width + w;
			Eigen::Vector3f rgb (rgbdata[3*ind+0],rgbdata[3*ind+1],rgbdata[3*ind+2]);
			float z		= idepth*float(depthdata[ind]);
			float nx	= normalsdata[3*ind+0];
			if(z > 0 && nx != 2){
				float x = (float(w) - cx) * z * ifx;
				float y = (float(h) - cy) * z * ify;
				float ny = normalsdata[3*ind+1];
				float nz = normalsdata[3*ind+2];

				float tx	= m00*x + m01*y + m02*z + m03;
				float ty	= m10*x + m11*y + m12*z + m13;
				float tz	= m20*x + m21*y + m22*z + m23;

				float tnx	= m00*nx + m01*ny + m02*nz;
				float tny	= m10*nx + m11*ny + m12*nz;
				float tnz	= m20*nx + m21*ny + m22*nz;

				ret[count++]	= superpoint(Vector3f(tx,ty,tz),Vector3f(tnx,tny,tnz),rgb, getInformation(z), 1, 0);
			}else{
				if(zeroinclude){
					ret[count++]	= superpoint(Eigen::Vector3f(0,0,0),Eigen::Vector3f(0,0,0),rgb, 0, 1, 0);
				}
			}
		}
	}
	ret.resize(count);
	return ret;
}

std::vector< std::vector<float> > RGBDFrame::getImageProbs(bool depthonly){
	const unsigned int width = camera->width;
	const unsigned int height = camera->height;
	const unsigned int nr_pixels = width*height;


	float * cedata = (float*)ce.data;
	float * dedata = (float*)de.data;
	unsigned char * det_dilatedata = det_dilate.data;

	std::vector<float> dxc;
	dxc.resize(nr_pixels);

	std::vector<float> dyc;
	dyc.resize(nr_pixels);

	for(unsigned int i = 0; i < nr_pixels;i++){
		//dxc[i] = 1.0-std::max(dedata[3*i+1],cedata[3*i+1]);
		//dyc[i] = 1.0-std::max(dedata[3*i+2],cedata[3*i+2]);
		//ProcessedImages



		if(!depthonly){
			if(!det_dilatedata[i]){
				dxc[i] = 1.0-std::max(dedata[3*i+1],0.8f*cedata[3*i+1]);
				dyc[i] = 1.0-std::max(dedata[3*i+2],0.8f*cedata[3*i+2]);
			}else{
				dxc[i] = 1.0-std::max(dedata[3*i+1],0.8f*cedata[3*i+1]);
				dyc[i] = 1.0-std::max(dedata[3*i+2],0.8f*cedata[3*i+2]);
			}
		}else{
			dxc[i] = 1.0-dedata[3*i+1];
			dyc[i] = 1.0-dedata[3*i+2];
		}

	}

	//	cv::namedWindow( "Colour edges"	, cv::WINDOW_AUTOSIZE );	cv::imshow( "Colour edges",	ce);
	//	cv::namedWindow( "Depth edges"	, cv::WINDOW_AUTOSIZE );	cv::imshow( "Depth edges",	de);
	//	cv::namedWindow( "rgb"			, cv::WINDOW_AUTOSIZE );	cv::imshow( "rgb",			rgb );
	//	cv::namedWindow( "det_dilate"	, cv::WINDOW_AUTOSIZE );	cv::imshow( "det_dilate",	det_dilate );
	//	cv::waitKey(0);

	std::vector< std::vector<float> > probs2;
	probs2.push_back(dxc);
	probs2.push_back(dyc);
	return probs2;
}

void RGBDFrame::saveCombinedImages(std::string path){
	unsigned int width = camera->width;
	unsigned int height = camera->height;
	cv::Mat combined;
	combined.create(camera->height,2*camera->width,CV_8UC3);
	unsigned char   * combineddata   = combined.data;
	unsigned char   * rgbdata   = rgb.data;
	unsigned short  * depthdata = (unsigned short *)(depth.data);

	for(unsigned long h = 0; h < height;h++){
		for(unsigned long w = 0; w < width;w++){
			combineddata[3*(h*(2*width)+w)+0] = rgbdata[3*(h*width+w)+0];
			combineddata[3*(h*(2*width)+w)+1] = rgbdata[3*(h*width+w)+1];
			combineddata[3*(h*(2*width)+w)+2] = rgbdata[3*(h*width+w)+2];
			combineddata[3*(h*(2*width)+w+width)+0] = depthdata[h*width+w]/256;
			combineddata[3*(h*(2*width)+w+width)+1] = depthdata[h*width+w]/256;
			combineddata[3*(h*(2*width)+w+width)+2] = depthdata[h*width+w]/256;
		}
	}

	cv::imwrite( path, combined );
}
void RGBDFrame::saveCombinedProcessedImages(std::string path){
	unsigned int width = camera->width;
	unsigned int height = camera->height;
	cv::Mat combined;
	combined.create(camera->height,3*camera->width,CV_8UC3);
	unsigned char   * combineddata   = combined.data;
	float  * normalsdata = (float *)(normals.data);
	float  * cedata = (float *)(ce.data);
	float  * dedata = (float *)(de.data);

	for(unsigned long h = 0; h < height;h++){
		for(unsigned long w = 0; w < width;w++){
			if(normalsdata[3*(h*width+w)+0] <= 1){
				combineddata[3*(h*(3*width)+w+0*width)+0] = 255.0*fabs(normalsdata[3*(h*width+w)+0]);
				combineddata[3*(h*(3*width)+w+0*width)+1] = 255.0*fabs(normalsdata[3*(h*width+w)+1]);
				combineddata[3*(h*(3*width)+w+0*width)+2] = 255.0*fabs(normalsdata[3*(h*width+w)+2]);
			}else{
				combineddata[3*(h*(3*width)+w+0*width)+0] = 0;
				combineddata[3*(h*(3*width)+w+0*width)+1] = 0;
				combineddata[3*(h*(3*width)+w+0*width)+2] = 0;
			}
			combineddata[3*(h*(3*width)+w+1*width)+0] = 255.0*cedata[3*(h*width+w)+0];
			combineddata[3*(h*(3*width)+w+1*width)+1] = 255.0*cedata[3*(h*width+w)+1];
			combineddata[3*(h*(3*width)+w+1*width)+2] = 255.0*cedata[3*(h*width+w)+2];
			combineddata[3*(h*(3*width)+w+2*width)+0] = 255.0*dedata[3*(h*width+w)+0];
			combineddata[3*(h*(3*width)+w+2*width)+1] = 255.0*dedata[3*(h*width+w)+1];
			combineddata[3*(h*(3*width)+w+2*width)+2] = 255.0*dedata[3*(h*width+w)+2];
		}
	}

	cv::imwrite( path, combined );

	//    cv::namedWindow( "combined"	, cv::WINDOW_AUTOSIZE );
	//    cv::imshow( "combined",	combined );
	//    cv::waitKey(0);
}

}
