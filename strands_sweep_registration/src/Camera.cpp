#include "../include/strands_sweep_registration/Camera.h"
#include <fstream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.hpp"
#include "opencv/highgui.h"

Camera::Camera(float fx_, float fy_, float cx_, float cy_,	int width_,	int height_){
	fx = fx_;
	fy = fy_;
	cx = cx_;
	cy = cy_;
	
	width = width_;
	height = height_;
	
	pixelsFunctions = new PixelFunction*[width*height];
	for(int i = 0; i < width*height; i++){
		pixelsFunctions[i] = new PixelFunction(); 
	}

	version = 2;

//	coefs_degree = 1;
//	coefs_width = width/40;
//	coefs_height = height/40;


//	multiplierCoeffs = new double[coefs_degree*coefs_width*coefs_height];
//	for(int w = 0; w < width; w++){
//		for(int h = 0; h < height; h++){
//			double * coeffs = getCoeffs(w,h);
//			coeffs[0] = 1.0;
//			for(unsigned int i = 1; i < coefs_degree; i++){
//				coeffs[i] = 0.0;
//			}
//		}
//	}

	//getMultiplier(55, 55, 1);
};

Camera::~Camera(){
	//delete[] multiplierCoeffs;

	for(int i = 0; i < width*height; i++){
		delete pixelsFunctions[i]; 
	}
	delete[] pixelsFunctions;
}

double * Camera::getCoeffs(unsigned int w, unsigned int h){
	return multiplierCoeffs + coefs_degree*(coefs_width*h + w);
}

double Camera::getGridMultiplier(unsigned int w, unsigned int h, double z){
	double * coeffs = getCoeffs(w,h);
	return 0;
}

double Camera::getMultiplier(double w, double h, double z){
	double wgrid = w/(coefs_width+1);
	double hgrid = h/(coefs_height+1);

	double pw = wgrid-double(int(wgrid));
	double ph = hgrid-double(int(hgrid));

	printf("%f %f -> %f %f -> %f %f\n",w,h,wgrid,hgrid,pw,ph);
exit(0);
	return 1;
}

void Camera::save(std::string path){
	std::vector<char *> pixeldata_char;
	std::vector<int> pixeldata_counter;
	for(int i = 0; i < width*height; i++){pixelsFunctions[i]->addOutput(pixeldata_char, pixeldata_counter);}
	
	int length = 4*sizeof(float)+3*sizeof(int)+1;
	int start_write_pixel_pos = length;
	for(unsigned int i = 0; i < pixeldata_counter.size(); i++){length += pixeldata_counter.at(i);}
	
	char * buffer			= new char[length];
	int * buffer_int		= (int *) buffer;
	float * buffer_float	= (float *) buffer;
	
	buffer_int[0] = version;
	buffer_int[1] = width;
	buffer_int[2] = height;
	
	buffer_float[3] = cx;
	buffer_float[4] = cy;
	buffer_float[5] = fx;
	buffer_float[6] = fy;
	
	int ind = start_write_pixel_pos;
	for(unsigned int i = 0; i < pixeldata_counter.size(); i++){
		int current_length = pixeldata_counter.at(i);
		char * current_data = pixeldata_char.at(i);
		for(int j = 0; j < current_length; j++){
			buffer[ind++] = current_data[j];
		}
		delete[] current_data;
	}
	
	std::ofstream outfile (path.c_str(),std::ofstream::binary);
	outfile.write (buffer,length);
	outfile.close();
	delete[] buffer;
}

void Camera::load(std::string path){
	std::streampos size;
	char * buffer;

	std::ifstream file (path.c_str(), std::ios::in|std::ios::binary|std::ios::ate);
	if (file.is_open()){
		size = file.tellg();
		buffer = new char [size];
		file.seekg (0, std::ios::beg);
		file.read (buffer, size);
		file.close();

		int * buffer_int		= (int *) buffer;
		float * buffer_float	= (float *) buffer;
		
		version = buffer_int[0];
		width	= buffer_int[1];
		height	= buffer_int[2];
	
		cx		= buffer_float[3];
		cy		= buffer_float[4];
		fx		= buffer_float[5];
		fy		= buffer_float[6];
		
		int pos = 4*sizeof(float)+3*sizeof(int)+1;
		for(int i = 0; i < width*height; i++){pixelsFunctions[i]->load(&(buffer[pos]), pos);}
	
		delete[] buffer;
	}else{ printf("Unable to open camera file\n");}
}

void Camera::print(){
	printf("--------------------------------------------------\n");
	printf("Camera paramters:\n");
	printf("version: %i\n",version);
	printf("resolution: %i x %i\n",width,height);
	printf("cx: %f cy: %f fx: %f fy: %f\n",cx,cy,fx,fy);
}


void Camera::show(){
	print();
	
	IplImage * dimg = cvCreateImage(cvSize(width,height), IPL_DEPTH_8U, 3);
	cvRectangle(dimg,cvPoint(0,0),cvPoint(width,height),cvScalar(0,0,0,0),-1 , 8, 0);
	
	double sum = 0;
	double minval = 1;
	double maxval = 1;
	for(int w = 0; w < width; w+=1){
		for(int h = 0; h < height; h+=1){
			double d_mul = pixelsFunctions[h*width+w]->d_mul;
			minval = std::min(minval,d_mul);
			maxval = std::max(maxval,d_mul);
			sum += d_mul;
			double br = 125+5000*(d_mul-1);
			cvRectangle(dimg,cvPoint(w,h),cvPoint(w,h),cvScalar(br,br,br,0),-1 , 8, 0);
		}
	}
	printf("mean d_mul: %f min: %f max: %f\n",sum/double(width*height),minval,maxval);
	printf("--------------------------------------------------\n");
	cvShowImage("dimg",dimg);
	cvWaitKey(50);
	cvReleaseImage(&dimg);
}
