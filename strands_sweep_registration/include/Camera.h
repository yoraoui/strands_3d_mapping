#ifndef Camera_H_
#define Camera_H_

#include <vector>
#include <string>
#include <stdio.h>

#include "PixelFunction.h"

class Camera {
	public:
		float fx;
		float fy;
		float cx;
		float cy;
		
		int width;
		int height;
		
		int version;
		
		PixelFunction ** pixelsFunctions;

		unsigned int coefs_degree;
		unsigned int coefs_width;
		unsigned int coefs_height;
		double * multiplierCoeffs;
		
		Camera(	float fx_, float fy_, float cx_, float cy_,	int width_,	int height_);
		virtual ~Camera();

		virtual double * getCoeffs(unsigned int w, unsigned int h);
		virtual double getGridMultiplier(unsigned int w, unsigned int h, double z);
		virtual double getMultiplier(double w, double h, double z);
		virtual void save(std::string path);
		virtual void load(std::string path);
		virtual void show();
		virtual void print();
};

#endif
