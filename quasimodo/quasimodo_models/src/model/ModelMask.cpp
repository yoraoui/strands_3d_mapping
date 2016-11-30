#include "model/ModelMask.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "highgui.h"

namespace reglib
{

int ModelMask_id = 0;
ModelMask::ModelMask(){}
ModelMask::ModelMask(cv::Mat mask_, std::string label_){

	savepath = "";
	mask = mask_;
	label = label_;
	sweepid = -1;
	id = ModelMask_id++;
	using namespace cv;
	unsigned char * maskdata = (unsigned char *)mask.data;
	width = mask.cols;
	height = mask.rows;

	maskvec = new bool[width*height];
	for(int i = 0; i < width*height; i++){maskvec[i] = maskdata[i] != 0;}

	int step = 0;
	for(int w = step; w < (width-1)-step; w++){
		for(int h = step; h < (height-1)-step; h++){
			if(maskdata[h*width+w] != 0){
				bool isok =  true;
				for(int w2 = w-step; w2 <= w+step; w2++){
					for(int h2 = h-step; h2 <= h+step; h2++){
						isok = isok && (maskdata[h2*width+w2] != 0);
					}
				}
				if(isok){
					testw.push_back(w);
					testh.push_back(h);
				}
			}
		}
	}

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

void ModelMask::saveFast(std::string path){
double startTime = getTime();
	savepath = path;

	cv::Mat mask = getMask();
	std::ofstream maskoutfile	(path+"mask.bin",std::ofstream::binary);
	maskoutfile.write ((char*)(mask.data),width*height*sizeof(char));
	maskoutfile.close();

	unsigned int nr_testw_size = testw.size();

	unsigned long buffersize = (6+2*nr_testw_size)*sizeof(unsigned int)+label.length();
	char * buffer = new char[buffersize];
	unsigned int * buffer_int = (unsigned int *)buffer;
	int counter = 0;
	buffer_int[counter++] = id;
	buffer_int[counter++] = width;
	buffer_int[counter++] = height;
	buffer_int[counter++] = sweepid;
	buffer_int[counter++] = label.length();
	buffer_int[counter++] = nr_testw_size;

	for(unsigned int i = 0; i < nr_testw_size;i++){
		buffer_int[counter++] = testh[i];
	}
	for(unsigned int i = 0; i < nr_testw_size;i++){
		buffer_int[counter++] = testh[i];
	}

	unsigned int count4 = sizeof(unsigned int)*counter;
	for(unsigned int i = 0; i < label.length();i++){
		buffer[count4++] = label[i];
	}

	std::ofstream outfile (path+"data.bin",std::ofstream::binary);
	outfile.write (buffer,buffersize);
	outfile.close();
	delete[] buffer;
//printf("ModelMask::saveFast(%s): %5.5fs\n",path.c_str(),getTime()-startTime);
}

ModelMask * ModelMask::loadFast(std::string path){
	double startTime = getTime();

	std::streampos size;
	char * buffer;
	std::ifstream file (path+"data.bin", std::ios::in | std::ios::binary | std::ios::ate);
	if (file.is_open()){
		size = file.tellg();
		buffer = new char [size];
		file.seekg (0, std::ios::beg);
		file.read (buffer, size);
		file.close();

		unsigned int * buffer_int = (unsigned int *)buffer;

		int counter = 0;
		unsigned int id = buffer_int[counter++];
		unsigned int width = buffer_int[counter++];
		unsigned int height = buffer_int[counter++];
		unsigned int sweepid = buffer_int[counter++];
		unsigned int labellength = buffer_int[counter++];
		unsigned int nr_testw_size = buffer_int[counter++];

		std::vector<int> testw;
		std::vector<int> testh;
		testw.resize(nr_testw_size);
		testh.resize(nr_testw_size);
		for(unsigned int i = 0; i < nr_testw_size;i++){
			testw[i] = buffer_int[counter++];
		}
		for(unsigned int i = 0; i < nr_testw_size;i++){
			testh[i] = buffer_int[counter++];
		}

		std::string label = "";
		label.resize(labellength);

		unsigned int count4 = sizeof(unsigned int)*counter;
		for(unsigned int i = 0; i < labellength;i++){
			label[i] = buffer[count4++];
		}
		delete[] buffer;

		std::ifstream maskfile (path+"mask.bin", std::ios::in | std::ios::binary | std::ios::ate);
		if (maskfile.is_open()){
			size = maskfile.tellg();
			maskfile.seekg (0, std::ios::beg);
			cv::Mat mask;
			mask.create(height,width,CV_8UC1);
			maskfile.read ((char*)(mask.data), size);
			maskfile.close();


			unsigned char * maskdata = (unsigned char *)mask.data;
			unsigned int nr_pixels = width*height;
			bool * maskvec = new bool[nr_pixels];
			for(int i = 0; i < nr_pixels; i++){maskvec[i] = maskdata[i] != 0;}

			ModelMask * mm = new ModelMask();
			mm->savepath = path;
			mm->id = id;
			mm->label = label;
			mm->mask = mask;
			mm->width = width;
			mm->height = height;
			mm->maskvec = maskvec;
			mm->testw = testw;
			mm->testh = testh;
			mm->sweepid = sweepid;

			//printf("ModelMask::loadFast(%s): %5.5fs\n",path.c_str(),getTime()-startTime);
			return mm;
		}else{printf("fail to open modelmaska mask\n");}
	}else{printf("fail to open modelmaska data\n");}
	//printf("ModelMask::loadFast(%s): %5.5fs\n",path.c_str(),getTime()-startTime);
	return 0;
}


}

