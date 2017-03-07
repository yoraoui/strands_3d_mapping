#include "ModelStorage/ModelStorage.h"
#include "Util/Util.h"

class CameraOptimizer{
public:
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    int visualizationLvl;
    int z_slider;

    CameraOptimizer(){
        visualizationLvl = 0;
        z_slider = 1000;
    }
    ~CameraOptimizer(){}

    void setVisualization(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_, int visualizationLvl_ = 1){
        visualizationLvl = visualizationLvl_;
        viewer = viewer_;
    }

	cv::Mat getPixelReliability(reglib::RGBDFrame * src){
		unsigned char  * src_detdata		= (unsigned char	*)(src->det_dilate.data);
		unsigned short * src_depthdata		= (unsigned short	*)(src->depth.data);
		float		   * src_normalsdata	= (float			*)(src->normals.data);

		reglib::Camera * src_camera				= src->camera;
		const unsigned int src_width	= src_camera->width;
		const unsigned int src_height	= src_camera->height;
		const float src_idepth			= src_camera->idepth_scale;
		const float src_cx				= src_camera->cx;
		const float src_cy				= src_camera->cy;
		const float src_fx				= src_camera->fx;
		const float src_fy				= src_camera->fy;
		const float src_ifx				= 1.0/src_camera->fx;
		const float src_ify				= 1.0/src_camera->fy;
		const unsigned int src_width2	= src_camera->width  - 2;
		const unsigned int src_height2	= src_camera->height - 2;



		cv::Mat reliability;
		reliability.create(src_height,src_width,CV_32FC1);
		float   * reliabilityddata   = (float*)(reliability.data);

		for(unsigned long src_h = 0; src_h < src_height;src_h ++){
			for(unsigned long src_w = 0; src_w < src_width;src_w ++){
				unsigned int src_ind = src_h * src_width + src_w;
				reliabilityddata[src_ind] = 0;

				if(src_detdata != 0 && src_detdata[src_ind] != 0){continue;}
				float z         = float(src_depthdata[src_ind]);
				float nx        = src_normalsdata[3*src_ind+0];

				if(z > 0 && nx != 2){
					float x     = (float(src_w) - src_cx) * z * src_ifx;
					float y     = (float(src_h) - src_cy) * z * src_ify;
					float ny    = src_normalsdata[3*src_ind+1];
					float nz    = src_normalsdata[3*src_ind+2];

					double snorm = sqrt(x*x+y*y+z*z);

					double ra = fabs(nx*x/snorm + ny*y/snorm + nz*z/snorm);

					reliabilityddata[src_ind] = ra*ra;
				}
			}
		}
		return reliability;
	}

	virtual void addConstraint(double w, double h, double z, double z2, double weight){
        printf("old addConstraint\n");
	}


    virtual void show(bool stop = false){
        cv::namedWindow(    "Scaling");
        cv::createTrackbar( "Scaling", "z", &z_slider, 10000);

		int width = 640;
		int height = 480;

        cv::Mat img;
        img.create(height,width,CV_8UC3);
        unsigned char * imgdata   = img.data;

        double z = double(z_slider)*0.001;
        double z_max = 0;
        double z_min = 10000;
        std::vector<double> z_ratios;
        z_ratios.resize(width*height);

		for(unsigned long h = 0; h < height; h++){
			for(unsigned long w = 0; w < width; w++){
                unsigned int ind = h * width + w;
				z_ratios[ind] = getRange(double(w)/double(width), double(h)/double(height),z)/z;
                z_max = std::max(z_ratios[ind],z_max);
                z_min = std::min(z_ratios[ind],z_min);
            }
        }

		double z_range = std::max(0.000001,z_max - z_min);

		for(unsigned long h = 0; h < height; h++){
			for(unsigned long w = 0; w < width; w++){
                unsigned int ind = h * width + w;
                double zr = (z_ratios[ind]-z_min)/z_range;
                imgdata[3*ind+0] = 255*zr;
                imgdata[3*ind+1] = 0;
                imgdata[3*ind+2] = 255*(1-zr);
            }
        }

        char buf [1024];
		sprintf(buf,"max: %7.7f min: %7.7f",z_max,z_min);

        cv::putText(img,std::string(buf),cv::Point(50,30), CV_FONT_HERSHEY_SIMPLEX, 1.0,cv::Scalar(0,255,0),2,8,false);

        cv::imshow( "Scaling",	img );
		if(stop){	cv::waitKey(0);}
		else{		cv::waitKey(30);}
    }

	virtual void show(reglib::RGBDFrame * src, reglib::RGBDFrame * dst, Eigen::Matrix4d p){
		printf("show\n");
		viewer->removeAllPointClouds();
		float m00 = p(0,0); float m01 = p(0,1); float m02 = p(0,2); float m03 = p(0,3);
		float m10 = p(1,0); float m11 = p(1,1); float m12 = p(1,2); float m13 = p(1,3);
		float m20 = p(2,0); float m21 = p(2,1); float m22 = p(2,2); float m23 = p(2,3);

		unsigned short * dst_depthdata		= (unsigned short	*)(dst->depth.data);

		reglib::Camera * dst_camera		= dst->camera;
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

		unsigned short * src_depthdata		= (unsigned short	*)(src->depth.data);

		reglib::Camera * src_camera				= src->camera;
		const unsigned int src_width	= src_camera->width;
		const unsigned int src_height	= src_camera->height;
		const float src_idepth			= src_camera->idepth_scale;
		const float src_cx				= src_camera->cx;
		const float src_cy				= src_camera->cy;
		const float src_ifx				= 1.0/src_camera->fx;
		const float src_ify				= 1.0/src_camera->fy;

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr src_before (new pcl::PointCloud<pcl::PointXYZRGB> ());
		for(unsigned long h = 0; h < src_height;h ++){
			for(unsigned long w = 0; w < src_width;w ++){
				float z = src_idepth*float(src_depthdata[h * src_width + w]);
				if(z > 0){
					float x     = (float(w) - src_cx) * z * src_ifx;
					float y     = (float(h) - src_cy) * z * src_ify;

					pcl::PointXYZRGB p;
					p.x	= m00*x + m01*y + m02*z + m03;
					p.y	= m10*x + m11*y + m12*z + m13;
					p.z	= m20*x + m21*y + m22*z + m23;
					p.r = 0;
					p.g = 255;
					p.b = 0;
					src_before->points.push_back(p);
				}
			}
		}

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr dst_before (new pcl::PointCloud<pcl::PointXYZRGB> ());
		for(unsigned long h = 0; h < dst_height;h ++){
			for(unsigned long w = 0; w < dst_width;w ++){
				float z = dst_idepth*float(dst_depthdata[h * dst_width + w]);
				if(z > 0){
					float x     = (float(w) - dst_cx) * z * dst_ifx;
					float y     = (float(h) - dst_cy) * z * dst_ify;

					pcl::PointXYZRGB p;
					p.x	= x;
					p.y	= y;
					p.z	= z;
					p.r = 255;
					p.g = 0;
					p.b = 0;
					dst_before->points.push_back(p);
				}
			}
		}


		pcl::PointCloud<pcl::PointXYZRGB>::Ptr src_after (new pcl::PointCloud<pcl::PointXYZRGB> ());
		for(unsigned long h = 0; h < src_height;h ++){
			for(unsigned long w = 0; w < src_width;w ++){
				float z = src_idepth*float(src_depthdata[h * src_width + w]);
				if(z > 0){
					z = getRange(float(w)/float(src_width),float(h)/float(src_height),z);
					float x     = (float(w) - src_cx) * z * src_ifx;
					float y     = (float(h) - src_cy) * z * src_ify;

					pcl::PointXYZRGB p;
					p.x	= m00*x + m01*y + m02*z + m03;
					p.y	= m10*x + m11*y + m12*z + m13;
					p.z	= m20*x + m21*y + m22*z + m23;
					p.r = 0;
					p.g = 255;
					p.b = 0;
					p.x += 3;
					src_after->points.push_back(p);
				}
			}
		}

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr dst_after (new pcl::PointCloud<pcl::PointXYZRGB> ());
		for(unsigned long h = 0; h < dst_height;h ++){
			for(unsigned long w = 0; w < dst_width;w ++){
				float z = dst_idepth*float(dst_depthdata[h * dst_width + w]);
				if(z > 0){
					z = getRange(float(w)/float(dst_width),float(h)/float(dst_height),z);
					float x     = (float(w) - dst_cx) * z * dst_ifx;
					float y     = (float(h) - dst_cy) * z * dst_ify;

					pcl::PointXYZRGB p;
					p.x	= x;
					p.y	= y;
					p.z	= z;
					p.r = 255;
					p.g = 0;
					p.b = 0;
					p.x += 3;
					dst_after->points.push_back(p);
				}
			}
		}



		viewer->addPointCloud<pcl::PointXYZRGB> (src_before, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(src_before), "src_before");
		viewer->addPointCloud<pcl::PointXYZRGB> (dst_before, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(dst_before), "dst_before");
		viewer->addPointCloud<pcl::PointXYZRGB> (src_after, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(src_after), "src_after");
		viewer->addPointCloud<pcl::PointXYZRGB> (dst_after, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(dst_after), "dst_after");
		viewer->spin();
		viewer->removeAllPointClouds();
	}



	virtual void normalize(){}

    virtual double getRange(double w, double h, double z){return z;}

    virtual void addTrainingData( reglib::RGBDFrame * src, reglib::RGBDFrame * dst, Eigen::Matrix4d p){
		double start1 = reglib::getTime();
		cv::Mat src_reliability = getPixelReliability(src);
		float *  src_rel = (float*)src_reliability.data;
		cv::Mat dst_reliability = getPixelReliability(dst);
		float *  dst_rel = (float*)dst_reliability.data;
		double stop1 = reglib::getTime();

		double start2 = reglib::getTime();

        float m00 = p(0,0); float m01 = p(0,1); float m02 = p(0,2); float m03 = p(0,3);
        float m10 = p(1,0); float m11 = p(1,1); float m12 = p(1,2); float m13 = p(1,3);
        float m20 = p(2,0); float m21 = p(2,1); float m22 = p(2,2); float m23 = p(2,3);

		unsigned short * dst_depthdata		= (unsigned short	*)(dst->depth.data);

		reglib::Camera * dst_camera		= dst->camera;
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

		unsigned short * src_depthdata		= (unsigned short	*)(src->depth.data);

        reglib::Camera * src_camera				= src->camera;
        const unsigned int src_width	= src_camera->width;
        const unsigned int src_height	= src_camera->height;
        const float src_idepth			= src_camera->idepth_scale;
        const float src_cx				= src_camera->cx;
        const float src_cy				= src_camera->cy;
        const float src_ifx				= 1.0/src_camera->fx;
        const float src_ify				= 1.0/src_camera->fy;

        for(unsigned long src_h = 0; src_h < src_height;src_h ++){
            for(unsigned long src_w = 0; src_w < src_width;src_w ++){
                unsigned int src_ind = src_h * src_width + src_w;
				double sr = src_rel[src_ind];

				if(sr == 0){continue;}

				float z = getRange(src_w/float(src_width),src_h/float(src_height),src_idepth*float(src_depthdata[src_ind]));//src_idepth*float(src_depthdata[src_ind]);//getRange(src_w/float(src_width),src_h/float(src_height),src_idepth*float(src_depthdata[src_ind]));

				if(z > 0){
                    float x     = (float(src_w) - src_cx) * z * src_ifx;
                    float y     = (float(src_h) - src_cy) * z * src_ify;

                    float src_tx	= m00*x + m01*y + m02*z + m03;
                    float src_ty	= m10*x + m11*y + m12*z + m13;
                    float src_tz	= m20*x + m21*y + m22*z + m23;

                    float itz	= 1.0/src_tz;
                    float dst_w	= dst_fx*src_tx*itz + dst_cx;
                    float dst_h	= dst_fy*src_ty*itz + dst_cy;

                    if((dst_w > 0) && (dst_h > 0) && (dst_w < dst_width2) && (dst_h < dst_height2)){
                        unsigned int dst_ind = unsigned(dst_h+0.5) * dst_width + unsigned(dst_w+0.5);
						double dr = dst_rel[dst_ind];
						if(dr == 0){continue;}

                        float dst_z = dst_idepth*float(dst_depthdata[dst_ind]);
						if(dst_z > 0){
							double src_cov = z*z*z*z;
//							double src_info = 1.0 / src_info;
							double dst_cov = dst_z*dst_z*dst_z*dst_z;
//							double dst_info = 1.0 /dst_cov;
							double total_cov = src_cov + dst_cov;
//							double total_std = sqrt(total_cov);
//							double scale = 1.0/total_std;
							double dz = dst_z-src_tz;
//							double scaled_rz = scale*dz/0.015;

							double scaled_rz = dz*dz/(0.01*0.01*total_cov);

							double contribution_weight = dst_cov/(src_cov+dst_cov);
							double reliability_weight = dr*sr;
							double overlap_prob = fabs(scaled_rz) < 1.0;//exp(-0.5*scaled_rz*scaled_rz);
							double total_weight = overlap_prob*reliability_weight*contribution_weight;
							if(total_weight > 0.001){
								addConstraint(dst_w/float(dst_width),dst_h/float(dst_height),dst_z,src_tz, total_weight);
							}

                        }
                    }
                }
            }
        }


		double stop2 = reglib::getTime();
		//printf("time: %5.5fs %5.5fs\n",stop1-start1,stop2-start2);

    }

	virtual void print(){}
	virtual void shrink(){}
};

class CameraOptimizerGridXY : public CameraOptimizer{
public:
    int grid_w;
    int grid_h;
    std::vector<double> mul;
    std::vector<double> sum;

	int getInd(int w, int h){return h*grid_w+w;}

    CameraOptimizerGridXY(int gw = 6, int gh = 4, double bias = 100){
        grid_w = gw;
        grid_h = gh;
		int nr_corners = (1+grid_w)*(1+grid_h);
		mul.resize(nr_corners);
		sum.resize(nr_corners);
		for(unsigned long i = 0; i < nr_corners;i++){
            sum[i] = bias;
            mul[i] = sum[i];
        }
        visualizationLvl = 0;
    }

    ~CameraOptimizerGridXY(){}

	virtual void shrink(){
		for(unsigned long i = 0; i < mul.size();i++){
			mul[i] *= 0.5;
			sum[i] *= 0.5;

			if(sum[i] < 100){
				mul[i] = 100.0*mul[i]/sum[i];
				sum[i] = 100.0;
			}
		}
	}

    virtual void addConstraint(double w, double h, double z, double z2, double weight){
        double ratio = z2/z;

		double wg = double(grid_w)*w;
		double hg = double(grid_h)*h;
		//printf("%f %f -> %f %f\n",w,h,wg,hg);
		double dw = wg-std::floor(wg);
		double dh = hg-std::floor(hg);
		int ind00 = getInd(wg  , hg  );//int(wg) + int(hg)*grid_w;
		int ind01 = getInd(wg  , hg+1);//int(wg) + int(hg+1)*grid_w;
		int ind10 = getInd(wg+1, hg  );//int(wg+1) + int(hg)*grid_w;
		int ind11 = getInd(wg+1, hg+1);//int(wg+1) + int(hg+1)*grid_w;
		double mul00 = (1-dw)*(1-dh);
		double mul01 = (1-dw)*(dh);
		double mul10 = (dw)*(1-dh);
		double mul11 = (dw)*(dh);



		sum[ind00] +=       weight*mul00;
		mul[ind00] += ratio*weight*mul00;

		sum[ind01] +=       weight*mul01;
		mul[ind01] += ratio*weight*mul01;

		sum[ind10] +=       weight*mul10;
		mul[ind10] += ratio*weight*mul10;

		sum[ind11] +=       weight*mul11;
		mul[ind11] += ratio*weight*mul11;
    }

    virtual double getRange(double w, double h, double z){

		//return z;

        double wg = double(grid_w)*w;
        double hg = double(grid_h)*h;
		//printf("%f %f -> %f %f\n",w,h,wg,hg);
        double dw = wg-std::floor(wg);
        double dh = hg-std::floor(hg);
		int ind00 = getInd(wg  , hg  );//int(wg) + int(hg)*grid_w;
		int ind01 = getInd(wg  , hg+1);//int(wg) + int(hg+1)*grid_w;
		int ind10 = getInd(wg+1, hg  );//int(wg+1) + int(hg)*grid_w;
		int ind11 = getInd(wg+1, hg+1);//int(wg+1) + int(hg+1)*grid_w;
        double mul00 = (1-dw)*(1-dh);
        double mul01 = (1-dw)*(dh);
        double mul10 = (dw)*(1-dh);
		double mul11 = (dw)*(dh);

		double r00 = mul[ind00]/sum[ind00];
		double r01 = mul[ind01]/sum[ind01];
		double r10 = mul[ind10]/sum[ind10];
		double r11 = mul[ind11]/sum[ind11];
		double ratio = (mul00*r00 + mul01*r01  + mul10*r10 + mul11*r11);
        return z*ratio;
    }


	virtual void print(){
		for(unsigned int w = 0; w < grid_w; w++){
			for(unsigned int h = 0; h < grid_h; h++){
				unsigned int ind = h * grid_w + w;
				printf("%i %i -> %5.5f / %5.5f -> %5.5f\n",w,h,mul[ind],sum[ind],mul[ind]/sum[ind]);
			}
		}
	}

	virtual void normalize(){
		double totsum = 0;
		for(unsigned long i = 0; i < mul.size();i++){
			totsum += mul[i]/sum[i];
		}
		double mean = totsum / double(mul.size());
		for(unsigned long i = 0; i < mul.size();i++){
			mul[i] /= mean;
		}
	}
};



void train_cam(reglib::Model * model){
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("viewer"));
    viewer->setBackgroundColor(1.0,1.0,1.0);
    viewer->removeAllPointClouds();
    viewer->spinOnce();




	CameraOptimizerGridXY * co = new CameraOptimizerGridXY(640/5,480/5);
    co->setVisualization(viewer,1);
	for(int it = 0; it < 3; it++){
		for(unsigned int i = 0; i < model->frames.size(); i++){
			printf("%i / %i\n",i+1,model->frames.size());
			for(unsigned int j = 0; j < model->frames.size(); j++){
				if(i == j){continue;}
				Eigen::Matrix4d m = model->relativeposes[j].inverse() * model->relativeposes[i];
				co->addTrainingData( model->frames[i], model->frames[j], m);//model->relativeposes[i] * model->relativeposes[j].inverse());
				co->normalize();

			}
			co->show(false);
			co->normalize();
		}
		co->print();
		co->show(true);
		co->shrink();
	}

	for(unsigned int i = 0; i < model->frames.size(); i++){
		for(unsigned int j = 0; j < model->frames.size(); j++){
			if(i == j){continue;}
			Eigen::Matrix4d m = model->relativeposes[j].inverse() * model->relativeposes[i];
			co->show( model->frames[i], model->frames[j], m);//model->relativeposes[i] * model->relativeposes[j].inverse());
		}
	}
	exit(0);
}

int main(int argc, char **argv){
    ModelStorageFile * storage = new ModelStorageFile("fb_model/");
    std::vector<std::string> str = storage->loadAllModels();
    for(unsigned int i = 0; i < str.size(); i++){
        train_cam(storage->fetch(str[i]));
    }
}
