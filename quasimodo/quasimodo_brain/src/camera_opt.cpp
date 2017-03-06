#include "ModelStorage/ModelStorage.h"
#include "Util/Util.h"

class CameraOptimizer{
public:
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    int visualizationLvl;

    CameraOptimizer(){
        visualizationLvl = 0;
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

	}
    virtual double getRange(double w, double h, double z){return z;}

    virtual void addTrainingData( reglib::RGBDFrame * src, reglib::RGBDFrame * dst, Eigen::Matrix4d p){

		cv::Mat src_reliability = getPixelReliability(src);
		float *  src_rel = (float*)src_reliability.data;
		cv::Mat dst_reliability = getPixelReliability(dst);
		float *  dst_rel = (float*)dst_reliability.data;

        float m00 = p(0,0); float m01 = p(0,1); float m02 = p(0,2); float m03 = p(0,3);
        float m10 = p(1,0); float m11 = p(1,1); float m12 = p(1,2); float m13 = p(1,3);
        float m20 = p(2,0); float m21 = p(2,1); float m22 = p(2,2); float m23 = p(2,3);

        unsigned char  * dst_detdata		= (unsigned char	*)(dst->det_dilate.data);
        unsigned short * dst_depthdata		= (unsigned short	*)(dst->depth.data);
        float		   * dst_normalsdata	= (float			*)(dst->normals.data);

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

        cv::Mat rz;
        rz.create(src_height,src_width,CV_8UC3);
        unsigned char   * rzdata   = rz.data;

        for(unsigned long src_h = 0; src_h < src_height;src_h ++){
            for(unsigned long src_w = 0; src_w < src_width;src_w ++){
                unsigned int src_ind = src_h * src_width + src_w;

                rzdata[3*src_ind+0] = 0;
                rzdata[3*src_ind+1] = 0;
				rzdata[3*src_ind+2] = 0;
				double sr = src_rel[src_ind];

				if(sr == 0){continue;}
				float z         = getRange(src_w,src_h,src_idepth*float(src_depthdata[src_ind]));

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
							double src_info = 1.0 / src_info;
							double dst_cov = dst_z*dst_z*dst_z*dst_z;
							double dst_info = 1.0 /dst_cov;
							double total_cov = src_cov + dst_cov;
							double total_std = sqrt(total_cov);
							double scale = 1.0/total_std;
                            double dz = dst_z-src_tz;
							double scaled_rz = scale*dz/0.015;

							double contribution_weight = 1.0-(dst_info/(src_info+dst_info));
							double reliability_weight = dr*sr;
							double overlap_prob = exp(-0.5*scaled_rz*scaled_rz);
							double total_weight = overlap_prob*reliability_weight*contribution_weight;

							addConstraint(dst_w,dst_h,dst_z,src_tz, total_weight);


//							rzdata[3*src_ind+0] = 255.0 * exp(-0.5*scaled_rz*scaled_rz);
//							rzdata[3*src_ind+1] = rzdata[3*src_ind+0];
//							rzdata[3*src_ind+2] = rzdata[3*src_ind+0];

							if(scaled_rz >= 1){
								rzdata[3*src_ind+0] = 0;
								rzdata[3*src_ind+1] = 0;
								rzdata[3*src_ind+2] = 0;
							}if(scaled_rz <= -1){
								rzdata[3*src_ind+0] = 0;
								rzdata[3*src_ind+1] = 0;
								rzdata[3*src_ind+2] = 0;
							}else{

								rzdata[3*src_ind+0] = 0;
								rzdata[3*src_ind+1] = 255.0 * std::min(1.0,std::max(0.0,(0.5 + 0.25*scaled_rz)))	 * reliability_weight * contribution_weight;
								rzdata[3*src_ind+2] = 255.0 * std::min(1.0,std::max(0.0,(0.5 + 0.25*(1-scaled_rz)))) * reliability_weight * contribution_weight;
							}

                        }
                    }
                }
            }
        }





		cv::namedWindow( "rz"	, cv::WINDOW_AUTOSIZE );
		cv::imshow( "rz",	rz );

		cv::namedWindow( "src_reliability"	, cv::WINDOW_AUTOSIZE );
		cv::imshow( "src_reliability",	src_reliability );

		cv::namedWindow( "dst_reliability"	, cv::WINDOW_AUTOSIZE );
		cv::imshow( "dst_reliability",	dst_reliability );

        cv::waitKey(0);

//        if(visualizationLvl == 1){
//            viewer->removeAllPointClouds();
//            pcl::PointCloud<pcl::PointXYZRGB>::Ptr src_cld = src->getSmallPCLcloud();
//            pcl::PointCloud<pcl::PointXYZRGB>::Ptr dst_cld = dst->getSmallPCLcloud();

//            for(unsigned int i = 0; i < src_cld->points.size(); i++){
//                src_cld->points[i].r = 255;
//                src_cld->points[i].g = 0;
//                src_cld->points[i].b = 0;
//            }

//            for(unsigned int i = 0; i < dst_cld->points.size(); i++){
//                dst_cld->points[i].r = 0;
//                dst_cld->points[i].g = 255;
//                dst_cld->points[i].b = 0;
//            }

//            pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZRGB> ());
//            pcl::transformPointCloud (*src_cld, *transformed_cloud, p);

//            viewer->addPointCloud<pcl::PointXYZRGB> (transformed_cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(transformed_cloud), "src_cloud");
//            viewer->addPointCloud<pcl::PointXYZRGB> (dst_cld, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(dst_cld), "dst_cloud");
//            viewer->spin();
//        }

    }



};



void train_cam(reglib::Model * model){
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("viewer"));
    viewer->setBackgroundColor(1.0,1.0,1.0);
    viewer->removeAllPointClouds();
    viewer->spinOnce();




    CameraOptimizer * co = new CameraOptimizer();
    co->setVisualization(viewer,1);
    for(unsigned int i = 0; i < model->frames.size(); i++){
        for(unsigned int j = 0; j < model->frames.size(); j++){
            if(i == j){continue;}
            Eigen::Matrix4d m = model->relativeposes[j].inverse() * model->relativeposes[i];
            co->addTrainingData( model->frames[i], model->frames[j], m);//model->relativeposes[i] * model->relativeposes[j].inverse());
        }
    }
}

int main(int argc, char **argv){
    ModelStorageFile * storage = new ModelStorageFile("fb_model/");
    std::vector<std::string> str = storage->loadAllModels();
    for(unsigned int i = 0; i < str.size(); i++){
        train_cam(storage->fetch(str[i]));
    }
}
