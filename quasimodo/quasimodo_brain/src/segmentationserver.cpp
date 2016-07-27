#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sensor_msgs/PointCloud2.h>
#include <string.h>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include "eigen_conversions/eigen_msg.h"
#include "tf_conversions/tf_eigen.h"

#include "quasimodo_msgs/segment_model.h"


#include <image_geometry/pinhole_camera_model.h>
#include <sensor_msgs/CameraInfo.h>

#include "modelupdater/ModelUpdater.h"
#include "core/RGBDFrame.h"

#include "Util/Util.h"


using namespace std;


bool visualization = false;
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;


//Strategy:
//Register frames to background+other frames in the same model DONE
//Determine occlusions inside current model to filter out motion(most likeley people etc) //Done
    //Remove from current model
//Determine occlusion of current model to background
    //These pixels are dynamic objects
//Update background
    //How?

bool segment_model(quasimodo_msgs::segment_model::Request  & req, quasimodo_msgs::segment_model::Response & res){
	printf("segment_model\n");
	std::vector< reglib::Model * > models;
	for(unsigned int i = 0; i < req.models.size(); i++){
		models.push_back(quasimodo_brain::getModelFromMSG(req.models[i]));
	}

	reglib::RegistrationRandom *	reg	= new reglib::RegistrationRandom();
	reglib::ModelUpdaterBasicFuse * mu	= new reglib::ModelUpdaterBasicFuse( models.front(), reg);
	mu->occlusion_penalty               = 15;
	mu->massreg_timeout                 = 60*4;
	mu->viewer							= viewer;

	reglib::Model * bg = quasimodo_brain::getModelFromMSG(req.backgroundmodel);

	reglib::MassRegistrationPPR2 * bgmassreg = new reglib::MassRegistrationPPR2(0.0);
	bgmassreg->timeout = 1200;
	bgmassreg->viewer = viewer;
	bgmassreg->visualizationLvl = 0;
	bgmassreg->maskstep = 10;//std::max(1,int(0.4*double(models[i]->frames.size())));
	bgmassreg->nomaskstep = 10;//std::max(3,int(0.5+0.*double(models[i]->frames.size())));//std::max(1,int(0.5+1.0*double(model->frames.size())));
	bgmassreg->nomask = true;
	bgmassreg->stopval = 0.0005;
	bgmassreg->setData(bg->frames,bg->modelmasks);
	reglib::MassFusionResults bgmfr = bgmassreg->getTransforms(bg->relativeposes);
	bg->relativeposes = bgmfr.poses;


	bg->points = mu->getSuperPoints(bg->relativeposes,bg->frames,bg->modelmasks,1,false);
//	printf("frames: %i\n",bg->frames.size());
//	for(int i = 0; i < bg->frames.size(); i++){
//		cv::imshow("modelmask", bg->modelmasks[i]->getMask());
//		bg->frames[i]->show(true);
//	}
//	printf("points: %i\n",bg->points.size());

//	viewer->removeAllPointClouds();
//	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = bg->getPCLcloud(1, false);
//	viewer->addPointCloud<pcl::PointXYZRGB> (cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(cloud), "cloud");
//	viewer->spin();
//	printf("cloudsize: %i\n",cloud->points.size());

	vector<Eigen::Matrix4d> cp;
	vector<reglib::RGBDFrame*> cf;
	vector<reglib::ModelMask*> mm;

	vector<Eigen::Matrix4d> cp_front;
	vector<reglib::RGBDFrame*> cf_front;
	vector<reglib::ModelMask*> mm_front;


	if(bg->frames.size() > 0){
		cp_front.push_back(Eigen::Matrix4d::Identity());
		cf_front.push_back(bg->frames.front());
		mm_front.push_back(bg->modelmasks.front());

		for(int j = 0; j < models.size(); j++){
			cp_front.push_back(bg->relativeposes.front().inverse() * models[j]->relativeposes.front());
			cf_front.push_back(models[j]->frames.front());
			mm_front.push_back(models[j]->modelmasks.front());
		}
	}else{
		for(int j = 0; j < models.size(); j++){
			cp_front.push_back(models.front()->relativeposes.front().inverse() * models[j]->relativeposes.front());
			cf_front.push_back(models[j]->frames.front());
			mm_front.push_back(models[j]->modelmasks.front());
		}
	}

	printf("cp_front: %i\n",cp_front.size());

    if(models.size() > 1 || (models.size() > 0 && bg->frames.size() > 0)){
        printf("%s::%i\n",__FILE__,__LINE__);
		reglib::MassRegistrationPPR2 * massreg = new reglib::MassRegistrationPPR2(0.05);
		massreg->timeout = 1200;
		massreg->viewer = viewer;
        massreg->visualizationLvl = 0;

		massreg->maskstep = 5;//std::max(1,int(0.4*double(models[i]->frames.size())));
		massreg->nomaskstep = 5;//std::max(3,int(0.5+0.*double(models[i]->frames.size())));//std::max(1,int(0.5+1.0*double(model->frames.size())));
		massreg->nomask = true;
		massreg->stopval = 0.0005;

		massreg->setData(cf_front,mm_front);


		reglib::MassFusionResults mfr_front = massreg->getTransforms(cp_front);
        for(int j = 0; j < mfr_front.poses.size(); j++){
            std::cout << mfr_front.poses[j] * mfr_front.poses.front().inverse() << std::endl << std::endl << std::endl;
        }

        if(bg->frames.size() > 0){
            printf("change:\n");
            for(int j = models.size()-1; j >= 0; j--){
                Eigen::Matrix4d change = mfr_front.poses[j+1] * cp_front[j].inverse();
                std::cout << change << std::endl << std::endl << std::endl;
                for(unsigned int k = 0; k < models[j]->relativeposes.size(); k++){
                    cp.push_back(change * models.front()->relativeposes.front().inverse() * models[j]->relativeposes[k]);
                    cf.push_back(models[j]->frames[k]);
                    mm.push_back(models[j]->modelmasks[k]);
                }
            }
        }else{
            printf("change:\n");
            for(int j = models.size()-1; j >= 0; j--){
                Eigen::Matrix4d change = mfr_front.poses[j] * cp_front[j].inverse();
                std::cout << change << std::endl << std::endl << std::endl;
                for(unsigned int k = 0; k < models[j]->relativeposes.size(); k++){
                    cp.push_back(change * models.front()->relativeposes.front().inverse() * models[j]->relativeposes[k]);
                    cf.push_back(models[j]->frames[k]);
                    mm.push_back(models[j]->modelmasks[k]);
                }
            }
        }
	}else{
        printf("%s::%i\n",__FILE__,__LINE__);
		for(int j = models.size()-1; j >= 0; j--){
			Eigen::Matrix4d change = Eigen::Matrix4d::Identity();//mfr_front.poses[j] * cp_front[j].inverse();
			for(unsigned int k = 0; k < models[j]->relativeposes.size(); k++){
				cp.push_back(change * models.front()->relativeposes.front().inverse() * models[j]->relativeposes[k]);
				cf.push_back(models[j]->frames[k]);
				mm.push_back(models[j]->modelmasks[k]);
			}
		}
	}

	printf("cp: %i\n",cp.size());

    reglib::MassRegistrationPPR2 * massreg2 = new reglib::MassRegistrationPPR2(0.05);
	massreg2->timeout = 1200;
	massreg2->viewer = viewer;
    massreg2->visualizationLvl = 1;

	massreg2->maskstep = 10;//std::max(1,int(0.4*double(models[i]->frames.size())));
	massreg2->nomaskstep = 10;//std::max(3,int(0.5+0.*double(models[i]->frames.size())));//std::max(1,int(0.5+1.0*double(model->frames.size())));
	massreg2->nomask = true;
	massreg2->stopval = 0.0005;

	massreg2->setData(cf,mm);

	if(bg->frames.size() > 0){
		massreg2->addModel(bg);
		cp.push_back(Eigen::Matrix4d::Identity());
		printf("---->added background model\n");
	}

	reglib::MassFusionResults mfr2 = massreg2->getTransforms(cp);
	cp = mfr2.poses;

	Eigen::Matrix4d relative_to_bg = Eigen::Matrix4d::Identity();
	if(bg->frames.size() > 0){
		relative_to_bg = cp.front().inverse()*cp.back();
		cp.pop_back();
	}

    vector<cv::Mat> masks;
    for(unsigned int i = 0; i < cf.size(); i++){
        cv::Mat mask;
        mask.create(cf[i]->camera->height,cf[i]->camera->width,CV_8UC1);
        printf("%i -> %i %i\n",i,cf[i]->camera->height,cf[i]->camera->width);
        unsigned char * maskdata = (unsigned char *)(mask.data);
        for(unsigned int j = 0; j < cf[i]->camera->height*cf[i]->camera->width;j++){
            maskdata[j] = 255;
        }
        masks.push_back(mask);
    }

    vector<Eigen::Matrix4d> bgcp;
    vector<reglib::RGBDFrame*> bgcf;
    vector<cv::Mat> bgmask;

    //std::vector<cv::Mat> internal_masks = mu->computeDynamicObject(bgcp,bgcf,bgmask,cp,cf,masks,cp,cf,masks);//Determine self occlusions

    for(unsigned int k = 0; k < bg->relativeposes.size(); k++){
        bgcp.push_back(bg->relativeposes[k]);
        bgcf.push_back(bg->frames[k]);
        bgmask.push_back(bg->modelmasks[k]->getMask());
    }
    std::vector<cv::Mat> external_masks = mu->computeDynamicObject(bgcp,bgcf,bgmask,bgcp,bgcf,bgmask,cp,cf,masks);//Determine self occlusions
/*
    for(unsigned int i = 0; i < cf.size(); i++){
        cv::Mat mask;
        mask.create(cf[i]->camera->height,cf[i]->camera->width,CV_8UC1);
        unsigned char * maskdata = (unsigned char *)(mask.data);
        unsigned char * internalmaskdata = (unsigned char *)(internal_masks[i].data);
        unsigned char * externalmaskdata = (unsigned char *)(external_masks[i].data);
        for(unsigned int j = 0; j < cf[i]->camera->height*cf[i]->camera->width;j++){
            if(externalmaskdata[j] > 0 && internalmaskdata[j] == 0 ){maskdata[j] = 255;}
            else{
                maskdata[j] = 0;
            }
            //maskdata[i] = std::max(internalmaskdata[i],dynamicmaskdata[i]);
        }


        cv::imshow( "rgb", cf[i]->rgb );
        cv::imshow( "externalmask", internal_masks[i] );
        cv::imshow( "externalmask", external_masks[i] );
        cv::imshow( "mask", mask );
        cv::waitKey(0);
//        bg->frames.push_back(cf[i]);
//        bg->relativeposes.push_back(cp[i]);
//        bg->modelmasks.push_back(new reglib::ModelMask(mask));
    }
*/
/*
    std::vector<cv::Mat> internal_masks = mu->computeDynamicObject(bg,relative_to_bg,cp,cf,masks);//Determine self occlusions
   // std::vector<cv::Mat> dynamic_masks = mu->computeDynamicObject(bg,relative_to_bg,cp2,cf2,internal_masks);//Determine occlusion of background occlusions
    for(unsigned int i = 0; i < cf.size(); i++){

        cv::Mat mask;
        mask.create(cf[i]->camera->height,cf[i]->camera->width,CV_8UC1);
        unsigned char * maskdata = (unsigned char *)(mask.data);
        unsigned char * internalmaskdata = (unsigned char *)(internal_masks[i].data);
        //unsigned char * dynamicmaskdata = (unsigned char *)(dynamic_masks[i].data);
        for(unsigned int i = 0; i < cf[i]->camera->height*cf[i]->camera->width;i++){
            //maskdata[i] = std::max(internalmaskdata[i],dynamicmaskdata[i]);
        }

        bg->frames.push_back(cf[i]);
        bg->relativeposes.push_back(cp[i]);
        bg->modelmasks.push_back(new reglib::ModelMask(mask));
    }
*/

	return true;
}

int main(int argc, char** argv){
	ros::init(argc, argv, "segmentationserver");
	ros::NodeHandle n;

	int inputstate = -1;
	for(int i = 1; i < argc;i++){
		printf("input: %s\n",argv[i]);
		if(std::string(argv[i]).compare("-v") == 0){           printf("visualization turned on\n");                visualization = true;}
	}

	if(visualization){
		viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("viewer"));
		viewer->addCoordinateSystem(0.01);
		viewer->setBackgroundColor(0.0,0.0,0.0);
	}


	ros::ServiceServer service = n.advertiseService("segment_model", segment_model);
	ROS_INFO("Ready to add use segment_model.");

	ros::spin();

/*
exit(0);
	viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("3D Viewer"));
	viewer->setBackgroundColor (0.5, 0, 0.5);
	viewer->addCoordinateSystem (1.0);
	viewer->initCameraParameters ();

	ros::NodeHandle pn("~");
	for(int ar = 1; ar < argc; ar++){
		string overall_folder = std::string(argv[ar]);

		vector<string> sweep_xmls = semantic_map_load_utilties::getSweepXmls<PointType>(overall_folder);
		printf("sweep_xmls\n");
		for (auto sweep_xml : sweep_xmls) {
			printf("sweep_xml: %s\n",sweep_xml.c_str());
			load2(sweep_xml);
		}
	}

	for(unsigned int i = 0; i < models.size(); i++){
		printf("%i -> %i\n",i,models[i]->frames.size());

		vector<Eigen::Matrix4d> cp;
		vector<reglib::RGBDFrame*> cf;
		vector<reglib::ModelMask*> mm;

		vector<Eigen::Matrix4d> cp_front;
		vector<reglib::RGBDFrame*> cf_front;
		vector<reglib::ModelMask*> mm_front;
		for(int j = 0; j <= i; j++){
			cp_front.push_back(models.front()->relativeposes.front().inverse() * models[j]->relativeposes.front());
			cf_front.push_back(models[j]->frames.front());
			mm_front.push_back(models[j]->modelmasks.front());
		}

		if(i > 0){
			reglib::MassRegistrationPPR2 * massreg = new reglib::MassRegistrationPPR2(0.05);
			massreg->timeout = 1200;
			massreg->viewer = viewer;
			massreg->visualizationLvl = 0;

			massreg->maskstep = 5;//std::max(1,int(0.4*double(models[i]->frames.size())));
			massreg->nomaskstep = 5;//std::max(3,int(0.5+0.*double(models[i]->frames.size())));//std::max(1,int(0.5+1.0*double(model->frames.size())));
			massreg->nomask = true;
			massreg->stopval = 0.0005;

			massreg->setData(cf_front,mm_front);


			reglib::MassFusionResults mfr_front = massreg->getTransforms(cp_front);

			for(int j = i; j >= 0; j--){
				Eigen::Matrix4d change = mfr_front.poses[j] * cp_front[j].inverse();
				for(unsigned int k = 0; k < models[j]->relativeposes.size(); k++){
					cp.push_back(change * models.front()->relativeposes.front().inverse() * models[j]->relativeposes[k]);
					cf.push_back(models[j]->frames[k]);
					mm.push_back(models[j]->modelmasks[k]);
				}
			}
		}else{
			for(int j = i; j >= 0; j--){
				Eigen::Matrix4d change = Eigen::Matrix4d::Identity();//mfr_front.poses[j] * cp_front[j].inverse();
				for(unsigned int k = 0; k < models[j]->relativeposes.size(); k++){
					cp.push_back(change * models.front()->relativeposes.front().inverse() * models[j]->relativeposes[k]);
					cf.push_back(models[j]->frames[k]);
					mm.push_back(models[j]->modelmasks[k]);
				}
			}
		}

		reglib::MassRegistrationPPR2 * massreg2 = new reglib::MassRegistrationPPR2(0.0);
		massreg2->timeout = 1200;
		massreg2->viewer = viewer;
		massreg2->visualizationLvl = 1;

		massreg2->maskstep = 10;//std::max(1,int(0.4*double(models[i]->frames.size())));
		massreg2->nomaskstep = 10;//std::max(3,int(0.5+0.*double(models[i]->frames.size())));//std::max(1,int(0.5+1.0*double(model->frames.size())));
		massreg2->nomask = true;
		massreg2->stopval = 0.0005;

		massreg2->setData(cf,mm);
		reglib::MassFusionResults mfr2 = massreg2->getTransforms(cp);
		cp = mfr2.poses;

		reglib::RegistrationRandom *	reg	= new reglib::RegistrationRandom();
		reglib::ModelUpdaterBasicFuse * mu	= new reglib::ModelUpdaterBasicFuse( models[i], reg);
		mu->occlusion_penalty               = 15;
		mu->massreg_timeout                 = 60*4;
		mu->viewer							= viewer;

		vector<cv::Mat> mats;
		mu->computeDynamicObject(cp,cf,mats);

		//delete mu;
	}


	*/
}
