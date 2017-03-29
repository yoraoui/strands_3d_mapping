/*
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sensor_msgs/PointCloud2.h>
#include <string.h>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include "eigen_conversions/eigen_msg.h"
#include "tf_conversions/tf_eigen.h"

#include "metaroom_xml_parser/simple_xml_parser.h"
#include "metaroom_xml_parser/simple_summary_parser.h"

#include <tf_conversions/tf_eigen.h>

#include "ros/ros.h"
#include <metaroom_xml_parser/load_utilities.h>
#include <pcl_ros/point_cloud.h>
#include <cv_bridge/cv_bridge.h>

#include "metaroom_xml_parser/load_utilities.h"
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_ros/transforms.h>

#include <tf_conversions/tf_eigen.h>

#include "quasimodo_msgs/model.h"
#include "quasimodo_msgs/rgbd_frame.h"
#include "quasimodo_msgs/model_from_frame.h"
#include "quasimodo_msgs/index_frame.h"
#include "quasimodo_msgs/fuse_models.h"
#include "quasimodo_msgs/get_model.h"
#include "quasimodo_msgs/segment_model.h"
#include "quasimodo_msgs/metaroom_pair.h"

#include "ros/ros.h"
#include <quasimodo_msgs/query_cloud.h>
#include <quasimodo_msgs/visualize_query.h>
#include <metaroom_xml_parser/load_utilities.h>
#include <pcl_ros/point_cloud.h>
#include <cv_bridge/cv_bridge.h>

#include "metaroom_xml_parser/load_utilities.h"
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_ros/transforms.h>

#include "quasimodo_msgs/model.h"
#include "quasimodo_msgs/rgbd_frame.h"
#include "quasimodo_msgs/model_from_frame.h"
#include "quasimodo_msgs/index_frame.h"
#include "quasimodo_msgs/fuse_models.h"
#include "quasimodo_msgs/get_model.h"

#include "metaroom_xml_parser/simple_xml_parser.h"
#include "metaroom_xml_parser/load_utilities.h"

#include <image_geometry/pinhole_camera_model.h>
#include <sensor_msgs/CameraInfo.h>

#include "modelupdater/ModelUpdater.h"
#include "core/RGBDFrame.h"
#include "Util/Util.h"

// Services
#include <object_manager_msgs/DynamicObjectsService.h>
#include <object_manager_msgs/GetDynamicObjectService.h>
#include <object_manager_msgs/ProcessDynamicObjectService.h>

// Registration service
#include <observation_registration_services/ObjectAdditionalViewRegistrationService.h>

// Additional view mask service
#include <object_manager_msgs/DynamicObjectComputeMaskService.h>


#include <sys/time.h>
#include <sys/resource.h>

// Custom messages
#include "object_manager/dynamic_object.h"
#include "object_manager/dynamic_object_xml_parser.h"
#include "object_manager/dynamic_object_utilities.h"
#include "object_manager/dynamic_object_mongodb_interface.h"

#include <object_manager_msgs/DynamicObjectTracks.h>
#include <object_manager_msgs/DynamicObjectTrackingData.h>


#include <semantic_map_msgs/RoomObservation.h>

#include <iostream>
#include <fstream>


std::vector< ros::ServiceServer > m_DynamicObjectsServiceServers;
std::vector< ros::ServiceServer > m_GetDynamicObjectServiceServers;

typedef typename object_manager_msgs::DynamicObjectsService::Request DynamicObjectsServiceRequest;
typedef typename object_manager_msgs::DynamicObjectsService::Response DynamicObjectsServiceResponse;

typedef typename object_manager_msgs::GetDynamicObjectService::Request GetDynamicObjectServiceRequest;
typedef typename object_manager_msgs::GetDynamicObjectService::Response GetDynamicObjectServiceResponse;

typedef typename object_manager_msgs::ProcessDynamicObjectService::Request ProcessDynamicObjectServiceRequest;
typedef typename object_manager_msgs::ProcessDynamicObjectService::Response ProcessDynamicObjectServiceResponse;

ros::ServiceClient segmentation_client;
using namespace std;
using namespace semantic_map_load_utilties;


typedef pcl::PointCloud<PointType> Cloud;
typedef typename Cloud::Ptr CloudPtr;
typedef pcl::search::KdTree<PointType> Tree;
typedef semantic_map_load_utilties::DynamicObjectData<PointType> ObjectData;

using namespace std;
using namespace semantic_map_load_utilties;

boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
int visualization_lvl		= 0;

Eigen::Matrix4d getPose(QXmlStreamReader * xmlReader){
	QXmlStreamReader::TokenType token = xmlReader->readNext();//Translation
	QString elementName = xmlReader->name().toString();

	token = xmlReader->readNext();//fx
	double tx = atof(xmlReader->readElementText().toStdString().c_str());

	token = xmlReader->readNext();//fy
	double ty = atof(xmlReader->readElementText().toStdString().c_str());

	token = xmlReader->readNext();//cx
	double tz = atof(xmlReader->readElementText().toStdString().c_str());

	token = xmlReader->readNext();//Translation
	elementName = xmlReader->name().toString();

	token = xmlReader->readNext();//Rotation
	elementName = xmlReader->name().toString();

	token = xmlReader->readNext();//qw
	double qw = atof(xmlReader->readElementText().toStdString().c_str());

	token = xmlReader->readNext();//qx
	double qx = atof(xmlReader->readElementText().toStdString().c_str());

	token = xmlReader->readNext();//qy
	double qy = atof(xmlReader->readElementText().toStdString().c_str());

	token = xmlReader->readNext();//qz
	double qz = atof(xmlReader->readElementText().toStdString().c_str());

	token = xmlReader->readNext();//Rotation
	elementName = xmlReader->name().toString();

	Eigen::Matrix4d regpose = (Eigen::Affine3d(Eigen::Quaterniond(qw,qx,qy,qz))).matrix();
	regpose(0,3) = tx;
	regpose(1,3) = ty;
	regpose(2,3) = tz;

	return regpose;
}

void readViewXML(std::string xmlFile, std::vector<reglib::RGBDFrame *> & frames, std::vector<Eigen::Matrix4d> & poses){
	QFile file(xmlFile.c_str());

	if (!file.exists())
	{
		ROS_ERROR("Could not open file %s to load room.",xmlFile.c_str());
		return;
	}

	QString xmlFileQS(xmlFile.c_str());
	int index = xmlFileQS.lastIndexOf('/');
	std::string roomFolder = xmlFileQS.left(index).toStdString();


	file.open(QIODevice::ReadOnly);
	ROS_INFO_STREAM("Parsing xml file: "<<xmlFile.c_str());

	QXmlStreamReader* xmlReader = new QXmlStreamReader(&file);


	while (!xmlReader->atEnd() && !xmlReader->hasError())
	{
		QXmlStreamReader::TokenType token = xmlReader->readNext();
		if (token == QXmlStreamReader::StartDocument)
			continue;

		if (xmlReader->hasError())
		{
			ROS_ERROR("XML error: %s in %s",xmlReader->errorString().toStdString().c_str(),xmlFile.c_str());
			return;
		}

		QString elementName = xmlReader->name().toString();

		if (token == QXmlStreamReader::StartElement)
		{

			if (xmlReader->name() == "View")
			{
				cv::Mat rgb;
				cv::Mat depth;
				//printf("elementName: %s\n",elementName.toStdString().c_str());
				QXmlStreamAttributes attributes = xmlReader->attributes();
				if (attributes.hasAttribute("RGB"))
				{
					std::string imgpath = attributes.value("RGB").toString().toStdString();
					printf("rgb filename: %s\n",(roomFolder+"/"+imgpath).c_str());
					rgb = cv::imread(roomFolder+"/"+imgpath, CV_LOAD_IMAGE_UNCHANGED);

					//QString rgbpath = attributes.value("RGB").toString();
					//rgb = cv::imread(roomFolder+"/"+(rgbpath.toStdString().c_str()), CV_LOAD_IMAGE_UNCHANGED);
					//rgb = cv::imread((rgbpath.toStdString()).c_str(), CV_LOAD_IMAGE_UNCHANGED);
				}else{break;}


				if (attributes.hasAttribute("DEPTH"))
				{
					std::string imgpath = attributes.value("DEPTH").toString().toStdString();
					printf("depth filename: %s\n",(roomFolder+"/"+imgpath).c_str());
					depth = cv::imread(roomFolder+"/"+imgpath, CV_LOAD_IMAGE_UNCHANGED);
					//QString depthpath = attributes.value("DEPTH").toString();
					//printf("depth filename: %s\n",depthpath.toStdString().c_str());
					//depth = cv::imread((roomFolder+"/"+depthpath.toStdString()).c_str(), CV_LOAD_IMAGE_UNCHANGED);
					//depth = cv::imread(roomFolder+"/"+(depthpath.toStdString().c_str()), CV_LOAD_IMAGE_UNCHANGED);
				}else{break;}


				token = xmlReader->readNext();//Stamp
				elementName = xmlReader->name().toString();

				token = xmlReader->readNext();//sec
				elementName = xmlReader->name().toString();
				int sec = atoi(xmlReader->readElementText().toStdString().c_str());

				token = xmlReader->readNext();//nsec
				elementName = xmlReader->name().toString();
				int nsec = atoi(xmlReader->readElementText().toStdString().c_str());
				token = xmlReader->readNext();//end stamp

				token = xmlReader->readNext();//Camera
				elementName = xmlReader->name().toString();

				reglib::Camera * cam = new reglib::Camera();

				token = xmlReader->readNext();//fx
				cam->fx = atof(xmlReader->readElementText().toStdString().c_str());

				token = xmlReader->readNext();//fy
				cam->fy = atof(xmlReader->readElementText().toStdString().c_str());

				token = xmlReader->readNext();//cx
				cam->cx = atof(xmlReader->readElementText().toStdString().c_str());

				token = xmlReader->readNext();//cy
				cam->cy = atof(xmlReader->readElementText().toStdString().c_str());

				token = xmlReader->readNext();//Camera
				elementName = xmlReader->name().toString();

				double time = double(sec)+double(nsec)/double(1e9);

				token = xmlReader->readNext();//RegisteredPose
				elementName = xmlReader->name().toString();

				Eigen::Matrix4d regpose = getPose(xmlReader);

				token = xmlReader->readNext();//RegisteredPose
				elementName = xmlReader->name().toString();


				token = xmlReader->readNext();//Pose
				elementName = xmlReader->name().toString();

				Eigen::Matrix4d pose = getPose(xmlReader);

				token = xmlReader->readNext();//Pose
				elementName = xmlReader->name().toString();

				reglib::RGBDFrame * frame = new reglib::RGBDFrame(cam,rgb,depth, time, pose);
				frames.push_back(frame);
				poses.push_back(pose);
			}
		}
	}
	delete xmlReader;
}

std::vector<Eigen::Matrix4d> readPoseXML(std::string xmlFile){
	std::vector<Eigen::Matrix4d> poses;
	QFile file(xmlFile.c_str());

	if (!file.exists()){
		ROS_ERROR("Could not open file %s to load poses.",xmlFile.c_str());
		return poses;
	}

	file.open(QIODevice::ReadOnly);
	ROS_INFO_STREAM("Parsing xml file: "<<xmlFile.c_str());

	QXmlStreamReader* xmlReader = new QXmlStreamReader(&file);

	while (!xmlReader->atEnd() && !xmlReader->hasError()){
		QXmlStreamReader::TokenType token = xmlReader->readNext();
		if (token == QXmlStreamReader::StartDocument)
			continue;

		if (xmlReader->hasError())
		{
			ROS_ERROR("XML error: %s in %s",xmlReader->errorString().toStdString().c_str(),xmlFile.c_str());
			return poses;
		}

		QString elementName = xmlReader->name().toString();

		if (token == QXmlStreamReader::StartElement){
			if (xmlReader->name() == "Pose"){
				Eigen::Matrix4d pose = getPose(xmlReader);

				token = xmlReader->readNext();//Pose
				elementName = xmlReader->name().toString();

				std::cout << pose << std::endl << std::endl;
				poses.push_back(pose);
			}
		}
	}
	delete xmlReader;
	printf("done readPoseXML\n");
	return poses;
}

std::vector<reglib::Model *> loadModels(std::string path){
	printf("loadModels: %s\n",path.c_str());
	std::vector<reglib::Model *> models;
	int slash_pos = path.find_last_of("/");
	std::string sweep_folder = path.substr(0, slash_pos) + "/";

	QStringList objectFiles = QDir(sweep_folder.c_str()).entryList(QStringList("dynamic_obj*.xml"));
	if(objectFiles.size() == 0){return models;}

	std::vector<reglib::RGBDFrame *> frames;
	std::vector<Eigen::Matrix4d> poses;
	readViewXML(sweep_folder+"ViewGroup.xml",frames,poses);


	for (auto objectFile : objectFiles){
		std::string object = sweep_folder+objectFile.toStdString();
		printf("object: %s\n",object.c_str());

		QFile file(object.c_str());

		if (!file.exists()){
			ROS_ERROR("Could not open file %s to masks.",object.c_str());
			continue;
		}

		file.open(QIODevice::ReadOnly);
		ROS_INFO_STREAM("Parsing xml file: "<<object.c_str());

		reglib::Model * mod = new reglib::Model();
		QXmlStreamReader* xmlReader = new QXmlStreamReader(&file);

		while (!xmlReader->atEnd() && !xmlReader->hasError()){
			QXmlStreamReader::TokenType token = xmlReader->readNext();
			if (token == QXmlStreamReader::StartDocument)
				continue;

			if (xmlReader->hasError()){
				ROS_ERROR("XML error: %s in %s",xmlReader->errorString().toStdString().c_str(),object.c_str());
				break;
			}

			QString elementName = xmlReader->name().toString();

			if (token == QXmlStreamReader::StartElement){
				if (xmlReader->name() == "Mask"){
					int number = 0;
					cv::Mat mask;
					QXmlStreamAttributes attributes = xmlReader->attributes();
					if (attributes.hasAttribute("filename")){
						QString maskpath = attributes.value("filename").toString();
						//printf("mask filename: %s\n",(sweep_folder+maskpath.toStdString()).c_str());
						mask = cv::imread(sweep_folder+"/"+(maskpath.toStdString().c_str()), CV_LOAD_IMAGE_UNCHANGED);
						//mask = cv::imread((maskpath.toStdString()).c_str(), CV_LOAD_IMAGE_UNCHANGED);
					}else{break;}


					if (attributes.hasAttribute("image_number")){
						QString depthpath = attributes.value("image_number").toString();
						number = atoi(depthpath.toStdString().c_str());
						printf("number: %i\n",number);
					}else{break;}

					mod->frames.push_back(frames[number]->clone());
					mod->relativeposes.push_back(poses[number]);
					mod->modelmasks.push_back(new reglib::ModelMask(mask));
				}
			}
		}

		delete xmlReader;
		models.push_back(mod);
	}

	for(unsigned int i = 0; i < frames.size(); i++){delete frames[i];}

	return models;
}



*/


//#include "ros/ros.h"
//#include "std_msgs/String.h"
//#include <sensor_msgs/PointCloud2.h>
//#include <pcl_ros/point_cloud.h>
//#include <pcl/ros/conversions.h>
//#include <pcl/point_cloud.h>
//#include <pcl/point_types.h>
//#include <pcl/io/pcd_io.h>
//#include <pcl_conversions/pcl_conversions.h>
//#include <sensor_msgs/PointCloud2.h>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include "core/RGBDFrame.h"
//#include <string.h>
//#include "Util/Util.h"
//#include "core/DescriptorExtractor.h"
//#include "ModelDatabase/ModelDatabase.h"
//#include "ModelStorage/ModelStorage.h"
//#include "CameraOptimizer/CameraOptimizer.h"


#include "Util/Util.h"

using namespace std;
using namespace semantic_map_load_utilties;
typedef pcl::PointXYZRGB PointType;

boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
int visualization_lvl		= 0;

std::string getRoomFolder(std::string xmlFile){
    QString xmlFileQS(xmlFile.c_str());
    int index = xmlFileQS.lastIndexOf('/');
    return xmlFileQS.left(index).toStdString();
}

std::vector<cv::Mat> getRGBvec(std::string xmlFile){
    std::vector<cv::Mat> rgbs;
    std::string roomFolder = getRoomFolder(xmlFile);

    QFile file((roomFolder+"/ViewGroup.xml").c_str());
    if (!file.exists()){
        ROS_ERROR("Could not open file %s to load room.",(roomFolder+"/ViewGroup.xml").c_str());
        return rgbs;
    }

    file.open(QIODevice::ReadOnly);

    ROS_INFO_STREAM("Parsing xml file: "<<(roomFolder+"/ViewGroup.xml").c_str());

    QXmlStreamReader* xmlReader = new QXmlStreamReader(&file);

    while (!xmlReader->atEnd() && !xmlReader->hasError()){
        QXmlStreamReader::TokenType token = xmlReader->readNext();
        if (token == QXmlStreamReader::StartDocument)
            continue;

        if (xmlReader->hasError()){
            ROS_ERROR("XML error: %s in %s",xmlReader->errorString().toStdString().c_str(), (roomFolder+"/ViewGroup.xml").c_str());
            return rgbs;
        }

        QString elementName = xmlReader->name().toString();

        if (token == QXmlStreamReader::StartElement){
            if (xmlReader->name() == "View"){
                QXmlStreamAttributes attributes = xmlReader->attributes();
                if (attributes.hasAttribute("RGB")){
                     rgbs.push_back(cv::imread(roomFolder+"/"+attributes.value("RGB").toString().toStdString(), CV_LOAD_IMAGE_UNCHANGED));
                }else{break;}
            }
        }
    }
    delete xmlReader;
    return rgbs;
}

void getQuasimodoObjects(std::string path, std::vector< std::vector<cv::Mat > > & all_objectMasks, std::vector< std::vector<unsigned int > > & all_imgNumber, std::vector<std::string > & all_names){
    std::string roomFolder = getRoomFolder(path);
    QStringList objectFiles = QDir(roomFolder.c_str()).entryList(QStringList("dynamic_obj*.xml"));
    if(objectFiles.size() == 0){return;}

    for (auto objectFile : objectFiles){
        std::string object = roomFolder+"/"+objectFile.toStdString();
        printf("object: %s\n",object.c_str());

        std::vector<cv::Mat > objectMasks;
        std::vector<unsigned int > imgNumber;

        QFile objfile(object.c_str());

        if (!objfile.exists()){
            ROS_ERROR("Could not open file %s to masks.",object.c_str());
            continue;
        }

        objfile.open(QIODevice::ReadOnly);
        ROS_INFO_STREAM("Parsing xml file: "<<object.c_str());

        QXmlStreamReader* objxmlReader = new QXmlStreamReader(&objfile);

        while (!objxmlReader->atEnd() && !objxmlReader->hasError()){
            QXmlStreamReader::TokenType token = objxmlReader->readNext();
            if (token == QXmlStreamReader::StartDocument)
                continue;

            if (objxmlReader->hasError()){
                ROS_ERROR("XML error: %s in %s",objxmlReader->errorString().toStdString().c_str(),object.c_str());
                break;
            }

            QString elementName = objxmlReader->name().toString();

            if (token == QXmlStreamReader::StartElement){
                if (objxmlReader->name() == "Mask"){
                    int number = 0;
                    QXmlStreamAttributes attributes = objxmlReader->attributes();
                    if (attributes.hasAttribute("filename")){
                        QString maskpath = attributes.value("filename").toString();
                        objectMasks.push_back(cv::imread(roomFolder+"/"+(maskpath.toStdString().c_str()), CV_LOAD_IMAGE_UNCHANGED));
                    }else{break;}

                    if (attributes.hasAttribute("image_number")){
                        QString depthpath = attributes.value("image_number").toString();
                        number = atoi(depthpath.toStdString().c_str());
                        imgNumber.push_back(number);
                    }else{break;}
                }
            }
        }
        all_objectMasks.push_back(objectMasks);
        all_imgNumber.push_back(imgNumber);
        all_names.push_back(roomFolder+"/"+objectFile.toStdString());
    }
}

void getMetaroomObjects(std::string path, std::vector< std::vector<cv::Mat > > & all_objectMasks, std::vector< std::vector<unsigned int > > & all_imgNumber, std::vector<std::string > & all_names){
    auto objects = semantic_map_load_utilties::loadAllDynamicObjectsFromSingleSweep<PointType>(path,true);
    for (auto object : objects){
        std::cout <<"Object has "<<object.vAdditionalViews.size()<<" additional clouds "<<std::endl;
        /*
            if (object.intermediateCloud->points.size()){

                ROS_INFO_STREAM("The object contains "<<object.vAdditionalViewMaskImages.size()<<" additional image masks");
                for (size_t i=0; i<object.vAdditionalViewMaskImages.size(); i++){
                    ROS_INFO_STREAM("Additional image mask "<<i<<" number of indices "<<object.vAdditionalViewMaskIndices[i].size());
                    cv::imshow( "Display window", object.vAdditionalViewMaskImages[i] );
                    cv::waitKey(0);

                }
            } else {
                std::cout<<"Intermediate cloud hasn't been set for this dynamic object!!!"<<std::endl;
            }
        }
        */
    }
}

int annotateObject(std::vector< cv::Mat > rgbs, std::vector<cv::Mat > objectMasks, std::vector<unsigned int > imgNumber){
    int anno = 0;

    int max_pixel = 0;
    int current_displayInd = 0;
    for(unsigned int i = 0; i < objectMasks.size(); i++){
        unsigned char * data = objectMasks[i].data;
        unsigned int pixels = objectMasks[i].rows * objectMasks[i].cols;
        unsigned int nrp = 0;
        for(unsigned int p = 0; p < pixels; p++){
            nrp +=  data[p] != 0;
        }
        if(nrp > max_pixel){
            max_pixel = nrp;
            current_displayInd = i;
        }
    }

    int fontFace = cv::FONT_HERSHEY_COMPLEX_SMALL;
    double fontScale = 1;
    int thickness = 1;
    int state = 0;

    cv::Mat rgb = rgbs[imgNumber[current_displayInd]].clone();
    cv::Mat mask = objectMasks[current_displayInd].clone();
    while(true){
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;

        cv::findContours( mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
        for( unsigned int i = 0; i < contours.size(); i++ ){
            cv::drawContours( rgb, contours, i, cv::Scalar( 0, 0, 255 ), 2, 8, hierarchy, 0, cv::Point() );
            cv::drawContours( rgb, contours, i, cv::Scalar( 0, 255, 0 ), 1, 8, hierarchy, 0, cv::Point() );
        }

        unsigned int height	= rgb.rows;
        unsigned int width	= rgb.cols;
        unsigned int newwidth = width+600;

        unsigned char * rgbdata = rgb.data;

        cv::Mat img;
        img.create(height,newwidth,CV_8UC3);
        unsigned char * imgdata = img.data;

        for(unsigned int i = 0; i < 3*height*newwidth; i++){imgdata[i] = 0;}

        for(unsigned int w = 0; w < width;w++){
            for(unsigned int h = 0; h < height;h++){
                int oind = h*width+w;
                int nind = h*newwidth+w;
                imgdata[3*nind + 0] = rgbdata[3*oind + 0];
                imgdata[3*nind + 1] = rgbdata[3*oind + 1];
                imgdata[3*nind + 2] = rgbdata[3*oind + 2];
            }
        }

        int textnr = 0;
        putText(img, "1: correct                   ",       cv::Point(width+10,	30+(textnr++)*25   ),	fontFace, fontScale, cv::Scalar::all(255), thickness, 8);
        putText(img, "2: junk                      ",       cv::Point(width+10,	30+(textnr++)*25   ),	fontFace, fontScale, cv::Scalar::all(255), thickness, 8);
        putText(img, "3: undersegmented (correct+junk",     cv::Point(width+10,	30+(textnr++)*25   ),	fontFace, fontScale, cv::Scalar::all(255), thickness, 8);
        putText(img, "4: undersegmented (correct+correct)", cv::Point(width+10,	30+(textnr++)*25   ),	fontFace, fontScale, cv::Scalar::all(255), thickness, 8);
        putText(img, "5: severly oversegmented",            cv::Point(width+10,	30+(textnr++)*25   ),	fontFace, fontScale, cv::Scalar::all(255), thickness, 8);
        putText(img, "6: unknown",                          cv::Point(width+10,	30+(textnr++)*25   ),	fontFace, fontScale, cv::Scalar::all(255), thickness, 8);
        putText(img, "next: W previous: Q",                 cv::Point(width+10,	30+(textnr++)*25   ),	fontFace, fontScale, cv::Scalar::all(255), thickness, 8);

        char buf [1024];
        sprintf(buf,"   Image:     %i / %i",current_displayInd+1,objectMasks.size());
        putText(img, std::string(buf), cv::Point(width+10,			30+(textnr++)*25   ), fontFace, fontScale, cv::Scalar::all(255), thickness, 8);

//        putText(img, "   Select state(pushed CTRL)",cv::Point(width+10,	30+(textnr++)*25   ),	fontFace, fontScale, cv::Scalar::all(255), thickness, 8);
//        putText(img, "--", cv::Point(width+5,	30+state*25   ), fontFace, fontScale, cv::Scalar(0,255,0), thickness, 8);

        cv::namedWindow( "Annotation tool",				cv::WINDOW_AUTOSIZE );	cv::imshow( "Annotation tool",img );
        char c = cv::waitKey(0);
        if(c == 'q' || c == 'Q'){
            current_displayInd = std::max(int(current_displayInd-1),0);
            rgb = rgbs[imgNumber[current_displayInd]].clone();
            mask = objectMasks[current_displayInd].clone();
        }

        if(c == 'w' || c == 'W'){
            current_displayInd = std::min(int(current_displayInd+1),int(imgNumber.size()-1));
            rgb = rgbs[imgNumber[current_displayInd]].clone();
            mask = objectMasks[current_displayInd].clone();
        }

        if(c == '1'){return 1;}
        if(c == '2'){return 2;}
        if(c == '3'){return 3;}
        if(c == '4'){return 4;}
        if(c == '5'){return 5;}
        if(c == '6'){return 5;}
    }

    return anno;
}

std::vector<int> annotateObjects(std::vector< cv::Mat > rgbs, std::vector< std::vector<cv::Mat > > & all_objectMasks, std::vector< std::vector<unsigned int > > & all_imgNumber){
    std::vector<int> annos;
    for(unsigned int i = 0; i < all_objectMasks.size(); i++){
        annos.push_back(annotateObject(rgbs,all_objectMasks[i], all_imgNumber[i]));
    }
    return annos;
}

void saveAnnotationResults(std::string filename, std::vector<int> labels,  std::vector<std::string > names){
    std::string total = "";
    for(unsigned int i = 0; i < labels.size(); i++){
        total += std::to_string(labels[i])+" "+names[i]+"\n";
    }
    std::cout << total << std::endl;

    std::ofstream myfile;
    myfile.open (filename);
    myfile << total;
    myfile.close();

}

bool annotate(std::string path){
    printf("annotate: %s\n",path.c_str());

    std::string roomFolder = getRoomFolder(path);

    std::vector<cv::Mat> rgbs =  getRGBvec(path);

	if(true){
    std::vector< std::vector<cv::Mat > > quasuimodo_all_objectMasks;
    std::vector< std::vector<unsigned int > > quasuimodo_all_imgNumber;
    std::vector<std::string > quasuimodo_all_names;
    getQuasimodoObjects(path,quasuimodo_all_objectMasks,quasuimodo_all_imgNumber,quasuimodo_all_names);

    std::vector<int> quasimodo_labels = annotateObjects(rgbs,quasuimodo_all_objectMasks,quasuimodo_all_imgNumber);
    saveAnnotationResults(roomFolder+"quasimodo_annotation.txt", quasimodo_labels, quasuimodo_all_names);
    }

	if(false){
    std::vector< std::vector<cv::Mat > > metaroom_all_objectMasks;
    std::vector< std::vector<unsigned int > > metaroom_all_imgNumber;
    std::vector<std::string > metaroom_all_names;
    getMetaroomObjects(path,metaroom_all_objectMasks,metaroom_all_imgNumber,metaroom_all_names);

    std::vector<int> metaroom_labels = annotateObjects(rgbs,metaroom_all_objectMasks,metaroom_all_imgNumber);
    saveAnnotationResults(roomFolder+"metaroom_annotation.txt", metaroom_labels, metaroom_all_names);
    }
    return true;
}

bool annotateFiles(std::string path){
    vector<string> sweep_xmls = semantic_map_load_utilties::getSweepXmls<PointType>(path);
    for (auto sweep_xml : sweep_xmls) {
        annotate(sweep_xml);
    }
    return false;
}

int main(int argc, char** argv){

	visualization_lvl = 0;


	std::vector< std::string > folders;
	int inputstate = 0;
	for(int i = 1; i < argc;i++){
		printf("input: %s\n",argv[i]);
		if(std::string(argv[i]).compare("-v") == 0){			inputstate	= 1;}
		else if(std::string(argv[i]).compare("-folder") == 0){	inputstate	= 2;}
		else if(std::string(argv[i]).compare("-folders") == 0){	inputstate	= 2;}
		else if(std::string(argv[i]).compare("-file") == 0){	inputstate	= 2;}
		else if(std::string(argv[i]).compare("-files") == 0){	inputstate	= 2;}
		else if(inputstate == 1){visualization_lvl = atoi(argv[i]);}
		else if(inputstate == 2){folders.push_back(std::string(argv[i]));}
	}

	if(visualization_lvl > 0){
		viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("3D Viewer"));
		viewer->setBackgroundColor (0.5, 0, 0.5);
		viewer->addCoordinateSystem (1.0);
		viewer->initCameraParameters ();
	}

	if(folders.size() == 0){folders.push_back(std::string(getenv ("HOME"))+"/.semanticMap/");}

    for(unsigned int i = 0; i < folders.size(); i++){annotateFiles(folders[i]);}

    printf("done annotating\n");

	return 0;
}
