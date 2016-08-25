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

ros::ServiceClient segmentation_client;
using namespace std;
using namespace semantic_map_load_utilties;

typedef pcl::PointXYZRGB PointType;
typedef pcl::PointCloud<PointType> Cloud;
typedef typename Cloud::Ptr CloudPtr;
typedef pcl::search::KdTree<PointType> Tree;
typedef semantic_map_load_utilties::DynamicObjectData<PointType> ObjectData;

using namespace std;
using namespace semantic_map_load_utilties;

std::string overall_folder = "~/.semanticMap/";
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
int visualization_lvl = 0;
std::string outtopic = "/some/topic";
ros::Publisher out_pub;
std::string posepath = "";
std::vector<Eigen::Matrix4d> sweepPoses;

void remove_old_seg(std::string sweep_folder){
	char buf [1024];
	sprintf(buf,"rm %s/dynamic_*.png",sweep_folder.c_str());
	printf("%s\n",buf);

	sprintf(buf,"rm %s/moving_*.png",sweep_folder.c_str());
	printf("%s\n",buf);
}

void writePose(QXmlStreamWriter* xmlWriter, Eigen::Matrix4d pose){
	Eigen::Quaterniond q = Eigen::Quaterniond ((Eigen::Affine3d (pose)).rotation());

	xmlWriter->writeStartElement("Translation");
	xmlWriter->writeStartElement("x");
	xmlWriter->writeCharacters(QString::number(pose(0,3)));
	xmlWriter->writeEndElement();
	xmlWriter->writeStartElement("y");
	xmlWriter->writeCharacters(QString::number(pose(1,3)));
	xmlWriter->writeEndElement();
	xmlWriter->writeStartElement("z");
	xmlWriter->writeCharacters(QString::number(pose(2,3)));
	xmlWriter->writeEndElement();
	xmlWriter->writeEndElement(); // Translation

	xmlWriter->writeStartElement("Rotation");
	xmlWriter->writeStartElement("w");
	xmlWriter->writeCharacters(QString::number(q.w()));
	xmlWriter->writeEndElement();
	xmlWriter->writeStartElement("x");
	xmlWriter->writeCharacters(QString::number(q.x()));
	xmlWriter->writeEndElement();
	xmlWriter->writeStartElement("y");
	xmlWriter->writeCharacters(QString::number(q.y()));
	xmlWriter->writeEndElement();
	xmlWriter->writeStartElement("z");
	xmlWriter->writeCharacters(QString::number(q.z()));
	xmlWriter->writeEndElement();
	xmlWriter->writeEndElement(); //Rotation
}

void writeXml(std::string xmlFile, std::vector<reglib::RGBDFrame *> & frames, std::vector<Eigen::Matrix4d> & poses){
	int slash_pos = xmlFile.find_last_of("/");
	std::string sweep_folder = xmlFile.substr(0, slash_pos) + "/";

	QFile file(xmlFile.c_str());
	if (file.exists()){file.remove();}


	if (!file.open(QIODevice::ReadWrite | QIODevice::Text))
	{
		std::cerr<<"Could not open file "<<sweep_folder<<"additionalViews.xml to save views as XML"<<std::endl;
		return;
	}

	QXmlStreamWriter* xmlWriter = new QXmlStreamWriter();
	xmlWriter->setDevice(&file);

	xmlWriter->writeStartDocument();
	xmlWriter->writeStartElement("Views");
	xmlWriter->writeAttribute("number_of_views", QString::number(frames.size()));
	for(unsigned int i = 0; i < frames.size(); i++){
		reglib::RGBDFrame * frame = frames[i];
		char buf [1024];

		xmlWriter->writeStartElement("View");
		sprintf(buf,"%s/view_RGB%10.10i.png",sweep_folder.c_str(),i);
		cv::imwrite(buf, frame->rgb );
		xmlWriter->writeAttribute("RGB", QString(buf));


		sprintf(buf,"%s/view_DEPTH%10.10i.png",sweep_folder.c_str(),i);
		cv::imwrite(buf, frame->depth );
		xmlWriter->writeAttribute("DEPTH", QString(buf));

		long nsec = 1e9*((frame->capturetime)-std::floor(frame->capturetime));
		long sec = frame->capturetime;
		xmlWriter->writeStartElement("Stamp");

		xmlWriter->writeStartElement("sec");
		xmlWriter->writeCharacters(QString::number(sec));
		xmlWriter->writeEndElement();

		xmlWriter->writeStartElement("nsec");
		xmlWriter->writeCharacters(QString::number(nsec));
		xmlWriter->writeEndElement();

		xmlWriter->writeEndElement(); // Stamp

		xmlWriter->writeStartElement("Camera");

		xmlWriter->writeStartElement("fx");
		xmlWriter->writeCharacters(QString::number(frame->camera->fx));
		xmlWriter->writeEndElement();

		xmlWriter->writeStartElement("fy");
		xmlWriter->writeCharacters(QString::number(frame->camera->fy));
		xmlWriter->writeEndElement();

		xmlWriter->writeStartElement("cx");
		xmlWriter->writeCharacters(QString::number(frame->camera->cx));
		xmlWriter->writeEndElement();

		xmlWriter->writeStartElement("cy");
		xmlWriter->writeCharacters(QString::number(frame->camera->cy));
		xmlWriter->writeEndElement();

		xmlWriter->writeEndElement(); // camera

		xmlWriter->writeStartElement("RegisteredPose");
		writePose(xmlWriter,poses[i]);
		xmlWriter->writeEndElement();

		xmlWriter->writeStartElement("Pose");
		writePose(xmlWriter,frame->pose);
		xmlWriter->writeEndElement();

		xmlWriter->writeEndElement();
	}
	xmlWriter->writeEndElement(); // Semantic Room
	xmlWriter->writeEndDocument();
	delete xmlWriter;
}

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
	QString roomFolder = xmlFileQS.left(index);


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
			ROS_ERROR("XML error: %s",xmlReader->errorString().toStdString().c_str());
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
					QString rgbpath = attributes.value("RGB").toString();
					//printf("rgb filename: %s\n",rgbpath.toStdString().c_str());
					rgb = cv::imread(rgbpath.toStdString().c_str(), CV_LOAD_IMAGE_UNCHANGED);
				}else{break;}


				if (attributes.hasAttribute("DEPTH"))
				{
					QString depthpath = attributes.value("DEPTH").toString();
					//printf("depth filename: %s\n",depthpath.toStdString().c_str());
					depth = cv::imread(depthpath.toStdString().c_str(), CV_LOAD_IMAGE_UNCHANGED);
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

reglib::Model * processAV(std::string path){
	printf("processAV: %s\n",path.c_str());

	int slash_pos = path.find_last_of("/");
	std::string sweep_folder = path.substr(0, slash_pos) + "/";

	std::vector<cv::Mat> viewrgbs;
	std::vector<cv::Mat> viewdepths;
	std::vector<tf::StampedTransform > viewtfs;

	QStringList objectFiles = QDir(sweep_folder.c_str()).entryList(QStringList("*object*.xml"));
	for (auto objectFile : objectFiles){
		auto object = loadDynamicObjectFromSingleSweep<PointType>(sweep_folder+objectFile.toStdString(),false);
		for (unsigned int i=0; i<object.vAdditionalViews.size(); i++){
			CloudPtr cloud = object.vAdditionalViews[i];

			cv::Mat rgb;
			rgb.create(cloud->height,cloud->width,CV_8UC3);
			unsigned char * rgbdata = (unsigned char *)rgb.data;

			cv::Mat depth;
			depth.create(cloud->height,cloud->width,CV_16UC1);
			unsigned short * depthdata = (unsigned short *)depth.data;

			unsigned int nr_data = cloud->height * cloud->width;
			for(unsigned int j = 0; j < nr_data; j++){
				PointType p = cloud->points[j];
				rgbdata[3*j+0]	= p.b;
				rgbdata[3*j+1]	= p.g;
				rgbdata[3*j+2]	= p.r;
				depthdata[j]	= short(5000.0 * p.z);
			}

			viewrgbs.push_back(rgb);
			viewdepths.push_back(depth);
			viewtfs.push_back(object.vAdditionalViewsTransforms[i]);
		}
	}

	reglib::Model * sweep = quasimodo_brain::load_metaroom_model(path);
	for(unsigned int i = 0; (i < sweep->frames.size()) && (sweep->frames.size() == sweepPoses.size()) ; i++){
		sweep->frames[i]->pose	= sweep->frames.front()->pose * sweepPoses[i];
		sweep->relativeposes[i] = sweepPoses[i];
	}

	std::vector<reglib::RGBDFrame *> frames;
	std::vector<reglib::ModelMask *> masks;
	std::vector<Eigen::Matrix4d> unrefined;

	std::vector<Eigen::Matrix4d> both_unrefined;
	both_unrefined.push_back(Eigen::Matrix4d::Identity());
	std::vector<double> times;
	for(unsigned int i = 0; i < viewrgbs.size(); i++){
		printf("additional view: %i\n",i);
		geometry_msgs::TransformStamped msg;
		tf::transformStampedTFToMsg(viewtfs[i], msg);
		long sec = msg.header.stamp.sec;
		long nsec = msg.header.stamp.nsec;
		double time = double(sec)+1e-9*double(nsec);

		Eigen::Matrix4d m = quasimodo_brain::getMat(viewtfs[i]);

		cout << m << endl << endl;

		unrefined.push_back(m);
		times.push_back(time);

		cv::Mat fullmask;
		fullmask.create(480,640,CV_8UC1);
		unsigned char * maskdata = (unsigned char *)fullmask.data;
		for(int j = 0; j < 480*640; j++){maskdata[j] = 255;}
		masks.push_back(new reglib::ModelMask(fullmask));

		reglib::Camera * cam		= sweep->frames.front()->camera->clone();
		reglib::RGBDFrame * frame	= new reglib::RGBDFrame(cam,viewrgbs[i],viewdepths[i],time, m);//a.matrix());
		frames.push_back(frame);

		//both_unrefined.push_back(sweep->frames.front()->pose.inverse()*a.matrix());
		both_unrefined.push_back(sweep->frames.front()->pose.inverse()*m);
	}

	reglib::RegistrationRandom *	reg	= new reglib::RegistrationRandom();
	reglib::ModelUpdaterBasicFuse * mu	= new reglib::ModelUpdaterBasicFuse( sweep, reg);
	mu->occlusion_penalty               = 15;
	mu->massreg_timeout                 = 60*4;
	mu->viewer							= viewer;

	sweep->points = mu->getSuperPoints(sweep->relativeposes,sweep->frames,sweep->modelmasks,1,false);

	//Not needed if metaroom well calibrated
	reglib::MassRegistrationPPR2 * bgmassreg = new reglib::MassRegistrationPPR2(0.15);
	bgmassreg->timeout = 20;
	bgmassreg->viewer = viewer;
	bgmassreg->use_surface = true;
	bgmassreg->use_depthedge = false;
	bgmassreg->visualizationLvl = 0;
	bgmassreg->maskstep = 5;
	bgmassreg->nomaskstep = 5;
	bgmassreg->nomask = true;
	bgmassreg->stopval = 0.0005;
	bgmassreg->addModel(sweep);
	bgmassreg->setData(frames,masks);
	reglib::MassFusionResults bgmfr = bgmassreg->getTransforms(both_unrefined);
	delete bgmassreg;

	for(unsigned int i = 0; i < frames.size(); i++){
		frames[i]->pose = sweep->frames.front()->pose * bgmfr.poses[i+1];
	}

	reglib::Model * fullmodel = new reglib::Model();
	fullmodel->frames = sweep->frames;
	fullmodel->relativeposes = sweep->relativeposes;
	fullmodel->modelmasks = sweep->modelmasks;
	for(unsigned int i = 0; i < frames.size(); i++){
		fullmodel->frames.push_back(frames[i]);
		fullmodel->modelmasks.push_back(masks[i]);
		fullmodel->relativeposes.push_back(bgmfr.poses[i+1]);
	}


	delete reg;
	delete mu;
	delete sweep;

	return fullmodel;
}

void savePoses(std::string xmlFile, std::vector<Eigen::Matrix4d> poses, int maxposes = -1){
	QFile file(xmlFile.c_str());
	if (file.exists()){file.remove();}

	if (!file.open(QIODevice::ReadWrite | QIODevice::Text)){
		std::cerr<<"Could not open file "<< xmlFile <<" to save views as XML"<<std::endl;
		return;
	}

	QXmlStreamWriter* xmlWriter = new QXmlStreamWriter();
	xmlWriter->setDevice(&file);

	xmlWriter->writeStartDocument();
	xmlWriter->writeStartElement("Poses");
	//	xmlWriter->writeAttribute("number_of_poses", QString::number(poses.size()));
	for(unsigned int i = 0; i < poses.size() && (i == -1 || i < maxposes); i++){
		printf("saving %i\n",i);
		xmlWriter->writeStartElement("Pose");
		writePose(xmlWriter,poses[i]);
		xmlWriter->writeEndElement();
	}
	xmlWriter->writeEndElement(); // Poses
	xmlWriter->writeEndDocument();
	delete xmlWriter;
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
			ROS_ERROR("XML error: %s",xmlReader->errorString().toStdString().c_str());
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
	return poses;
}

void processMetaroom(std::string path){
	printf("processing: %s\n",path.c_str());

	int slash_pos = path.find_last_of("/");
	std::string sweep_folder = path.substr(0, slash_pos) + "/";

	reglib::Model * fullmodel = processAV(path);

	savePoses(sweep_folder+"testposes.xml",fullmodel->relativeposes,17);

	writeXml(path+"ViewGroup.xml",fullmodel->frames,fullmodel->relativeposes);

	reglib::RegistrationRandom *	reg	= new reglib::RegistrationRandom();
	reglib::ModelUpdaterBasicFuse * mu	= new reglib::ModelUpdaterBasicFuse( fullmodel, reg);
	mu->occlusion_penalty               = 15;
	mu->massreg_timeout                 = 60*4;
	mu->viewer							= viewer;

	std::vector<Eigen::Matrix4d> po;
	std::vector<reglib::RGBDFrame*> fr;
	std::vector<reglib::ModelMask*> mm;
	fullmodel->getData(po, fr, mm);
	fullmodel->points = mu->getSuperPoints(po,fr,mm,1,false);

	SimpleXMLParser<pcl::PointXYZRGB> parser;
	SimpleXMLParser<pcl::PointXYZRGB>::RoomData current_roomData  = parser.loadRoomFromXML(path);
	std::string current_waypointid = current_roomData.roomWaypointId;

	if(overall_folder.back() == '/'){overall_folder.pop_back();}

	int prevind = -1;
	std::vector<std::string> sweep_xmls = semantic_map_load_utilties::getSweepXmls<pcl::PointXYZRGB>(overall_folder);
	for (int i = 0; i < sweep_xmls.size(); i++){
		SimpleXMLParser<pcl::PointXYZRGB>::RoomData other_roomData  = parser.loadRoomFromXML(sweep_xmls[i],std::vector<std::string>(),false,false);
		std::string other_waypointid = other_roomData.roomWaypointId;

		if(sweep_xmls[i].compare(path) == 0){break;}
		if(other_waypointid.compare(current_waypointid) == 0){prevind = i;}
	}

	if(prevind != -1){
		std::string prev = sweep_xmls[prevind];
		printf("prev: %s\n",prev.c_str());

		reglib::Model * bg = quasimodo_brain::load_metaroom_model(prev);
		bg->points = mu->getSuperPoints(bg->relativeposes,bg->frames,bg->modelmasks,1,false);

		std::vector< reglib::Model * > models;
		models.push_back(fullmodel);

		std::vector< std::vector< cv::Mat > > internal;
		std::vector< std::vector< cv::Mat > > external;
		std::vector< std::vector< cv::Mat > > dynamic;

		quasimodo_brain::segment(bg,models,internal,external,dynamic,false);

		bg->fullDelete();
		delete bg;

		remove_old_seg(sweep_folder);

		for(unsigned int i = 0; i < models.size(); i++){
			std::vector<cv::Mat> internal_masks = internal[i];
			std::vector<cv::Mat> external_masks = external[i];
			std::vector<cv::Mat> dynamic_masks	= dynamic[i];
			reglib::Model * model = models[i];


			std::vector<Eigen::Matrix4d> mod_po;
			std::vector<reglib::RGBDFrame*> mod_fr;
			std::vector<reglib::ModelMask*> mod_mm;
			model->getData(mod_po, mod_fr, mod_mm);

			std::vector<int> dynamic_frameid;
			std::vector<int> dynamic_pixelid;

			std::vector<int> moving_frameid;
			std::vector<int> moving_pixelid;

			pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr dynamiccloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
			pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr movingcloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
			pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

			for(unsigned int j = 0; j < mod_fr.size(); j++){
				reglib::RGBDFrame * frame = mod_fr[j];
				Eigen::Matrix4d p = mod_po[j];
				unsigned char  * rgbdata		= (unsigned char	*)(frame->rgb.data);
				unsigned short * depthdata		= (unsigned short	*)(frame->depth.data);
				float		   * normalsdata	= (float			*)(frame->normals.data);

				reglib::Camera * camera = frame->camera;

				unsigned char * internalmaskdata = (unsigned char *)(internal_masks[j].data);
				unsigned char * externalmaskdata = (unsigned char *)(external_masks[j].data);
				unsigned char * dynamicmaskdata = (unsigned char *)(dynamic_masks[j].data);

				float m00 = p(0,0); float m01 = p(0,1); float m02 = p(0,2); float m03 = p(0,3);
				float m10 = p(1,0); float m11 = p(1,1); float m12 = p(1,2); float m13 = p(1,3);
				float m20 = p(2,0); float m21 = p(2,1); float m22 = p(2,2); float m23 = p(2,3);

				const float idepth			= camera->idepth_scale;
				const float cx				= camera->cx;
				const float cy				= camera->cy;
				const float ifx				= 1.0/camera->fx;
				const float ify				= 1.0/camera->fy;
				const unsigned int width	= camera->width;
				const unsigned int height	= camera->height;


				for(unsigned int w = 0; w < width;w++){
					for(unsigned int h = 0; h < height;h++){
						int ind = h*width+w;
						float z = idepth*float(depthdata[ind]);
						if(z > 0){
							float x = (float(w) - cx) * z * ifx;
							float y = (float(h) - cy) * z * ify;

							pcl::PointXYZRGBNormal point;
							point.x = m00*x + m01*y + m02*z + m03;
							point.y = m10*x + m11*y + m12*z + m13;
							point.z = m20*x + m21*y + m22*z + m23;

							point.b = rgbdata[3*ind+0];
							point.g = rgbdata[3*ind+1];
							point.r = rgbdata[3*ind+2];

							if(dynamicmaskdata[ind] != 0){
								dynamiccloud->points.push_back(point);
								dynamic_frameid.push_back(j);
								dynamic_pixelid.push_back(ind);
								point.b = 0;
								point.g = 255;
								point.r = 0;
							}else if(internalmaskdata[ind] == 0){
								movingcloud->points.push_back(point);
								moving_frameid.push_back(j);
								moving_pixelid.push_back(ind);
								point.b = 0;
								point.g = 0;
								point.r = 255;
							}

							cloud->points.push_back(point);
						}
					}
				}
			}

			printf("dynamiccloud: %i\n",dynamiccloud->points.size());
			if(dynamiccloud->points.size() > 0){
				// Creating the KdTree object for the search method of the extraction
				pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr dynamictree (new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
				dynamictree->setInputCloud (dynamiccloud);

				std::vector<pcl::PointIndices> dynamic_indices;
				pcl::EuclideanClusterExtraction<pcl::PointXYZRGBNormal> dynamic_ec;
				dynamic_ec.setClusterTolerance (0.02); // 2cm
				dynamic_ec.setMinClusterSize (500);
				dynamic_ec.setMaxClusterSize (250000000);
				dynamic_ec.setSearchMethod (dynamictree);
				dynamic_ec.setInputCloud (dynamiccloud);
				dynamic_ec.extract (dynamic_indices);

				for (unsigned int d = 0; d < dynamic_indices.size(); d++){
					std::vector< std::vector<int> > inds;
					inds.resize(mod_fr.size());

					pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
					for (unsigned int ind = 0; ind < dynamic_indices[d].indices.size(); ind++){
						int pid = dynamic_indices[d].indices[ind];
						inds[dynamic_frameid[pid]].push_back(dynamic_pixelid[pid]);
						cloud_cluster->points.push_back(dynamiccloud->points[pid]);
					}

					char buf [1024];
					sprintf(buf,"%s/dynamic_object%10.10i.xml",sweep_folder.c_str(),d);
					QFile file(buf);
					if (file.exists()){file.remove();}
					if (!file.open(QIODevice::ReadWrite | QIODevice::Text)){std::cerr<<"Could not open file "<< buf <<" to save dynamic object as XML"<<std::endl;}

					QXmlStreamWriter* xmlWriter = new QXmlStreamWriter();
					xmlWriter->setDevice(&file);

					xmlWriter->writeStartDocument();
					xmlWriter->writeStartElement("Object");
					xmlWriter->writeAttribute("object_number", QString::number(d));

					for(unsigned int j = 0; j < mod_fr.size(); j++){
						if(inds[j].size() == 0){continue;}
						reglib::RGBDFrame * frame = mod_fr[j];
						reglib::Camera * cam = frame->camera;
						cv::Mat mask;
						mask.create(cam->height,cam->width,CV_8UC1);
						unsigned char * maskdata = (unsigned char *)(mask.data);
						for(unsigned int k = 0; k < cam->height*cam->width;k++){maskdata[k] = 0;}
						for(unsigned int k = 0; k < inds[j].size(); k++){maskdata[inds[j][k]] = 255;}

						char buf [1024];
						sprintf(buf,"%s/dynamicmask_%i_%i.png",sweep_folder.c_str(),d,j);
						cv::imwrite(buf, mask );
						xmlWriter->writeStartElement("Mask");
						xmlWriter->writeAttribute("filename", QString(buf));
						xmlWriter->writeAttribute("image_number", QString::number(j));
						xmlWriter->writeEndElement();
					}

					xmlWriter->writeEndElement();
					xmlWriter->writeEndDocument();
					delete xmlWriter;
				}
			}


			printf("movingcloud: %i\n",movingcloud->points.size());
			if(movingcloud->points.size() > 0){
				// Creating the KdTree object for the search method of the extraction
				pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr movingtree (new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
				movingtree->setInputCloud (movingcloud);

				std::vector<pcl::PointIndices> moving_indices;
				pcl::EuclideanClusterExtraction<pcl::PointXYZRGBNormal> moving_ec;
				moving_ec.setClusterTolerance (0.02); // 2cm
				moving_ec.setMinClusterSize (500);
				moving_ec.setMaxClusterSize (250000000);
				moving_ec.setSearchMethod (movingtree);
				moving_ec.setInputCloud (movingcloud);
				moving_ec.extract (moving_indices);

				for (unsigned int d = 0; d < moving_indices.size(); d++){
					std::vector< std::vector<int> > inds;
					inds.resize(mod_fr.size());

					pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
					for (unsigned int ind = 0; ind < moving_indices[d].indices.size(); ind++){
						int pid = moving_indices[d].indices[ind];
						inds[moving_frameid[pid]].push_back(moving_pixelid[pid]);
						cloud_cluster->points.push_back(movingcloud->points[pid]);
					}

					char buf [1024];
					sprintf(buf,"%s/moving_object%10.10i.xml",sweep_folder.c_str(),d);
					QFile file(buf);
					if (file.exists()){file.remove();}
					if (!file.open(QIODevice::ReadWrite | QIODevice::Text)){std::cerr<<"Could not open file "<< buf <<" to save moving object as XML"<<std::endl;}

					QXmlStreamWriter* xmlWriter = new QXmlStreamWriter();
					xmlWriter->setDevice(&file);

					xmlWriter->writeStartDocument();
					xmlWriter->writeStartElement("Object");
					xmlWriter->writeAttribute("object_number", QString::number(d));

					for(unsigned int j = 0; j < mod_fr.size(); j++){
						if(inds[j].size() == 0){continue;}
						reglib::RGBDFrame * frame = mod_fr[j];
						reglib::Camera * cam = frame->camera;
						cv::Mat mask;
						mask.create(cam->height,cam->width,CV_8UC1);
						unsigned char * maskdata = (unsigned char *)(mask.data);
						for(unsigned int k = 0; k < cam->height*cam->width;k++){maskdata[k] = 0;}
						for(unsigned int k = 0; k < inds[j].size(); k++){maskdata[inds[j][k]] = 255;}

						char buf [1024];
						sprintf(buf,"%s/movingmask_%i_%i.png",sweep_folder.c_str(),d,j);
						cv::imwrite(buf, mask );
						xmlWriter->writeStartElement("Mask");
						xmlWriter->writeAttribute("filename", QString(buf));
						xmlWriter->writeAttribute("image_number", QString::number(j));
						xmlWriter->writeEndElement();
					}

					xmlWriter->writeEndElement();
					xmlWriter->writeEndDocument();
					delete xmlWriter;
				}
			}
			if(visualization_lvl > 0){
				viewer->removeAllPointClouds();
				viewer->addPointCloud<pcl::PointXYZRGBNormal> (cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(cloud), "scloud");
				viewer->spin();
			}
		}

	}


	fullmodel->fullDelete();
	delete fullmodel;
	//	delete bgmassreg;
	delete reg;
	delete mu;

	std_msgs::String msg;
	msg.data = path;
	out_pub.publish(msg);
	ros::spinOnce();
}

void chatterCallback(const std_msgs::String::ConstPtr& msg){
	processMetaroom(msg->data);
}


void trainMetaroom(std::string path){
	printf("processing: %s\n",path.c_str());
	if(posepath.compare("")==0){
		printf("posepath not set, set before training\n");
		return ;
	}

	int slash_pos = path.find_last_of("/");
	std::string sweep_folder = path.substr(0, slash_pos) + "/";

	reglib::Model * fullmodel = processAV(path);

	//Not needed if metaroom well calibrated
	reglib::MassRegistrationPPR2 * bgmassreg = new reglib::MassRegistrationPPR2(0.01);
	bgmassreg->timeout = 600;
	bgmassreg->viewer = viewer;
	bgmassreg->use_surface = true;
	bgmassreg->use_depthedge = true;
	bgmassreg->visualizationLvl = 0;
	bgmassreg->maskstep = 5;
	bgmassreg->nomaskstep = 5;
	bgmassreg->nomask = true;
	bgmassreg->stopval = 0.0005;
	bgmassreg->setData(fullmodel->frames,fullmodel->modelmasks);
	reglib::MassFusionResults bgmfr = bgmassreg->getTransforms(fullmodel->relativeposes);

	savePoses(posepath,bgmfr.poses,17);
	fullmodel->fullDelete();
	delete fullmodel;
	delete bgmassreg;
}

std::vector<reglib::Model *> loadModels(std::string path){
	printf("loadModels: %s\n",path.c_str());
	std::vector<reglib::Model *> models;
	int slash_pos = path.find_last_of("/");
	std::string sweep_folder = path.substr(0, slash_pos) + "/";

	std::vector<reglib::RGBDFrame *> frames;
	std::vector<Eigen::Matrix4d> poses;
	readViewXML(sweep_folder+"ViewGroup.xml",frames,poses);

	QStringList objectFiles = QDir(sweep_folder.c_str()).entryList(QStringList("dynamic_object*.xml"));
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

/*
		while (!xmlReader->atEnd() && !xmlReader->hasError())
		{
			QXmlStreamReader::TokenType token = xmlReader->readNext();
			if (token == QXmlStreamReader::StartDocument)
				continue;

			if (xmlReader->hasError())
			{
				ROS_ERROR("XML error: %s",xmlReader->errorString().toStdString().c_str());
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
						QString rgbpath = attributes.value("RGB").toString();
						//printf("rgb filename: %s\n",rgbpath.toStdString().c_str());
						rgb = cv::imread(rgbpath.toStdString().c_str(), CV_LOAD_IMAGE_UNCHANGED);
					}else{break;}


					if (attributes.hasAttribute("DEPTH"))
					{
						QString depthpath = attributes.value("DEPTH").toString();
						//printf("depth filename: %s\n",depthpath.toStdString().c_str());
						depth = cv::imread(depthpath.toStdString().c_str(), CV_LOAD_IMAGE_UNCHANGED);
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
*/
		delete xmlReader;

//		QXmlStreamWriter* xmlWriter = new QXmlStreamWriter();
//		xmlWriter->setDevice(&file);

//		xmlWriter->writeStartDocument();
//		xmlWriter->writeStartElement("Object");
//		xmlWriter->writeAttribute("object_number", QString::number(d));

//		for(unsigned int j = 0; j < mod_fr.size(); j++){
//			if(inds[j].size() == 0){continue;}
//			reglib::RGBDFrame * frame = mod_fr[j];
//			reglib::Camera * cam = frame->camera;
//			cv::Mat mask;
//			mask.create(cam->height,cam->width,CV_8UC1);
//			unsigned char * maskdata = (unsigned char *)(mask.data);
//			for(unsigned int k = 0; k < cam->height*cam->width;k++){maskdata[k] = 0;}
//			for(unsigned int k = 0; k < inds[j].size(); k++){maskdata[inds[j][k]] = 255;}

//			char buf [1024];
//			sprintf(buf,"%s/dynamicmask_%i_%i.png",sweep_folder.c_str(),d,j);
//			cv::imwrite(buf, mask );
//			xmlWriter->writeStartElement("Mask");
//			xmlWriter->writeAttribute("filename", QString(buf));
//			xmlWriter->writeAttribute("image_number", QString::number(j));
//			xmlWriter->writeEndElement();
//		}

//		xmlWriter->writeEndElement();
//		xmlWriter->writeEndDocument();
//		delete xmlWriter;

		models.push_back(mod);

	}

//	std::vector<cv::Mat> rgbs;
//	std::vector<cv::Mat> depths;

//	for(int imgnr = 0; true; imgnr++){
//		char buf [1024];

//		sprintf(buf,"%s/view_RGB%10.10i.png",sweep_folder.c_str(),imgnr);
//		cv::Mat rgb = cv::imread(buf, CV_LOAD_IMAGE_UNCHANGED);

//		sprintf(buf,"%s/view_DEPTH%10.10i.png",sweep_folder.c_str(),imgnr);
//		cv::Mat depth = cv::imread(buf, CV_LOAD_IMAGE_UNCHANGED);
//		if(!rgb.data || !depth.data){break;}

//		rgbs.push_back(rgb);
//		depths.push_back(depth);

//		cv::namedWindow( "rgb",		cv::WINDOW_AUTOSIZE );	cv::imshow( "rgb",		rgb);
//		cv::namedWindow( "depth",	cv::WINDOW_AUTOSIZE );	cv::imshow( "depth",	depth);
//		cv::waitKey(0);
//	}

	return models;
/*


	std::vector<cv::Mat> viewrgbs;
	std::vector<cv::Mat> viewdepths;
	std::vector<tf::StampedTransform > viewtfs;

	QStringList objectFiles = QDir(sweep_folder.c_str()).entryList(QStringList("*object*.xml"));
	for (auto objectFile : objectFiles){
		auto object = loadDynamicObjectFromSingleSweep<PointType>(sweep_folder+objectFile.toStdString(),false);
		for (unsigned int i=0; i<object.vAdditionalViews.size(); i++){
			CloudPtr cloud = object.vAdditionalViews[i];

			cv::Mat rgb;
			rgb.create(cloud->height,cloud->width,CV_8UC3);
			unsigned char * rgbdata = (unsigned char *)rgb.data;

			cv::Mat depth;
			depth.create(cloud->height,cloud->width,CV_16UC1);
			unsigned short * depthdata = (unsigned short *)depth.data;

			unsigned int nr_data = cloud->height * cloud->width;
			for(unsigned int j = 0; j < nr_data; j++){
				PointType p = cloud->points[j];
				rgbdata[3*j+0]	= p.b;
				rgbdata[3*j+1]	= p.g;
				rgbdata[3*j+2]	= p.r;
				depthdata[j]	= short(5000.0 * p.z);
			}

			viewrgbs.push_back(rgb);
			viewdepths.push_back(depth);
			viewtfs.push_back(object.vAdditionalViewsTransforms[i]);
		}
	}

	reglib::Model * sweep = quasimodo_brain::load_metaroom_model(path);
	for(unsigned int i = 0; (i < sweep->frames.size()) && (sweep->frames.size() == sweepPoses.size()) ; i++){
		sweep->frames[i]->pose	= sweep->frames.front()->pose * sweepPoses[i];
		sweep->relativeposes[i] = sweepPoses[i];
	}

	std::vector<reglib::RGBDFrame *> frames;
	std::vector<reglib::ModelMask *> masks;
	std::vector<Eigen::Matrix4d> unrefined;

	std::vector<Eigen::Matrix4d> both_unrefined;
	both_unrefined.push_back(Eigen::Matrix4d::Identity());
	std::vector<double> times;
	for(unsigned int i = 0; i < viewrgbs.size(); i++){
		printf("additional view: %i\n",i);
		geometry_msgs::TransformStamped msg;
		tf::transformStampedTFToMsg(viewtfs[i], msg);
		long sec = msg.header.stamp.sec;
		long nsec = msg.header.stamp.nsec;
		double time = double(sec)+1e-9*double(nsec);

		Eigen::Matrix4d m = quasimodo_brain::getMat(viewtfs[i]);

		cout << m << endl << endl;

		unrefined.push_back(m);
		times.push_back(time);

		cv::Mat fullmask;
		fullmask.create(480,640,CV_8UC1);
		unsigned char * maskdata = (unsigned char *)fullmask.data;
		for(int j = 0; j < 480*640; j++){maskdata[j] = 255;}
		masks.push_back(new reglib::ModelMask(fullmask));

		reglib::Camera * cam		= sweep->frames.front()->camera->clone();
		reglib::RGBDFrame * frame	= new reglib::RGBDFrame(cam,viewrgbs[i],viewdepths[i],time, m);//a.matrix());
		frames.push_back(frame);

		//both_unrefined.push_back(sweep->frames.front()->pose.inverse()*a.matrix());
		both_unrefined.push_back(sweep->frames.front()->pose.inverse()*m);
	}

	reglib::RegistrationRandom *	reg	= new reglib::RegistrationRandom();
	reglib::ModelUpdaterBasicFuse * mu	= new reglib::ModelUpdaterBasicFuse( sweep, reg);
	mu->occlusion_penalty               = 15;
	mu->massreg_timeout                 = 60*4;
	mu->viewer							= viewer;

	sweep->points = mu->getSuperPoints(sweep->relativeposes,sweep->frames,sweep->modelmasks,1,false);

	//Not needed if metaroom well calibrated
	reglib::MassRegistrationPPR2 * bgmassreg = new reglib::MassRegistrationPPR2(0.01);
	bgmassreg->timeout = 20;
	bgmassreg->viewer = viewer;
	bgmassreg->use_surface = true;
	bgmassreg->use_depthedge = false;
	bgmassreg->visualizationLvl = 0;
	bgmassreg->maskstep = 5;
	bgmassreg->nomaskstep = 5;
	bgmassreg->nomask = true;
	bgmassreg->stopval = 0.0005;
	bgmassreg->addModel(sweep);
	bgmassreg->setData(frames,masks);
	reglib::MassFusionResults bgmfr = bgmassreg->getTransforms(both_unrefined);

	for(unsigned int i = 0; i < frames.size(); i++){
		frames[i]->pose = sweep->frames.front()->pose * bgmfr.poses[i+1];
	}

	reglib::Model * fullmodel = new reglib::Model();
	fullmodel->frames = sweep->frames;
	fullmodel->relativeposes = sweep->relativeposes;
	fullmodel->modelmasks = sweep->modelmasks;
	for(unsigned int i = 0; i < frames.size(); i++){
		fullmodel->frames.push_back(frames[i]);
		fullmodel->modelmasks.push_back(masks[i]);
		fullmodel->relativeposes.push_back(bgmfr.poses[i+1]);
	}

	delete bgmassreg;
	delete reg;
	delete mu;
	delete sweep;

	return fullmodel;
	*/
}

int main(int argc, char** argv){
	ros::init(argc, argv, "metaroom_additional_view_processing");
	ros::NodeHandle n;

	int inputstate = 0;
	for(int i = 1; i < argc;i++){
		printf("input: %s\n",argv[i]);
		if(std::string(argv[i]).compare("-file") == 0){				inputstate = 2;}
		else if(std::string(argv[i]).compare("-intopic") == 0){		inputstate = 0;}
		else if(std::string(argv[i]).compare("-outtopic") == 0){	inputstate = 1;}
		else if(std::string(argv[i]).compare("-folder") == 0){		inputstate = 4;}
		else if(std::string(argv[i]).compare("-train") == 0){		inputstate = 5;}
		else if(std::string(argv[i]).compare("-posepath") == 0){	inputstate = 6;}
		else if(std::string(argv[i]).compare("-loadposes") == 0){	inputstate = 7;}
		else if(std::string(argv[i]).compare("-v") == 0){
			viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("3D Viewer"));
			viewer->setBackgroundColor (0.5, 0, 0.5);
			viewer->addCoordinateSystem (1.0);
			viewer->initCameraParameters ();
			visualization_lvl = 1;
			inputstate = 3;
		}else if(inputstate == 0){
			out_pub = n.advertise<std_msgs::String>(outtopic, 1000);
			ros::Subscriber sub = n.subscribe(std::string(argv[i]), 1000, chatterCallback);
			ros::spin();
		}else if(inputstate == 1){
			outtopic = std::string(argv[i]);
		}else if(inputstate == 3){
			visualization_lvl = atoi(argv[i]);
		}else if(inputstate == 4){
			overall_folder = std::string(argv[i]);
		}else if(inputstate == 5){
			trainMetaroom(std::string(argv[i]));
		}else if(inputstate == 6){
			posepath = std::string(argv[i]);
		}else if(inputstate == 7){
			sweepPoses = readPoseXML(std::string(argv[i]));
			//exit(0);
		}else{
			out_pub = n.advertise<std_msgs::String>(outtopic, 1000);
			std::vector<reglib::Model *> mods = loadModels(std::string(argv[i]));
			exit(0);
			processMetaroom(std::string(argv[i]));
		}
	}
	return 0;
}
