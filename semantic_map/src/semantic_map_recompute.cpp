#include "semantic_map/semantic_map_node.h"

#include "eigen_conversions/eigen_msg.h"
#include "tf_conversions/tf_eigen.h"


#include "Util/Util.h"

typedef pcl::PointXYZRGB PointType;
typedef pcl::PointCloud<PointType> Cloud;
typedef typename Cloud::Ptr CloudPtr;
typedef pcl::search::KdTree<PointType> Tree;
typedef semantic_map_load_utilties::DynamicObjectData<PointType> ObjectData;

using namespace std;
using namespace semantic_map_load_utilties;

boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

std::string getPrevXML(std::string xml_file_name){
	std::cout<<"File name "<<xml_file_name<<std::endl;

	if ( ! boost::filesystem::exists( xml_file_name ) )
	{
		return "";
	}

	SemanticRoomXMLParser<PointType> parser;
	SemanticRoom<PointType> aRoom = SemanticRoomXMLParser<PointType>::loadRoomFromXML(xml_file_name,true);
	//	    aRoom.resetRoomTransform();
	//	    CloudPtr room_complete_cloud = aRoom.getCompleteRoomCloud();

	//	    // update summary xml
	//	    m_SummaryParser.createSummaryXML<PointType>();

	//	    // update list of rooms & metarooms
	//	    m_SummaryParser.refresh();

	ROS_INFO_STREAM("Summary XML created. Looking for metaroom for room with id: "<<aRoom.getRoomStringId());
	return "";
//	    boost::shared_ptr<MetaRoom<PointType> > metaroom;
//	    bool found = false;

//	    std::string matchingMetaroomXML = "";
//	    // first check if the matching metaroom has already been loaded
//	    for (size_t i=0; i<m_vLoadedMetarooms.size(); i++)
//	    {
//	        // compare by waypoint, if set
//	        if (aRoom.getRoomStringId() != "")
//	        {
//	            if (aRoom.getRoomStringId() == m_vLoadedMetarooms[i]->m_sMetaroomStringId)
//	            {
//	                metaroom = m_vLoadedMetarooms[i];
//	                found = true;
//	                break;
//	            }
//	        } else {
//	            // if not set, compare by centroid
//	            double centroidDistance = pcl::distances::l2(m_vLoadedMetarooms[i]->getCentroid(),aRoom.getCentroid());
//	            if (! (centroidDistance < ROOM_CENTROID_DISTANCE) )
//	            {
//	                continue;
//	            } else {
//	                ROS_INFO_STREAM("Matching metaroom already loaded.");
//	                metaroom = m_vLoadedMetarooms[i];
//	                found = true;
//	                break;
//	            }
//	        }
//	    }

//	    // if not loaded already, look through already saved metarooms
//	    if (!found)
//	    {
//	        std::vector<Entities> allMetarooms = m_SummaryParser.getMetaRooms();
//	        ROS_INFO_STREAM("Loaded "<<allMetarooms.size()<<" metarooms.");
//	        for (size_t i=0; i<allMetarooms.size();i++)
//	        {
//	            // compare by waypoint, if set
//	            if (aRoom.getRoomStringId() != "")
//	            {
//	                if (aRoom.getRoomStringId() == allMetarooms[i].stringId)
//	                {
//	                    matchingMetaroomXML = allMetarooms[i].roomXmlFile;
//	                    break;
//	                }
//	            } else {
//	                // if not set, compare by centroid
//	                double centroidDistance = pcl::distances::l2(allMetarooms[i].centroid,aRoom.getCentroid());
//	                if (! (centroidDistance < ROOM_CENTROID_DISTANCE) )
//	                {
//	                    continue;
//	                } else {
//	                    matchingMetaroomXML = allMetarooms[i].roomXmlFile;
//	                    break;
//	                }
//	            }
//	        }

//	        if (matchingMetaroomXML == "")
//	        {
//	            ROS_INFO_STREAM("No matching metaroom found. Create new metaroom.");
//	            metaroom =  boost::shared_ptr<MetaRoom<PointType> >(new MetaRoom<PointType>());
//	            metaroom->m_sMetaroomStringId = aRoom.getRoomStringId(); // set waypoint, to figure out where to save

//	        } else {
//	            ROS_INFO_STREAM("Matching metaroom found. XML file: "<<matchingMetaroomXML);
//	            metaroom =  MetaRoomXMLParser<PointType>::loadMetaRoomFromXML(matchingMetaroomXML,true);
//	            found = true;
//	        }
//	    } else {
//	        MetaRoomXMLParser<PointType> meta_parser;
//	        matchingMetaroomXML= meta_parser.findMetaRoomLocation(metaroom.get()).toStdString();
//	        matchingMetaroomXML+="/metaroom.xml";
//	    }

//	    if (!found)
//	    {
//	        ROS_INFO_STREAM("Initializing metaroom.");
//	        metaroom->setUpdateMetaroom(m_bUpdateMetaroom);
//	        metaroom->setSaveIntermediateSteps(m_bSaveIntermediateData);
//	        // also update the metaroom here
//	        auto updateIteration = metaroom->updateMetaRoom(aRoom,"",true);
//	        // save metaroom
//	        ROS_INFO_STREAM("Saving metaroom.");
//	        MetaRoomXMLParser<PointType> meta_parser;
//	        meta_parser.saveMetaRoomAsXML(*metaroom);
//	    }

//		CloudPtr difference(new Cloud());
//	    if(segmentationtype == 1){
//	        std::string previousObservationXml;
//	        passwd* pw = getpwuid(getuid());
//	        std::string dataPath(pw->pw_dir);
//	        dataPath+="/.semanticMap";

//	        std::vector<std::string> matchingObservations = semantic_map_load_utilties::getSweepXmlsForTopologicalWaypoint<PointType>(dataPath, aRoom.getRoomStringId());
//	        if (matchingObservations.size() == 0) // no observations -> first observation
//	        {
//	            ROS_INFO_STREAM("No observations for this waypoint saved on the disk "+aRoom.getRoomStringId()+" cannot compare with previous observation.");
//	            std_msgs::String msg;
//	            msg.data = "finished_processing_observation";
//	            m_PublisherStatus.publish(msg);
//	            return;
//	        }
//	        if (matchingObservations.size() == 1) // first observation at this waypoint
//	        {
//	            ROS_INFO_STREAM("No dynamic clusters.");
//	            std_msgs::String msg;
//	            msg.data = "finished_processing_observation";
//	            m_PublisherStatus.publish(msg);
//	            return;
//	        }

//	        int stopind = 0;

//	        for(unsigned int i = 0; i < matchingObservations.size(); i++){
//	            if(matchingObservations[i].compare(xml_file_name) == 0){
//	                break;
//	            }else{
//	                stopind = i;//latest = matchingObservations[i];
//	            }
//	        }

//	        if(stopind == 0) // first observation at this waypoint
//	        {
//	            ROS_INFO_STREAM("No dynamic clusters.");
//	            std_msgs::String msg;
//	            msg.data = "finished_processing_observation";
//	            m_PublisherStatus.publish(msg);
//	            return;
//	        }
//	        previousObservationXml = matchingObservations[stopind];
//	        printf("prev: %s\n",previousObservationXml.c_str());

//			quasimodo_msgs::metaroom_pair sm;
//			sm.request.background = previousObservationXml;
//			sm.request.foreground = xml_file_name;

//			if (m_segmentation_client.call(sm)){
//				CloudPtr dynamiccloud(new Cloud());
//				std::vector<cv::Mat> dynamicmasks;
//				for(unsigned int i = 0; i < sm.response.dynamicmasks.size(); i++){
//					cv_bridge::CvImagePtr			img_ptr;
//					try{							img_ptr = cv_bridge::toCvCopy(sm.response.dynamicmasks[i], sensor_msgs::image_encodings::MONO8);}
//					catch (cv_bridge::Exception& e){ROS_ERROR("cv_bridge exception: %s", e.what());}
//					cv::Mat mask = img_ptr->image;
//					dynamicmasks.push_back(mask);

//					CloudPtr currentcloud = aRoom.getIntermediateClouds()[i];
//					CloudPtr currentcloudTMP(new Cloud());
//					aRoom.getIntermediateCloudTransformsRegistered()[i];
//					//pcl::transformPointCloud (*currentcloud, *currentcloudTMP, aRoom.m_vIntermediateRoomCloudTransformsRegistered[i]);

//					pcl_ros::transformPointCloud(*currentcloud, *currentcloudTMP,aRoom.getIntermediateCloudTransformsRegistered()[i]);
//					unsigned char * maskdata = mask.data;
//					for(unsigned int j = 0; j < currentcloudTMP->points.size(); j++){
//						if(maskdata[j] > 0){
//							difference->points.push_back(currentcloudTMP->points[j]);
//						}
//					}
//				}

//	//			std::vector<CloudPtr> vClusters = MetaRoom<PointType>::clusterPointCloud(dynamiccloud,0.03,100,1000000);
//	//			ROS_INFO_STREAM("Clustered differences. "<<vClusters.size()<<" different clusters.");
//			}else{
//				ROS_ERROR("Failed to call service segment_model");
//				std_msgs::String msg;
//				msg.data = "error_processing_observation";
//				m_PublisherStatus.publish(msg);
//				return;
//			}
}



std::string outtopic = "/some/topic";
ros::Publisher out_pub;
void processMetaroom(std::string path){
    printf("processing: %s\n",path.c_str());

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
            //viewcams.push_back(roomData.vIntermediateRoomCloudCamParams.front());

            cv::namedWindow("rgbimage",     cv::WINDOW_AUTOSIZE);
            cv::imshow(		"rgbimage",     rgb);
            cv::namedWindow("depthimage",	cv::WINDOW_AUTOSIZE);
            cv::imshow(		"depthimage",	depth);
            cv::waitKey(30);

        }
    }

	reglib::Model * sweep = quasimodo_brain::load_metaroom_model(path);

    std::vector<Eigen::Matrix4d> unrefined;
    std::vector<double> times;
    for(unsigned int i = 0; i < viewrgbs.size(); i++){
        geometry_msgs::TransformStamped msg;
        tf::transformStampedTFToMsg(viewtfs[i], msg);
        long sec = msg.header.stamp.sec;
        long nsec = msg.header.stamp.nsec;
        double time = double(sec)+1e-9*double(nsec);

        Eigen::Quaterniond q;
        q.w() = msg.transform.rotation.w;
        q.x() = msg.transform.rotation.x;
        q.y() = msg.transform.rotation.y;
        q.z() = msg.transform.rotation.z;

        Eigen::Affine3d a (q);
        a(0,3) = msg.transform.translation.x;
        a(1,3) = msg.transform.translation.y;
        a(2,3) = msg.transform.translation.z;


        unrefined.push_back(a.matrix());
        times.push_back(time);

    }

//    xmlWriter->writeStartElement("Transform");
//    xmlWriter->writeStartElement("Translation");
//    xmlWriter->writeStartElement("x");
//    xmlWriter->writeCharacters(QString::number(msg.transform.translation.x));
//    xmlWriter->writeEndElement();
//    xmlWriter->writeStartElement("y");
//    xmlWriter->writeCharacters(QString::number(msg.transform.translation.y));
//    xmlWriter->writeEndElement();
//    xmlWriter->writeStartElement("z");
//    xmlWriter->writeCharacters(QString::number(msg.transform.translation.z));
//    xmlWriter->writeEndElement();
//    xmlWriter->writeEndElement(); // Translation
//    xmlWriter->writeStartElement("Rotation");
//    xmlWriter->writeStartElement("w");
//    xmlWriter->writeCharacters(QString::number(msg.transform.rotation.w));
//    xmlWriter->writeEndElement();
//    xmlWriter->writeStartElement("x");
//    xmlWriter->writeCharacters(QString::number(msg.transform.rotation.x));
//    xmlWriter->writeEndElement();
//    xmlWriter->writeStartElement("y");
//    xmlWriter->writeCharacters(QString::number(msg.transform.rotation.y));
//    xmlWriter->writeEndElement();
//    xmlWriter->writeStartElement("z");
//    xmlWriter->writeCharacters(QString::number(msg.transform.rotation.z));

    std_msgs::String msg;
    msg.data = path;
    out_pub.publish(msg);
    ros::spinOnce();
}

void chatterCallback(const std_msgs::String::ConstPtr& msg){processMetaroom(msg->data);}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "metaroom_additional_view_processing");
    ros::NodeHandle n;


    int inputstate = 0;
    for(int i = 1; i < argc;i++){
        printf("input: %s\n",argv[i]);
        if(std::string(argv[i]).compare("-file") == 0){inputstate = 2;}
        else if(std::string(argv[i]).compare("-intopic") == 0){inputstate = 0;}
        else if(std::string(argv[i]).compare("-outtopic") == 0){inputstate = 1;}
        else if(inputstate == 0){
            out_pub = n.advertise<std_msgs::String>(outtopic, 1000);
            ros::Subscriber sub = n.subscribe(std::string(argv[i]), 1000, chatterCallback);
            ros::spin();
        }else if(inputstate == 1){
            outtopic = std::string(argv[i]);
        }else{

            out_pub = n.advertise<std_msgs::String>(outtopic, 1000);
            processMetaroom(std::string(argv[i]));
        }
    }

//    std::string listentopic = "/some/stupid/topic";
//    if(argc == 2){listentopic = std::string(argv[1]);}
//    else{printf("robot listener: incorrect number of args. Should take a single arguement with a topicname!\n");}



//	for(int ar = 1; ar < argc; ar++){
//		std::string overall_folder = std::string(argv[ar]);
//		printf("overall folder: %s\n",overall_folder.c_str());

//		std::vector<std::string> sweep_xmls = semantic_map_load_utilties::getSweepXmls<pcl::PointXYZRGB>(overall_folder);
//		for (int i = 0; i < sweep_xmls.size(); i++){//auto sweep_xml : sweep_xmls) {
//			printf("current xml: %s\n",sweep_xmls[i].c_str());
//			fixXml(sweep_xmls[i]);
//			if(i > 1){
//				segmentWithAdditionalViewsXml(sweep_xmls[i-1],sweep_xmls[i]);
//			}
//		}
//	}

//	printf("test\n");
//	exit(0);
//	// Set up ROS.
//	ros::init(argc, argv, "Semantic_map_node");
//	ros::NodeHandle n;

//	ros::NodeHandle aRosNode("~");
//	printf("test\n");
//	//SemanticMapNode<pcl::PointXYZRGB> aSemanticMapNode(aRosNode);
//	SemanticMapNode<pcl::PointXYZRGB> aSemanticMapNode(n);
//	aSemanticMapNode.processRoomObservation(argv[1]);
//	exit(0);
//	printf("test\n");
//	ros::Rate loop_rate(10);
//	while (ros::ok())
//	{
//		ros::spinOnce();
//		loop_rate.sleep();
//	}
}
