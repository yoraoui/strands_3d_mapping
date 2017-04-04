#include "ros/ros.h"
#include "std_msgs/String.h"
#include <string.h>

#include <cv_bridge/cv_bridge.h>

#include "eigen_conversions/eigen_msg.h"
#include "tf_conversions/tf_eigen.h"

#include "metaroom_xml_parser/simple_xml_parser.h"
#include "metaroom_xml_parser/simple_summary_parser.h"
#include <metaroom_xml_parser/load_utilities.h>

#include <observation_registration_services/ObjectAdditionalViewRegistrationService.h>
#include <observation_registration_services/AdditionalViewRegistrationService.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/extract_indices.h>
#include <pcl_ros/transforms.h>

#include <image_geometry/pinhole_camera_model.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>

#include <observation_registration_services/ObjectAdditionalViewRegistrationService.h>
#include <observation_registration_services/ObservationRegistrationService.h>



#include "object_manager/dynamic_object.h"
#include "object_manager/dynamic_object_xml_parser.h"
#include "object_manager/dynamic_object_utilities.h"
#include "object_manager/dynamic_object_mongodb_interface.h"
#include <object_manager_msgs/DynamicObjectTracks.h>
#include <object_manager_msgs/DynamicObjectTrackingData.h>
#include <object_manager_msgs/DynamicObjectComputeMaskService.h>
#include <object_manager_msgs/DynamicObjectsService.h>
#include <object_manager_msgs/GetDynamicObjectService.h>
#include <object_manager_msgs/ProcessDynamicObjectService.h>

#include <semantic_map_msgs/RoomObservation.h>
#include <semantic_map/room_xml_parser.h>
#include <semantic_map/ndt_registration.h>
#include <semantic_map/metaroom.h>

#include <iostream>
#include <tuple>
#include <string>

// ******************** TYPEDEFS ***************************** //
using namespace std;
using namespace semantic_map_load_utilties;
typedef pcl::PointXYZRGB PointType;
typedef pcl::PointCloud<PointType> Cloud;
typedef typename Cloud::Ptr CloudPtr;
typedef pcl::search::KdTree<PointType> Tree;
typedef semantic_map_load_utilties::DynamicObjectData<PointType> ObjectData;

// ******************** PARAMETERS ***************************** //
bool VISUALIZE = false;
double CUTOFF_DISTANCE = 3.0;
int MIN_CLUSTER_SIZE = 250;
int MAX_CLUSTER_SIZE = 25000;
double MASK_NEIGHBOR_DISTANCE =  0.001;
bool REBUILD_COMPLETE_CLOUD = false;
string SAVE_DATA_FOLDER = "mr_clusters";

// ******************** CLUSTER DATATYPE ***************************** //
struct ClusterData{
    CloudPtr cluster;
    vector<vector<int> > masks_indices;
    vector<cv::Mat> masks;
    vector<int> int_cloud_id;
};

// ******************** HELPER METHODS ***************************** //
void process_waypoint_sweeps(const string& sweep_1,
                             const string& sweep_2,
                             ros::ServiceClient& client,
                             pcl::visualization::PCLVisualizer* pg);
CloudPtr rebuildCloud(std::vector<CloudPtr> intermediate_clouds,
                      std::vector<tf::StampedTransform> intermediate_transforms);
CloudPtr filterPointCloud(CloudPtr input,
                          const double& distance);
CloudPtr compute_mask_per_view_from_object(CloudPtr view,
                                           CloudPtr object,
                                           cv::Mat& mask_image,
                                           vector<int>& mask_indices,
                                           const double& neighbor_distance);
void save_cluster_data(const ClusterData& cluster_data,
                       const int& cluster_id,
                       const string& sweep_xml,
                       const string& save_data_folder);


// ******************** CODE ***************************** //
int main(int argc, char** argv){

    if (argc < 2){
        cout<<"USAGE: ./mr_segment_dynamic_clusters PATH_TO_SWEEP_FOLDER"<<endl;
        exit(0);
    }

    // initialize ros node
    ros::init(argc, argv, "mr_segment_dynamic_clusters");
    ros::NodeHandle n;
    ros::ServiceClient registration_client = n.serviceClient<observation_registration_services::ObservationRegistrationService>("observation_registration_server");
    pcl::visualization::PCLVisualizer* pg;
    if (VISUALIZE){
        pg = new pcl::visualization::PCLVisualizer (argc, argv, "mr_segment_dynamic_clusters");
        pg->addCoordinateSystem(1.0);
    }

    string sweep_folder = argv[1];
    vector<string> sweep_xmls = semantic_map_load_utilties::getSweepXmls<PointType>(argv[1]);
    cout<<"Found "<<sweep_xmls.size()<<" sweeps. "<<endl;
    map<string, vector<string> > waypoint_to_sweepxml_map;
    for (const string xml : sweep_xmls){
        // find waypoint
        auto room_data = SimpleXMLParser<PointType>::loadRoomFromXML(xml, {}, false, false);
        string waypoint = room_data.roomWaypointId;
        if (waypoint_to_sweepxml_map.find(waypoint) == waypoint_to_sweepxml_map.end()){
            waypoint_to_sweepxml_map[waypoint] = vector<string>();
        }
        waypoint_to_sweepxml_map[waypoint].push_back(xml);
    }

    for (auto it=waypoint_to_sweepxml_map.begin(); it != waypoint_to_sweepxml_map.end(); ++it){
        std::sort(it->second.begin(), it->second.end()); // simple sort
        cout<<it->first<<" ---> observations: "<<it->second.size()<<endl;
        if (it->first!="WayPoint16"){
            continue;
        }
        for (int i=1; i<it->second.size(); ++i){
            process_waypoint_sweeps(it->second[i-1], it->second[i], registration_client, pg);
        }
    }

    return 1;
}

void process_waypoint_sweeps(const string& sweep_1,
                             const string& sweep_2,
                             ros::ServiceClient& client,
                             pcl::visualization::PCLVisualizer* pg){

    ROS_INFO_STREAM("------------------------ processing :------------------------\n"<<sweep_1<<"\n"<<sweep_2<<"\n------------------------------------------------------------------------------------------------------------------");
    // ******************** REGISTER SWEEPS ***************************** //
    bool registration_successfull = true;
    observation_registration_services::ObservationRegistrationService srv;
    srv.request.source_observation_xml = sweep_1;
    srv.request.target_observation_xml = sweep_2;
    tf::Transform registered_transform;

    if (client.call(srv))
    {
        ROS_INFO_STREAM("Registration done. Number of constraints "<<srv.response.total_correspondences);
        if (srv.response.total_correspondences <= 200){
            ROS_ERROR("Registration unsuccessful due to insufficient constraints. ABORTING.");
                return;
                registered_transform.setIdentity(); // -> and hope for the best
                registration_successfull = false;
        } else {
            tf::transformMsgToTF(srv.response.transform, registered_transform);
            registration_successfull = true;
        }
    }
    else
    {
        ROS_ERROR("Failed to call service observation_registration_server");
        return;
    }

    // ******************** LOAD SWEEP DATA ***************************** //
    SemanticRoomXMLParser<PointType> parser;
    auto obs1_data = parser.loadRoomFromXML(sweep_1);
    auto obs2_data = parser.loadRoomFromXML(sweep_2);

    // ******************** REBUILD COMPLETE CLOUDS ***************************** //
    CloudPtr obs1_cloud (new Cloud), obs2_cloud(new Cloud);

    if (REBUILD_COMPLETE_CLOUD){
        if ((obs1_data.getIntermediateCloudTransformsRegistered().size() == 0) || (obs2_data.getIntermediateCloudTransformsRegistered().size() == 0)){
            ROS_ERROR_STREAM("Could not find intermediate registered transforms. Aborting.");
            return;
        } else {
            ROS_INFO_STREAM("Rebuilding merged cloud using registered transforms");
            obs1_cloud = rebuildCloud(obs1_data.getIntermediateClouds(), obs1_data.getIntermediateCloudTransformsRegistered());
            obs2_cloud = rebuildCloud(obs2_data.getIntermediateClouds(), obs2_data.getIntermediateCloudTransformsRegistered());
            pcl_ros::transformPointCloud(*obs1_cloud, *obs1_cloud,obs1_data.getIntermediateCloudTransforms()[0]); // transform to global frame
            pcl_ros::transformPointCloud(*obs2_cloud, *obs2_cloud,obs2_data.getIntermediateCloudTransforms()[0]); // transform to global frame
        }
    } else {
        obs1_cloud = obs1_data.getCompleteRoomCloud();
        obs2_cloud = obs2_data.getCompleteRoomCloud();
        pcl_ros::transformPointCloud(*obs1_cloud, *obs1_cloud,obs1_data.getIntermediateCloudTransforms()[0]); // transform to global frame
        pcl_ros::transformPointCloud(*obs2_cloud, *obs2_cloud,obs2_data.getIntermediateCloudTransforms()[0]); // transform to global frame
    }

    // ******************** TRANSFORM CLOUD USING COMPUTED TRANSFORMATION ***************************** //
    CloudPtr obs1_registered_cloud(new Cloud);
    if (registration_successfull){
        pcl_ros::transformPointCloud(*obs1_cloud, *obs1_registered_cloud,registered_transform);
    } else {
        // ******************** COMPUTE TRANSFORMATION WITH NDT (IF UNSUCCESSFUL THE FIRST TIME) ***************************** //
        ROS_INFO_STREAM("Registering with NDT");
        Eigen::Matrix4f ndt_transform;
        obs1_registered_cloud = NdtRegistration<PointType>::registerClouds(obs1_cloud, obs2_cloud,ndt_transform);
    }

    // ******************** DOWNSAMPLE ***************************** //
    CloudPtr obs1_downsampled = MetaRoom<PointType>::downsampleCloud(obs1_registered_cloud->makeShared());
    CloudPtr obs2_downsampled = MetaRoom<PointType>::downsampleCloud(obs2_cloud->makeShared());

    // ******************** VISUALIZE ***************************** //
    if (VISUALIZE){
        pcl::visualization::PointCloudColorHandlerCustom<PointType> obs2_handler_original (obs2_downsampled, 0, 255, 0);
        pcl::visualization::PointCloudColorHandlerCustom<PointType> obs1_handler_registered (obs1_downsampled, 255, 0, 0);

        pg->addPointCloud (obs1_downsampled,obs1_handler_registered,"obs1_cloud_registered");
        pg->addPointCloud (obs2_downsampled,obs2_handler_original,"obs2_cloud_2");
        pg->spin();
        pg->removeAllPointClouds();
    }


    // ******************** DISTANCE FILTERING ***************************** //
    obs1_downsampled = filterPointCloud(obs1_downsampled, CUTOFF_DISTANCE); // distance filtering
    obs2_downsampled = filterPointCloud(obs2_downsampled, CUTOFF_DISTANCE); // distance filtering

    // ******************** COMPUTE DIFFERENCE ***************************** //
    CloudPtr differenceRoomToPrevRoom(new Cloud);
    CloudPtr differencePrevRoomToRoom(new Cloud);

    // compute the differences
    pcl::SegmentDifferences<PointType> segment;
    segment.setInputCloud(obs2_downsampled);
    segment.setTargetCloud(obs1_downsampled);
    segment.setDistanceThreshold(0.001);
    typename Tree::Ptr tree (new pcl::search::KdTree<PointType>);
    tree->setInputCloud (obs1_downsampled);
    segment.setSearchMethod(tree);
    segment.segment(*differenceRoomToPrevRoom);

    segment.setInputCloud(obs1_downsampled);
    segment.setTargetCloud(obs2_downsampled);
    tree->setInputCloud(obs2_downsampled);

    segment.segment(*differencePrevRoomToRoom);

    CloudPtr toBeAdded(new Cloud());
    CloudPtr toBeRemoved(new Cloud());

    OcclusionChecker<PointType> occlusionChecker;
    occlusionChecker.setSensorOrigin(obs2_data.getIntermediateCloudTransforms()[0].getOrigin()); // since it's already transformed in the metaroom frame of ref
    auto occlusions = occlusionChecker.checkOcclusions(differenceRoomToPrevRoom,differencePrevRoomToRoom, 720 );
    CloudPtr difference(new Cloud());
    *difference = *occlusions.toBeRemoved;

    if (difference->points.size() > obs2_downsampled->points.size() * 0.1){
        // probably registration failure
        ROS_ERROR_STREAM("Difference has too many points, probably registration failure. Aborting. "<<difference->points.size());
        return;
    }

    // ******************** COMPUTE CLUSTERS ***************************** //
    std::vector<CloudPtr> vClusters = MetaRoom<PointType>::clusterPointCloud(difference,0.03,MIN_CLUSTER_SIZE,MAX_CLUSTER_SIZE);
    ROS_INFO_STREAM("Clustered differences. "<<vClusters.size()<<" different clusters.");
    MetaRoom<PointType>::filterClustersBasedOnDistance(obs2_data.getIntermediateCloudTransforms()[0].getOrigin(), vClusters,CUTOFF_DISTANCE);
    ROS_INFO_STREAM(vClusters.size()<<" different clusters after max distance filtering.");
    ROS_INFO_STREAM("Clustered differences. "<<vClusters.size()<<" different clusters.");


    // ******************** DISCARD PLANAR CLUSTERS ***************************** //
    std::vector<CloudPtr> final_clusters;
    for (size_t i=0; i<vClusters.size(); i++)
    {
        pcl::SACSegmentation<PointType> seg;
        pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);

        seg.setOptimizeCoefficients (true);
        seg.setModelType (pcl::SACMODEL_PLANE);
        seg.setMethodType (pcl::SAC_RANSAC);
        seg.setMaxIterations (100);
        seg.setDistanceThreshold (0.02);

        seg.setInputCloud (vClusters[i]);
        seg.segment (*inliers, *coefficients);
        if (inliers->indices.size () > 0.9 * vClusters[i]->points.size())
        {
            ROS_INFO_STREAM("Discarding planar dynamic cluster");
        } else {
            final_clusters.push_back(vClusters[i]);
        }
    }

    // ******************** VISUALIZE ***************************** //
    if (VISUALIZE){
        pcl::visualization::PointCloudColorHandlerCustom<PointType> obs2_handler_original (obs2_cloud, 0, 255, 0);
        pg->addPointCloud (obs2_downsampled,obs2_handler_original,"obs2_cloud_2");
        for (size_t i=0; i<final_clusters.size(); ++i ){
            stringstream ss; ss<<"cluster_"<<i;
            pcl::visualization::PointCloudColorHandlerCustom<PointType> cluster_handler (final_clusters[i], rand()%255, rand()%255, rand()%255);
            pg->addPointCloud (final_clusters[i],cluster_handler,ss.str());
        }
        pg->spin();
        pg->removeAllPointClouds();
    }

    // ******************** FIND INTERMEDIATE CLOUDS && COMPUTE MASKS ***************************** //
    for (size_t i=0; i<final_clusters.size(); ++i ){

        boost::shared_ptr<pcl::PointCloud<PointType>> diff(new pcl::PointCloud<PointType>());
        pcl::SegmentDifferences<PointType> segment_object;
        segment_object.setDistanceThreshold(0.001);
        ClusterData cluster_data;
        cluster_data.cluster = final_clusters[i];

        for (size_t j=0; j<obs2_data.getIntermediateClouds().size(); ++j ){
            std::cout << '\r'
                      <<"Processing cluster [" << std::setw(2) << i+1 << "/" << final_clusters.size()
                      <<"] Looking at int. cloud ["<< std::setw(2) << j+1 << "/"<<obs2_data.getIntermediateClouds().size() << "]"<<std::flush;

            CloudPtr int_cloud(new Cloud);
            CloudPtr cluster (new Cloud);
            // transform so that they are in the same frame of ref
            if (obs2_data.getIntermediateCloudTransformsRegistered().size() == 0){
                pcl_ros::transformPointCloud(*obs2_data.getIntermediateClouds()[j], *int_cloud,obs2_data.getIntermediateCloudTransforms()[j]);
                *cluster = *final_clusters[i];
            } else {
                pcl_ros::transformPointCloud(*obs2_data.getIntermediateClouds()[j], *int_cloud,obs2_data.getIntermediateCloudTransformsRegistered()[j]);
                pcl_ros::transformPointCloud(*final_clusters[i], *cluster,obs2_data.getIntermediateCloudTransforms()[0].inverse());
            }

            diff->clear();
            segment_object.setInputCloud(int_cloud);
            segment_object.setTargetCloud(cluster);
            segment_object.segment(*diff);



            if (diff->points.size() != int_cloud->points.size()){
                CloudPtr remainder_object(new Cloud);
                segment_object.setInputCloud(int_cloud);
                segment_object.setTargetCloud(diff);
                segment_object.setDistanceThreshold(0.001);
                segment_object.segment(*remainder_object);
                if (!remainder_object->points.size()){
                    continue; // probably some error here
                }


                // get mask
                cv::Mat mask_image;
                vector<int> mask_indices;
                CloudPtr mask = compute_mask_per_view_from_object(int_cloud,
                                                                  remainder_object,
                                                                  mask_image,
                                                                  mask_indices,
                                                                  MASK_NEIGHBOR_DISTANCE);
                if (mask_indices.size() > MIN_CLUSTER_SIZE){
//                    if (VISUALIZE){
//                        cv::namedWindow("mask",CV_WINDOW_OPENGL);
//                        cv::imshow("mask",mask_image);
//                        cv::waitKey(50);
//                        pcl::visualization::PointCloudColorHandlerCustom<PointType> int_handler (int_cloud, 0, 255, 0);
//                        pcl::visualization::PointCloudColorHandlerCustom<PointType> cl_handler (cluster, 255, 0, 0);
//                        pg->addPointCloud (int_cloud,int_handler,"int_cloud");
//                        pg->addPointCloud (cluster,cl_handler,"cluster");
//                        pg->spin();
//                        pg->removeAllPointClouds();
//                    }
                    cluster_data.masks.push_back(mask_image);
                    cluster_data.masks_indices.push_back(mask_indices);
                    cluster_data.int_cloud_id.push_back(j);
                }
            } else {
                continue;

            }
        }

        if (cluster_data.masks.size()){
            save_cluster_data(cluster_data,
                              i,
                              sweep_2,
                              SAVE_DATA_FOLDER);
        }
    }
}

CloudPtr rebuildCloud(std::vector<CloudPtr> intermediate_clouds, std::vector<tf::StampedTransform> intermediate_transforms){
    CloudPtr mergedCloud(new Cloud);

    for (size_t i=0; i<intermediate_clouds.size(); i++)
    {
        Cloud transformed_cloud;
        pcl_ros::transformPointCloud(*intermediate_clouds[i], transformed_cloud,intermediate_transforms[i]);
        *mergedCloud+=transformed_cloud;
    }
    return mergedCloud;
}

CloudPtr filterPointCloud(CloudPtr input, const double& distance) // distance filtering from the centroid of the point cloud, remove outliers and nans
{
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*input, centroid);

    for (size_t i=0; i<input->points.size(); i++)
    {
        Eigen::Vector4f point(input->points[i].x,input->points[i].y,input->points[i].z,0.0);
        double dist_from_centroid = pcl::distances::l2(centroid,point);

        if (fabs(dist_from_centroid) > distance)
        {
            inliers->indices.push_back(i);
        }
    }

    // filter points based on indices
    pcl::ExtractIndices<PointType> extract;
    extract.setInputCloud (input);
    extract.setIndices (inliers);
    extract.setNegative (true);

    CloudPtr filtered_cloud(new Cloud);
    extract.filter (*filtered_cloud);
    filtered_cloud->header = input->header;

    *input = *filtered_cloud;

    return input;
}

CloudPtr compute_mask_per_view_from_object(CloudPtr view,
                                           CloudPtr object,
                                           cv::Mat& mask_image,
                                           vector<int>& mask_indices,
                                           const double& neighbor_distance){
    using namespace std;
    // mask image -> empty by default
    mask_image = cv::Mat::zeros(480, 640, CV_8UC3);
    CloudPtr mask(new Cloud);
    if (!object->points.size()){
        ROS_ERROR_STREAM("Could not find mask. The segmented object has 0 points.");
        return mask;
    }

    // compute mask
    // find indices in original point cloud
    std::vector<int> nn_indices (1);
    std::vector<float> nn_distances (1);
    typename pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType>);
//    tree->setInputCloud (object);
    tree->setInputCloud (view);

    // Iterate through the source data set
    for (int i = 0; i < static_cast<int> (object->points.size ()); ++i)
    {
        if (!isFinite (object->points[i]))
            continue;
        // Search for the closest point in the target data set (number of neighbors to find = 1)
        if (!tree->nearestKSearch (object->points[i], 1, nn_indices, nn_distances))
        {
            PCL_WARN ("No neighbor found for point %zu (%f %f %f)!\n", i, object->points[i].x, object->points[i].y, object->points[i].z);
            continue;
        }

        if (nn_distances[0] < neighbor_distance)
        {
            mask_indices.push_back (nn_indices[0]);
            mask->push_back(object->points[i]);
        }
    }

    // create mask image
    for (int index : mask_indices)
    {
        pcl::PointXYZRGB point = view->points[index];
        int y = index / mask_image.cols;
        int x = index % mask_image.cols;
        mask_image.at<cv::Vec3b>(y, x)[0] = point.b;
        mask_image.at<cv::Vec3b>(y, x)[1] = point.g;
        mask_image.at<cv::Vec3b>(y, x)[2] = point.r;
    }
    return mask;
}

void save_cluster_data(const ClusterData& cluster_data,
                       const int& cluster_id,
                       const string& sweep_xml,
                       const string& save_data_folder){
    int ind = sweep_xml.find_last_of('/');
    string sweep_folder = sweep_xml.substr(0, ind+1);
    string save_folder = sweep_folder + "/" + save_data_folder + "/";

    ROS_INFO_STREAM("Saving cluster data in "<<save_folder);
    if (!QDir(save_folder.c_str()).exists())
    {
        QDir().mkdir(save_folder.c_str());
    }

    // save PCD
    char buf [1024];
    sprintf(buf,"%s/dynamic_obj%10.10i.pcd",save_folder.c_str(),cluster_id);
    pcl::io::savePCDFileBinaryCompressed(std::string(buf),*(cluster_data.cluster));

    // cluster XML
    sprintf(buf,"%s/dynamic_obj%10.10i.xml",save_folder.c_str(),cluster_id);

    // compute centroid
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*(cluster_data.cluster), centroid);

    // save data
    QFile file(buf);
    if (file.exists()){file.remove();}
    if (!file.open(QIODevice::ReadWrite | QIODevice::Text)){
        std::cerr<<"Could not open file "<< buf <<" to save dynamic object as XML"<<std::endl;
        return;
    }
    QXmlStreamWriter* xmlWriter = new QXmlStreamWriter();
    xmlWriter->setDevice(&file);

    xmlWriter->writeStartDocument();
    xmlWriter->writeStartElement("Object");
    xmlWriter->writeAttribute("object_number", QString::number(cluster_id));
    xmlWriter->writeAttribute("classname", QString(""));
    xmlWriter->writeAttribute("instancename", QString(""));
    xmlWriter->writeAttribute("tags", QString(""));
    xmlWriter->writeStartElement("Mean");
    xmlWriter->writeAttribute("x", QString::number(centroid(0)));
    xmlWriter->writeAttribute("y", QString::number(centroid(1)));
    xmlWriter->writeAttribute("z", QString::number(centroid(2)));
    xmlWriter->writeEndElement();

    for(unsigned int j = 0; j < cluster_data.masks.size(); j++){
        char buf [1024];
        sprintf(buf,"%s/dynamicmask_%i_%i.png",save_folder.c_str(), cluster_id, cluster_data.int_cloud_id[j]);
        cv::imwrite(buf, cluster_data.masks[j] );

        sprintf(buf,"dynamicmask_%i_%i.png",cluster_id,cluster_data.int_cloud_id[j]);
        xmlWriter->writeStartElement("Mask");
        xmlWriter->writeAttribute("filename", QString(buf));
        xmlWriter->writeAttribute("image_number", QString::number(cluster_data.int_cloud_id[j]));
        xmlWriter->writeEndElement();
    }
    xmlWriter->writeEndElement();
    xmlWriter->writeEndDocument();
    delete xmlWriter;
}
