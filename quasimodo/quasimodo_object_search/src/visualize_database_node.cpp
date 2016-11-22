#include <ros/ros.h>
#include <mongodb_store/message_store.h>
#include <quasimodo_msgs/fused_world_state_object.h>
#include <std_msgs/Empty.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;
using PointT = pcl::PointXYZRGB;
using CloudT = pcl::PointCloud<PointT>;

class visualization_server {
public:

    ros::NodeHandle n;
    mongodb_store::MessageStoreProxy db_client;
    ros::Subscriber sub;
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    size_t counter;

    visualization_server() : db_client(n, "quasimodo", "world_state"), viewer(new pcl::visualization::PCLVisualizer("3D Viewer")), counter(0)
    {
        sub = n.subscribe("/model/added_to_db", 1, &visualization_server::callback, this);

        viewer->setBackgroundColor(1, 1, 1);

        viewer->initCameraParameters();

    }

    void callback(const std_msgs::Empty& empty_msg)
    {
        std::vector<boost::shared_ptr<quasimodo_msgs::fused_world_state_object> > messages;
        db_client.query(messages);

        viewer->removeAllPointClouds();
        size_t start_counter = counter;
        for (boost::shared_ptr<quasimodo_msgs::fused_world_state_object>& msg : messages) {
            if (!msg->removed_at.empty()) {
                continue;
            }
            if (msg->inserted_at.empty()) {
                cout << "No insertion for msg with ID: " << msg->object_id << endl;
            }
            cout << "Adding " << counter << ":th point cloud..." << endl;
            CloudT::Ptr cloud(new CloudT);
            pcl::fromROSMsg(msg->surfel_cloud, *cloud);

            float offset = 1.0f*float(counter - start_counter);
            for (PointT& p : cloud->points) {
                p.x += offset;
            }

            pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
            string cloud_name = string("cloud") + std::to_string(counter);
            viewer->addPointCloud<PointT>(cloud, rgb, cloud_name);
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloud_name);
            ++counter;
        }
    }

    void run()
    {
        while (!viewer->wasStopped()) {
            ros::spinOnce();
            viewer->spinOnce(100);
        }
    }

};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "visualize_database_node");

    visualization_server vs;
    vs.run();

    return 0;
}
