#include "semantic_map/semantic_map_node.h"

int main(int argc, char **argv)
{
    printf("test\n");
    // Set up ROS.
    ros::init(argc, argv, "Semantic_map_node");
    ros::NodeHandle n;

    ros::NodeHandle aRosNode("~");
    printf("test\n");
	//SemanticMapNode<pcl::PointXYZRGB> aSemanticMapNode(aRosNode);
	SemanticMapNode<pcl::PointXYZRGB> aSemanticMapNode(n);
    aSemanticMapNode.processRoomObservation(argv[1]);
    exit(0);
    printf("test\n");
    ros::Rate loop_rate(10);
    while (ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep();
    }
}
