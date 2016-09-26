#include "semantic_map/semantic_map_node.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Semantic_map_node");
    ros::NodeHandle n;

    ros::NodeHandle aRosNode("~");
	SemanticMapNode<pcl::PointXYZRGB> aSemanticMapNode(n);
    aSemanticMapNode.processRoomObservation(argv[1]);

    ros::Rate loop_rate(10);
    while (ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep();
    }
}
