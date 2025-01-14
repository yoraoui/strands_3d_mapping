#include "ros/ros.h"
#include <exception>
#include <dynamic_object_retrieval/dynamic_retrieval.h>
#include <dynamic_object_retrieval/visualize.h>
#include <object_3d_benchmark/benchmark_retrieval.h>
#include <object_3d_benchmark/benchmark_visualization.h>
#include "quasimodo_msgs/query_cloud.h"
#include "quasimodo_msgs/visualize_query.h"
#include "quasimodo_msgs/retrieval_query_result.h"
#include "quasimodo_msgs/string_array.h"
#include <pcl_ros/point_cloud.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <tf_conversions/tf_eigen.h>
#include <eigen_conversions/eigen_msg.h>

// for logging to soma in mongodb
#include <soma_msgs/SOMAObject.h>
#include <soma_manager/SOMAInsertObjs.h>
#include <mongodb_store/MongoInsert.h>

using namespace std;

using PointT = pcl::PointXYZRGB;
using CloudT = pcl::PointCloud<PointT>;
using HistT = pcl::Histogram<250>;

string random_string( size_t length )
{
    auto randchar = []() -> char
    {
        const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
        const size_t max_index = (sizeof(charset) - 1);
        return charset[ rand() % max_index ];
    };
    string str(length,0);
    generate_n(str.begin(), length, randchar);
    return str;
}

class logging_server {
public:
    ros::NodeHandle n;
    ros::Subscriber sub;
    string topic_input;

    logging_server(const std::string& name)
    {
        ros::NodeHandle pn("~");
        pn.param<std::string>("topic_input", topic_input, std::string("/retrieval_result"));
        sub = n.subscribe(topic_input, 1, &logging_server::callback, this);
    }

    void callback(const quasimodo_msgs::retrieval_query_result& res)
    {
        soma_manager::SOMAInsertObjs::Request soma_req;

        string search_id = random_string(6);

        size_t N = res.result.retrieved_clouds.size();
        soma_req.objects.resize(N);
        for (size_t i = 0; i < N; ++i) {
            //cout << "Retrieved image paths length: " << res.result.retrieved_image_paths.size() << endl;
            boost::filesystem::path sweep_xml = boost::filesystem::path(res.result.retrieved_image_paths[i].strings[0]).parent_path() / "room.xml";
            //cout << "Sweep xml: " << sweep_xml.string() << endl;
            boost::posix_time::time_duration diff;
            if (sweep_xml.parent_path().parent_path().stem() == "temp") {
                soma_req.objects[i].cloud = res.result.retrieved_clouds[i];
                //diff = boost::posix_time::time_duration(10, 10, 10); // just placeholder for now, should get real timestamp and transform from mongodb object
                boost::posix_time::ptime start_time = boost::posix_time::second_clock::local_time();
                boost::posix_time::ptime time_t_epoch(boost::gregorian::date(1970,1,1));
                diff = start_time - time_t_epoch;
                soma_req.objects[i].type = "quasimodo_db_result";
            }
            else {
                auto data = SimpleXMLParser<PointT>::loadRoomFromXML(sweep_xml.string(), vector<string>({"RoomIntermediateCloud"}), false, false);
                boost::posix_time::ptime start_time = data.roomLogStartTime;
                boost::posix_time::ptime time_t_epoch(boost::gregorian::date(1970,1,1));
                diff = start_time - time_t_epoch;
                Eigen::Affine3d AT;
                cout << "Intermediate room cloud transforms length: " << data.vIntermediateRoomCloudTransforms.size() << endl;
                tf::transformTFToEigen(data.vIntermediateRoomCloudTransforms[0], AT);
                pcl_ros::transformPointCloud(AT.matrix().cast<float>(), res.result.retrieved_clouds[i], soma_req.objects[i].cloud);
                soma_req.objects[i].type = "quasimodo_metaroom_result";
            }
            //soma_req.objects[i].cloud = res.result.retrieved_clouds[i];
            cout << "Retrieved initial poses length: " << res.result.retrieved_initial_poses.size() << endl;
            soma_req.objects[i].pose = res.result.retrieved_initial_poses[i].poses[0];
            soma_req.objects[i].logtimestamp = diff.total_seconds(); //   ros::Time::now().sec;
            soma_req.objects[i].id = res.result.retrieved_image_paths[i].strings[0];
            soma_req.objects[i].config = "quasimodo_retrieval_result";
            soma_req.objects[i].metadata = search_id;
        }

        std::sort(soma_req.objects.begin(), soma_req.objects.end(), [](const soma_msgs::SOMAObject& o1, const soma_msgs::SOMAObject& o2) {
            return o1.logtimestamp < o2.logtimestamp;
        });

        for (size_t i = 0; i < N; ++i) {
            soma_req.objects[i].timestep = i;
        }

        /*
        soma_req.objects[N].cloud = res.query.cloud;
        //soma_req.objects[N].pose = res.query.room_transform;
        soma_req.objects[N].logtimestamp = ros::Time::now().sec;
        soma_req.objects[N].id = "retrieval_query";
        */

        ros::ServiceClient soma_service = n.serviceClient<soma_manager::SOMAInsertObjs>("/soma/insert_objects");
        soma_manager::SOMAInsertObjs::Response soma_resp;
        try {
            if (soma_service.call(soma_req, soma_resp)) {
                cout << "Quasimodo logging server: successfully inserted queries!" << endl;
            }
            else {
                cout << "Quasimodo logging server: Could not connect to SOMA!" << endl;
            }
        }
        catch (exception& e) {
            cout << "Quasimodo logging server: Soma service called failed with: " << e.what() << endl;
        }

        /*
        mongodb_store::MongoInsert::Request db_req;
        //db_req.

        ros::ServiceClient db_service = n.serviceClient<mongodb_store::MongoInsert>("/soma/insert_objects");
        mongodb_store::MongoInsert::Response db_resp;
        if (db_service.call(db_req, db_resp)) {
            cout << "Successfully inserted queries!" << endl;
        }
        else {
            cout << "Could not connect to SOMA!" << endl;
        }
        */
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "quasimodo_logging_server");

    logging_server ls(ros::this_node::getName());

    ros::spin();

    return 0;
}
