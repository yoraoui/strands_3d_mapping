#include "ModelStorage/ModelStorage.h"
#include "Util/Util.h"
#include "CameraOptimizer/CameraOptimizer.h"


void train_cam(reglib::Model * model){
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("viewer"));
    viewer->setBackgroundColor(1.0,1.0,1.0);
    viewer->removeAllPointClouds();
    viewer->spinOnce();

    CameraOptimizerGridXYZ * co = new CameraOptimizerGridXYZ(64,48,10,100);
    co->setVisualization(viewer,1);

    for(unsigned int i = 0; i < model->frames.size(); i++){
        printf("%i / %i\n",i+1,model->frames.size());
        for(unsigned int j = 0; j < model->frames.size(); j++){
            if(i == j){continue;}
            Eigen::Matrix4d m = model->relativeposes[j].inverse() * model->relativeposes[i];
            co->addTrainingData( model->frames[i], model->frames[j], m);//model->relativeposes[i] * model->relativeposes[j].inverse());
            co->normalize();

        }
        co->show(false);
        co->normalize();
    }

    co->save("current_CameraOptimizerGridXYZ.bin");
    //co->print();
    co->show(true);

	for(unsigned int i = 0; i < model->frames.size(); i++){
		for(unsigned int j = 0; j < model->frames.size(); j++){
			if(i == j){continue;}
			Eigen::Matrix4d m = model->relativeposes[j].inverse() * model->relativeposes[i];
			co->show( model->frames[i], model->frames[j], m);//model->relativeposes[i] * model->relativeposes[j].inverse());
		}
	}
}

void test_cam(reglib::Model * model){
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("viewer"));
    viewer->setBackgroundColor(1.0,1.0,1.0);
    viewer->removeAllPointClouds();
    viewer->spinOnce();

    CameraOptimizer * co = CameraOptimizer::load("current_CameraOptimizerGridXYZ.bin");
    co->setVisualization(viewer,1);
    co->show(true);

    for(unsigned int i = 0; i < model->frames.size(); i++){
        for(unsigned int j = 0; j < model->frames.size(); j++){
            if(i == j){continue;}
            Eigen::Matrix4d m = model->relativeposes[j].inverse() * model->relativeposes[i];
            co->show( model->frames[i], model->frames[j], m);//model->relativeposes[i] * model->relativeposes[j].inverse());
        }
    }
}

int main(int argc, char **argv){
    ModelStorageFile * storage = new ModelStorageFile("fb_model/");
    std::vector<std::string> str = storage->loadAllModels();
    for(unsigned int i = 0; i < str.size(); i++){
        //test_cam(storage->fetch(str[i]));
        train_cam(storage->fetch(str[i]));
    }
}
