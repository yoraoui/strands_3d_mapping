#include "modelupdater/ModelUpdaterBasicFuse.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include <emmintrin.h>

#include "registration/MassRegistration.h"

namespace reglib
{

//------------MODEL UPDATER-----------
ModelUpdaterBasicFuse::ModelUpdaterBasicFuse(Registration * registration_){
	registration = registration_;
	model = new reglib::Model();

    show_init_lvl = 0;//init show
    show_refine_lvl = 0;//refine show
    show_scoring = false;//fuse scoring show
}

ModelUpdaterBasicFuse::ModelUpdaterBasicFuse(Model * model_, Registration * registration_){
	registration = registration_;
	model = model_;

    show_init_lvl = 0;//init show
    show_refine_lvl = 0;//refine show
    show_scoring = false;//fuse scoring show
}

ModelUpdaterBasicFuse::~ModelUpdaterBasicFuse(){
	//printf("deleting ModelUpdaterBasicFuse\n");
}

FusionResults ModelUpdaterBasicFuse::registerModel(Model * model2, Eigen::Matrix4d guess, double uncertanity){
	if(model->points.size() > 0 && model2->points.size() > 0){
		registration->viewer	= viewer;
		int step1 = std::max(int(model->frames.size())/1,1);
		int step2 = std::max(int(model2->frames.size())/1,1);//int step2 = 5;//std::min(int(model2->frames.size()),5);
double register_setup_start = getTime();
		CloudData * cd1 = model ->getCD(model->points.size()/step1);
		registration->setDst(cd1);

		CloudData * cd2	= model2->getCD(model2->points.size()/step2);
		registration->setSrc(cd2);
//printf("register_setup_start:          %5.5f\n",getTime()-register_setup_start);
double register_compute_start = getTime();
		FusionResults fr = registration->getTransform(guess);
//printf("register_compute_start:          %5.5f\n",getTime()-register_compute_start);
        double best = -99999999999999;
        int best_id = -1;
double register_evaluate_start = getTime();

vector<Model *> testmodels;
vector<Matrix4d> testrps;
addModelsToVector(testmodels,testrps,model,Eigen::Matrix4d::Identity());
addModelsToVector(testmodels,testrps,model2,Eigen::Matrix4d::Identity());

int todo = fr.candidates.size();
double expectedCost = double(todo)*computeOcclusionScoreCosts(testmodels);
//printf("expectedCost: %f\n",expectedCost);


		int step = 0.5 + expectedCost/11509168.5;// ~1 sec predicted
		step = std::max(1,step);



//		printf("step: %i\n",step);
		for(unsigned int ca = 0; ca < todo; ca++){
			Eigen::Matrix4d pose = fr.candidates[ca];

			vector<Model *> models;
			vector<Matrix4d> rps;

			addModelsToVector(models,rps,model,Eigen::Matrix4d::Identity());
			unsigned int nr_models1 = models.size();
			addModelsToVector(models,rps,model2,pose);

			vector<vector < OcclusionScore > > ocs = computeOcclusionScore(models,rps,step,false);
			std::vector<std::vector < float > > scores = getScores(ocs);
			std::vector<int> partition = getPartition(scores,2,5,2);

//			for(unsigned int i = 0; i < scores.size(); i++){
//				for(unsigned int j = 0; j < scores.size(); j++){
//					if(scores[i][j] >= 0){printf(" ");}
//					printf("%5.5f ",0.00001*scores[i][j]);
//				}
//				printf("\n");
//			}
//printf("partition "); for(unsigned int i = 0; i < partition.size(); i++){printf("%i ", partition[i]);} printf("\n");

			double sumscore1 = 0;
			for(unsigned int i = 0; i < models.size(); i++){
				for(unsigned int j = 0; j < models.size(); j++){
					if(i < nr_models1 && j < nr_models1){sumscore1 += scores[i][j];}
					if(i >= nr_models1 && j >= nr_models1){sumscore1 += scores[i][j];}
				}
			}

			double sumscore2 = 0;
			for(unsigned int i = 0; i < scores.size(); i++){
				for(unsigned int j = 0; j < scores.size(); j++){
					if(partition[i] == partition[j]){sumscore2 += scores[i][j];}
				}
			}

			double improvement = sumscore2-sumscore1;
			//printf("improvement: %f\n",improvement);

			if(improvement > best){
				computeOcclusionScore(models,rps,step,false);

				best = improvement;
				best_id = ca;
			}
		}

//printf("register_evaluate_start:          %5.5f\n",getTime()-register_evaluate_start);

        if(best_id != -1){
            fr.score = 9999999;
            fr.guess = fr.candidates[best_id];
        }

		delete cd1;
		delete cd2;

		return fr;
	}
	return FusionResults();
}

void ModelUpdaterBasicFuse::fuse(Model * model2, Eigen::Matrix4d guess, double uncertanity){}


UpdatedModels ModelUpdaterBasicFuse::fuseData(FusionResults * f, Model * model1, Model * model2){
	UpdatedModels retval = UpdatedModels();
	Eigen::Matrix4d pose = f->guess;

//	printf("MODEL1 ");
//	model1->print();
//	printf("MODEL2 ");
//	model2->print();
//	printf("input pose\n");
//	std::cout << pose << std::endl << std::endl;

//	std::vector<Eigen::Matrix4d>	current_poses;
//	std::vector<RGBDFrame*>			current_frames;
//	std::vector<ModelMask*>			current_modelmasks;

//	for(unsigned int i = 0; i < model1->frames.size(); i++){
//		current_poses.push_back(				model1->relativeposes[i]);
//		current_frames.push_back(				model1->frames[i]);
//		current_modelmasks.push_back(			model1->modelmasks[i]);
//	}

//	for(unsigned int i = 0; i < model2->frames.size(); i++){
//		current_poses.push_back(	pose	*	model2->relativeposes[i]);
//		current_frames.push_back(				model2->frames[i]);
//		current_modelmasks.push_back(			model2->modelmasks[i]);
//	}

	vector<Model *> models;
	vector<Matrix4d> rps;
	addModelsToVector(models,rps,model,Eigen::Matrix4d::Identity());
	unsigned int nr_models1 = models.size();
	addModelsToVector(models,rps,model2,pose);

	double expectedCost = computeOcclusionScoreCosts(models);
//	printf("expectedCost: %f\n",expectedCost);
	int step = 0.5 + expectedCost/(10.0*11509168.5);// ~10 sec predicted max time
	step = std::max(1,step);

    vector<vector < OcclusionScore > > ocs = computeOcclusionScore(models,rps,step,show_scoring);
	std::vector<std::vector < float > > scores = getScores(ocs);
	std::vector<int> partition = getPartition(scores,2,5,2);

//	for(unsigned int i = 0; i < scores.size(); i++){
//		for(unsigned int j = 0; j < scores.size(); j++){
//			if(scores[i][j] >= 0){printf(" ");}
//			printf("%5.5f ",0.00001*scores[i][j]);
//		}
//		printf("\n");
//	}
//	printf("partition "); for(unsigned int i = 0; i < partition.size(); i++){printf("%i ", partition[i]);} printf("\n");


	double sumscore1 = 0;
	for(unsigned int i = 0; i < models.size(); i++){
		for(unsigned int j = 0; j < models.size(); j++){
			if(i < nr_models1 && j < nr_models1){sumscore1 += scores[i][j];}
			if(i >= nr_models1 && j >= nr_models1){sumscore1 += scores[i][j];}
		}
	}

	double sumscore2 = 0;
	for(unsigned int i = 0; i < scores.size(); i++){
		for(unsigned int j = 0; j < scores.size(); j++){
			if(partition[i] == partition[j]){sumscore2 += scores[i][j];}
		}
	}

	double improvement = sumscore2-sumscore1;

//	for(unsigned int i = 0; i < scores.size(); i++){
//		for(unsigned int j = 0; j < scores.size(); j++){
//			if(scores[i][j] > 0){printf(" ");}
//			printf("%5.5f ",0.0001*scores[i][j]);
//		}
//		printf("\n");
//	}
//	printf("partition "); for(unsigned int i = 0; i < partition.size(); i++){printf("%i ", partition[i]);} printf("\n");
//	printf("sumscore before part: %f\n",sumscore1);
//	printf("sumscore after  part: %f\n",sumscore2);
//	printf("improvement:          %f\n",improvement);
//exit(0);
	std::vector<int> count;
	for(unsigned int i = 0; i < partition.size(); i++){
		if(int(count.size()) <= partition[i]){count.resize(partition[i]+1);}
		count[partition[i]]++;
	}

	int minpart = count[0];
	for(unsigned int i = 1; i < count.size(); i++){minpart = std::min(minpart,count[i]);}

	if(count.size() == 1){
       printf("points before: %i\n",model1->points.size());
		model1->merge(model2,pose);

		model1->recomputeModelPoints();
        printf("points after: %i\n",model1->points.size());
		//model1->scores = scores;
		//model1->total_scores = sumscore;
		retval.updated_models.push_back(model1);
		retval.deleted_models.push_back(model2);
	}else if(improvement > 1){//Cannot fully fuse... separating...

		int c = 0;

		int model1_ind = partition.front();
		bool model1_same = true;
		for(unsigned int i = 0; i < model1->submodels.size(); i++){
			if(partition[c] != model1_ind){model1_same = false;}
			c++;
		}

		int model2_ind = partition.back();
		bool model2_same = true;
		for(unsigned int i = 0; i < model2->submodels.size(); i++){
			if(partition[c] != model2_ind){model2_same = false;}
			c++;
		}

		if(!model1_same || !model2_same){//If something changed, update models
			for(unsigned int i = 0; i < count.size(); i++){retval.new_models.push_back(new Model());}

			for(unsigned int i = 0; i < partition.size(); i++){
				retval.new_models[partition[i]]->submodels.push_back(models[i]);
				retval.new_models[partition[i]]->submodels_relativeposes.push_back(rps[i]);
			}

			for(unsigned int part = 0; part < retval.new_models.size(); part++){
				retval.new_models[part]->recomputeModelPoints();
			}

			retval.deleted_models.push_back(model1);
			retval.deleted_models.push_back(model2);
		}else{
			retval.unchanged_models.push_back(model1);
			retval.unchanged_models.push_back(model2);
		}
		return retval;
	}

	retval.unchanged_models.push_back(model1);
	retval.unchanged_models.push_back(model2);
	return retval;
}

void ModelUpdaterBasicFuse::computeMassRegistration(std::vector<Eigen::Matrix4d> current_poses, std::vector<RGBDFrame*> current_frames,std::vector<cv::Mat> current_masks){
	printf("void ModelUpdaterBasicFuse::computeMassRegistration\n");
	printf("WARNING: THIS METHOD NOT IMPLEMENTED\n");
	exit(0);
}

void ModelUpdaterBasicFuse::setRegistration( Registration * registration_){
	if(registration != 0){delete registration;}
	registration = registration;
}

}


