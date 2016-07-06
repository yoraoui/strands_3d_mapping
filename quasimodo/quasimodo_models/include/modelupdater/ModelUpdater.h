#ifndef reglibModelUpdater_H
#define reglibModelUpdater_H

#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h> 
#include <chrono>

#include <Eigen/Dense>
#include "../model/Model.h"
#include "../registration/Registration.h"
#include "../registration/MassRegistration.h"
#include "../mesher/Mesh.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>


#include <pcl/visualization/pcl_visualizer.h>

namespace reglib
{
	using namespace std;
	using namespace Eigen;
	class UpdatedModels{
		public:
		std::vector< Model * > new_models;
		std::vector< Model * > updated_models;
		std::vector< Model * > unchanged_models;
		std::vector< Model * > deleted_models;
		
		UpdatedModels(){}
	};
	
	class OcclusionScore{
		public:
		double score;
		double occlusions;

		OcclusionScore(){score = 0;occlusions = 0;}
		OcclusionScore(	double score_ ,double occlusions_){score = score_;occlusions = occlusions_;}
		~OcclusionScore(){}

		void add(OcclusionScore oc){
			score += oc.score;
			occlusions += oc.occlusions;
		}

		void print(){printf("score: %5.5f occlusions: %5.5f\n",score,occlusions);}
	};

	class ModelUpdater{
        public:
        double occlusion_penalty;
        double massreg_timeout;
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
		Model * model;
		Mesh * mesher;

        int show_init_lvl;//init show
        int show_refine_lvl;//refine show
        bool show_scoring;//fuse scoring show

		ModelUpdater();
		ModelUpdater(Model * model_);
		~ModelUpdater();


		virtual FusionResults registerModel(Model * model2, Eigen::Matrix4d guess = Eigen::Matrix4d::Identity(), double uncertanity = -1);
		virtual void fuse(Model * model2, Eigen::Matrix4d guess = Eigen::Matrix4d::Identity(), double uncertanity = -1);
		virtual UpdatedModels fuseData(FusionResults * f, Model * model1,Model * model2);

		virtual bool isRefinementNeeded();
		virtual bool refineIfNeeded();

		virtual void makeInitialSetup();

		virtual double getCompareUtility(Matrix4d p, RGBDFrame* frame, ModelMask* mask, vector<Matrix4d> & cp, vector<RGBDFrame*> & cf, vector<ModelMask*> & cm);
		virtual void getGoodCompareFrames(vector<Matrix4d> & cp, vector<RGBDFrame*> & cf, vector<ModelMask*> & cm);

		virtual pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr getPCLnormalcloud(vector<superpoint> & points);
		virtual void addSuperPoints(vector<superpoint> & spvec, Matrix4d cp, RGBDFrame* cf, ModelMask* cm, int type = 1, bool debugg = false);
		virtual vector<superpoint> getSuperPoints(vector<Matrix4d> cp, vector<RGBDFrame*> cf, vector<ModelMask*> cm, int type = 1, bool debugg = false);

        virtual void getAreaWeights(Matrix4d p, RGBDFrame* frame1, double * weights1, double * overlaps1, double * total1, RGBDFrame* frame2, double * weights2, double * overlaps2, double * total2);

        virtual void computeOcclusionAreas(vector<Matrix4d> cp, vector<RGBDFrame*> cf, vector<ModelMask*> cm);



		virtual OcclusionScore computeOcclusionScore(vector<superpoint> & spvec, Matrix4d cp, RGBDFrame* cf, ModelMask* cm, int step = 1,  bool debugg = false);
		virtual OcclusionScore computeOcclusionScore(Model * mod, vector<Matrix4d> cp, vector<RGBDFrame*> cf, vector<ModelMask*> cm, Matrix4d rp = Matrix4d::Identity(), int step = 1, bool debugg = false);
		virtual OcclusionScore computeOcclusionScore(Model * model1, Model * model2, Matrix4d rp = Matrix4d::Identity(),int step = 1, bool debugg = false);
		virtual vector<vector < OcclusionScore > > computeOcclusionScore(vector<Model *> models, vector<Matrix4d> rps, int step = 1, bool debugg = false);

		virtual double computeOcclusionScoreCosts(vector<Model *> models);

		virtual void addModelsToVector(vector<Model *> & models, vector<Matrix4d> & rps, Model * model, Matrix4d rp);

		virtual void refine(double reg = 0.05,bool useFullMask = false, int visualization = 0);
		virtual void show(bool stop = true);
		virtual void pruneOcclusions();
//		virtual OcclusionScore computeOcclusionScore(RGBDFrame * src, cv::Mat src_mask, ModelMask * src_modelmask, RGBDFrame * dst, cv::Mat dst_mask, ModelMask * dst_modelmask, Eigen::Matrix4d p, int step = 1, bool debugg = false);
		virtual OcclusionScore computeOcclusionScore(RGBDFrame * src, ModelMask * src_modelmask, RGBDFrame * dst, ModelMask * dst_modelmask, Eigen::Matrix4d p, int step = 1, bool debugg = false);
		virtual void computeResiduals(std::vector<float> & residuals, std::vector<float> & weights, RGBDFrame * src, cv::Mat src_mask, ModelMask * src_modelmask, RGBDFrame * dst, cv::Mat dst_mask, ModelMask * dst_modelmask, Eigen::Matrix4d p, bool debugg = false);

		virtual std::vector<std::vector< OcclusionScore > >	computeAllOcclusionScores(	RGBDFrame * src, cv::Mat src_mask, RGBDFrame * dst, cv::Mat dst_mask,Eigen::Matrix4d p, bool debugg = false);
		virtual	void computeMassRegistration(std::vector<Eigen::Matrix4d> current_poses, std::vector<RGBDFrame*> current_frames,std::vector<cv::Mat> current_masks);

		std::vector<std::vector < OcclusionScore > > getOcclusionScores(std::vector<Eigen::Matrix4d> current_poses, std::vector<RGBDFrame*> current_frames, std::vector<ModelMask*> current_modelmasks,  bool debugg_scores = false, double speedup = 1.0);
        std::vector<std::vector < float > > getScores(std::vector<std::vector < OcclusionScore > > occlusionScores);
		std::vector<int> getPartition(std::vector< std::vector< float > > & scores, int dims = 2, int nr_todo = 5, double timelimit = 2);

		virtual void recomputeScores();

		CloudData * getCD(std::vector<Eigen::Matrix4d> current_poses, std::vector<RGBDFrame*> current_frames,std::vector<cv::Mat> current_masks, int step);
	};

}

#include "ModelUpdaterBasic.h"
#include "ModelUpdaterBasicFuse.h"
#endif // reglibModelUpdater_H
