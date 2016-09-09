#ifndef MassRegistrationPPR2_H
#define MassRegistrationPPR2_H

#include "MassRegistration.h"
#include <time.h>

namespace reglib
{

	class TestMatch{
		public:

		long src;
		long dst;
		double d;

		TestMatch(long src_, long dst_,	double d_){
			src = src_;
			dst = dst_;
			d = d_;
		}

		~TestMatch(){}
	};

	class MassRegistrationPPR2 : public MassRegistration
	{
		public:

		MatchType type;

		bool fast_opt;
		bool use_PPR_weight;
		bool use_features;
		bool normalize_matchweights;

		double stopval;
		unsigned steps;

		double rematch_time;
		double residuals_time;
		double opt_time;
		double computeModel_time;
		double setup_matches_time;
		double setup_equation_time;
		double setup_equation_time2;
		double solve_equation_time;
		double total_time_start;

		const unsigned long maxcount = 1000000;

		Model * model;

		std::vector<long > nr_datas;

		std::vector< bool > is_ok;

		std::vector< Eigen::Matrix<double, 3, Eigen::Dynamic> > points;
		std::vector< Eigen::Matrix<double, 3, Eigen::Dynamic> > colors;
		std::vector< Eigen::Matrix<double, 3, Eigen::Dynamic> > normals;
		std::vector< Eigen::Matrix<double, 3, Eigen::Dynamic> > transformed_points;
		std::vector< Eigen::Matrix<double, 3, Eigen::Dynamic> > transformed_normals;
		std::vector< Eigen::VectorXd > informations;



		std::vector< long >				kp_nr_arraypoints;
		std::vector< double * >			kp_arraypoints;
		std::vector< double * >			kp_arraynormals;
		std::vector< double * >			kp_arrayinformations;
		std::vector< uint64_t * >		kp_arraydescriptors;
		std::vector< std::vector< std::vector< TestMatch > > > kp_matches;
		double * kp_Qp_arr;
		double * kp_Qn_arr;
		double * kp_Xp_arr;
		double * kp_Xn_arr;
		double * kp_rangeW_arr;
		DistanceWeightFunction2PPR2 * kpfunc;
		//std::vector< std::vector< std::vector<long > > > matchids;

		std::vector< long >      frameid;

		bool use_surface;
		std::vector< long >		nr_arraypoints;
		std::vector< double * > arraypoints;
		std::vector< double * > arraynormals;
		std::vector< double * > arraycolors;
		std::vector< double * > arrayinformations;
		std::vector< Tree3d * > trees3d;
		std::vector< ArrayData3D<double> * > a3dv;
		std::vector<long > nr_matches;
		std::vector< std::vector< std::vector<long > > > matchids;
        std::vector< std::vector< std::vector<double> > > matchdists;
		double * Qp_arr;
		double * Qn_arr;
		double * Xp_arr;
		double * Xn_arr;
		double * rangeW_arr;
		DistanceWeightFunction2PPR2 * func;
		std::vector< std::vector< double > > matchscores;


		bool use_depthedge;
		//long depthedge_nr_neighbours;
		std::vector< long >		depthedge_nr_arraypoints;
        std::vector< double * > depthedge_arraypoints;
		//std::vector< long * >    depthedge_neighbours;
		std::vector< double * > depthedge_arrayinformations;
		std::vector< Tree3d * > depthedge_trees3d;
		std::vector< ArrayData3D<double> * > depthedge_a3dv;
		std::vector<long > depthedge_nr_matches;
		std::vector< std::vector< std::vector<long > > > depthedge_matchids;
		std::vector< std::vector< std::vector<double> > > depthedge_matchdists;
		double * depthedge_Qp_arr;
		double * depthedge_Xp_arr;
		double * depthedge_rangeW_arr;
		DistanceWeightFunction2PPR2 * depthedge_func;

		std::vector<long > sweepids;
		std::vector<long > background_nr_datas;
//		std::vector< Eigen::Matrix<double, 3, Eigen::Dynamic> > background_points;
//		std::vector< Eigen::Matrix<double, 3, Eigen::Dynamic> > background_colors;
//		std::vector< Eigen::Matrix<double, 3, Eigen::Dynamic> > background_normals;
//		std::vector< Eigen::Matrix<double, 3, Eigen::Dynamic> > background_transformed_points;
//		std::vector< Eigen::Matrix<double, 3, Eigen::Dynamic> > background_transformed_normals;
//		std::vector< Eigen::VectorXd > background_informations;
//		std::vector< nanoflann::KDTreeAdaptor<Eigen::Matrix3Xd, 3, nanoflann::metric_L2_Simple> * > background_trees;
//		std::vector<long > background_nr_matches;
//		std::vector< std::vector< std::vector<long > > > background_matchids;


//		std::vector<long > feature_start;//Dimension of data a specific feature starts, if the feature is RGB this should be 3
//		std::vector<long > feature_end;//Dimension of data a specific feature ends, if the feature is RGB this should be 5
//		std::vector< DistanceWeightFunction2 * > feature_func;




		MassRegistrationPPR2(double startreg = 0.05, bool visualize = false);
		~MassRegistrationPPR2();

		void clearData();
        void addModel(Model * model);
		void addModelData(Model * model, bool submodels = true);
		void addData(RGBDFrame* frame, ModelMask * mmask);
		void addData(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud);

		MassFusionResults getTransforms(std::vector<Eigen::Matrix4d> guess);
		void setData(std::vector<RGBDFrame*> frames_, std::vector<ModelMask *> mmasks);
		void setData(std::vector< pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr > all_clouds);

		void rematchKeyPoints(std::vector<Eigen::Matrix4d> poses, std::vector<Eigen::Matrix4d> prev_poses, bool first);
		void rematch(std::vector<Eigen::Matrix4d> poses, std::vector<Eigen::Matrix4d> prev_poses, bool rematch_surface, bool rematch_edges,bool first);
		Eigen::MatrixXd getAllResiduals(std::vector<Eigen::Matrix4d> poses);
		Eigen::MatrixXd depthedge_getAllResiduals(std::vector<Eigen::Matrix4d> poses);
		Eigen::MatrixXd getAllKpResiduals(std::vector<Eigen::Matrix4d> poses);


		void showEdges(std::vector<Eigen::Matrix4d> poses);

		//Eigen::MatrixXd getAllKPResiduals(std::vector<Eigen::Matrix4d> poses);

		std::vector<Eigen::Matrix4d> optimize(std::vector<Eigen::Matrix4d> poses);
	};

}

#endif
