#ifndef MassRegistrationPPR3_H
#define MassRegistrationPPR3_H

#include "MassRegistration.h"
#include <time.h>

namespace reglib
{
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    typedef Eigen::Matrix<double, 6, 6> Matrix6d;

    class DataNode {
        public:

        double meandist;

        //Randomized color
        int randr;
        int randg;
        int randb;

        //Surface
        unsigned int                surface_nrp;
        double *                    surface_p;
        double *                    surface_n;
        double *                    surface_i;
        bool *                      surface_valid;
        ArrayData3D<double> *       surface_a3d;
        Tree3d *                    surface_trees3d;
        unsigned int                surface_nr_active;
        unsigned int *              surface_active;


        //Surface
        unsigned int                edge_nrp;
        double *                    edge_p;
        double *                    edge_i;
        bool *                      edge_valid;
        ArrayData3D<double> *       edge_a3d;
        Tree3d *                    edge_trees3d;
        unsigned int                edge_nr_active;
        unsigned int *              edge_active;

        void init();
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr getPCLcloud(Eigen::Matrix4d p, int r, int g, int b);
        DataNode(Model * model, bool useSurfacePoints = true, unsigned int surface_nr_active_target = 5000);
        ~DataNode();
    };

    class EdgeData {
        public:

        int nr_matches_orig;

        double score;
        double weights;
        double weightstd;
        DataNode * node1;
        DataNode * node2;
        bool rematched;
        Eigen::Matrix4d rematchPoseLast;
        Eigen::Matrix4d modelPoseLast;
        Eigen::Matrix4d optPoseLast;

        std::vector<unsigned int>   surface_match1;
        std::vector<unsigned int>   surface_match2;
        std::vector<double>         surface_rangew;

		int surfaceRematch(Eigen::Matrix4d p, double convergence = 0.0001, bool force = false, int visualization = 0);
        void computeSurfaceResiduals(Eigen::Matrix4d p, double * residuals, unsigned long & counter);

        bool needRefinement(Eigen::Matrix4d p, double convergence = 0.0001);
        void computeSTDchange(DistanceWeightFunction2 * func, Eigen::Matrix4d p1, Eigen::Matrix4d p2_end,Eigen::Matrix4d p2_start);

		void addSurfaceOptimization(DistanceWeightFunction2 * func, Eigen::Matrix4d p1, Eigen::Matrix4d p2, Matrix6d & ATA, Vector6d & ATb , int visualization = 0);
        void showMatches(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer, Eigen::Matrix4d p = Eigen::Matrix4d::Identity());
        void showMatchesW(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,  DistanceWeightFunction2 * func, Eigen::Matrix4d p1, Eigen::Matrix4d p2);

        EdgeData(DataNode * node1_, DataNode * node2_);
        ~EdgeData();
    };

    class MassRegistrationPPR3 : public MassRegistration
	{
		public:

		reglib::Timer timer;

        int func_setup;

        bool useSurfacePoints;
        DistanceWeightFunction2 * surface_func;

        std::vector< DataNode * > nodes;
        std::vector< std::vector< EdgeData * > > edges;
        std::vector< std::vector< DistanceWeightFunction2 * > > edge_surface_func;


        double next_regularizer;
        bool tune_regularizer;
        double convergence_mul;

        MassRegistrationPPR3(DistanceWeightFunction2 * func);
        MassRegistrationPPR3();
        ~MassRegistrationPPR3();

        double getConvergence();

//        void transform(Eigen::Matrix4d p);
        void show(std::vector<Eigen::Matrix4d> poses, bool stop = true);
        unsigned long rematch(std::vector<Eigen::Matrix4d> poses, bool force = false, bool debugg_print = false);
        unsigned long model(std::vector<Eigen::Matrix4d> poses, bool force = false, bool debugg_print = false);
        unsigned long refine(std::vector<Eigen::Matrix4d> & poses, bool force = false,  bool debugg_print = false);
        void normalizePoses(std::vector<Eigen::Matrix4d> & poses);

        int total_nonconverged(std::vector<Eigen::Matrix4d> before, std::vector<Eigen::Matrix4d> after);

		void addModel(Model * model);
		void addModel(Model * model, int active);
        void removeLastNode();
        MassFusionResults getTransforms(std::vector<Eigen::Matrix4d> poses);

		virtual void clearData();
	};

}

#endif
