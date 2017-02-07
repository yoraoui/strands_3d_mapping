#ifndef RegistrationRefinement3_H
#define RegistrationRefinement3_H

#include "Registration.h"
#include <time.h>
#include "nanoflann.hpp"

namespace reglib
{
    class RegistrationRefinement3;
    class RegistrationRefinement3Instance{
        public:
        RegistrationRefinement3 * refinement;
        Eigen::Matrix4d result;
        Eigen::Matrix4d lastRematch;
        Eigen::Matrix4d lastModel;
        double meandist;

        bool timestopped;
        double score;
        double stop;
        double lastModelNoise;


        double stopD2;


        bool useKeyPoints;
        bool useSurfacePoints;

        DistanceWeightFunction2 *   surface_func;
        DistanceWeightFunction2 *   kp_func;


        //Surface
        Tree3d * surface_trees3d;
        unsigned int surface_d_nrp;
        double * surface_dp;
        double * surface_dn;
        double * surface_di;
        bool * surface_dvalid;

        unsigned int surface_s_nrp;
        double * surface_sp;
        double * surface_sn;
        double * surface_tsp;
        double * surface_tsn;
        double * surface_si;
        double * surface_residuals;

        unsigned int * surface_matches;
        double * surface_rangew;
        double * surface_valid;
        double * surface_Qp;
        double * surface_Qn;

        //Keypoints
        unsigned int kp_d_nrp;
        double * kp_dp;
        double * kp_di;

        unsigned int kp_s_nrp;
        double * kp_sp;
        double * kp_tsp;
        double * kp_si;
        double * kp_residuals;
        int nr_kp_residuals;

        unsigned int * kp_matches;
        double * kp_rangew;
        double * kp_valid;
        double * kp_Qp;


        int capacity;
        size_t *  ret_indexes;
        double * out_dists_sqr;



        void transform(Eigen::Matrix4d p);
        void computeResiduals();
        int rematch(bool force = false);
        int model(bool force = false);
        int refine();
        int update();

        RegistrationRefinement3Instance( RegistrationRefinement3 * ref, Eigen::MatrixXd guess);
        ~RegistrationRefinement3Instance();

        FusionResults getTransform();
    };

    class RegistrationRefinement3 : public Registration
    {
        public:

        int max_rematches;
        int max_remodels;

        bool useKeyPoints;
        bool useSurfacePoints;

        MatchType type;
        bool use_PPR_weight;
        bool use_features;
        bool normalize_matchweights;
        bool use_timer;

        unsigned int d_nrp;
        double * dp;
        double * dn;
        double * di;
        bool   * dvalid;

        unsigned int kp_d_nrp;
        double * kp_dp;
        double * kp_di;

        Tree3d * trees3d;
        ArrayData3D<double> * a3d;


        DistanceWeightFunction2 * func;

        std::vector<int> keypoint_matches;

        RegistrationRefinement3(DistanceWeightFunction2 * func);
        ~RegistrationRefinement3();


        virtual std::string getString();

        void setDst(std::vector<superpoint> & dst_);

        void setSrc(std::vector<superpoint> & src_);


        virtual void setSrc(Model * src, bool recompute = false);
        virtual void setDst(Model * dst, bool recompute = false);

        FusionResults getTransform(Eigen::MatrixXd guess);

        double getChange(Eigen::Matrix4d & change, double meandist);
        double getChange(Eigen::Matrix4d & before, Eigen::Matrix4d & after, double meandist);
    };
}

#endif
