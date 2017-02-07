#ifndef RegistrationRefinement2_H
#define RegistrationRefinement2_H

#include "Registration.h"
#include <time.h>
#include "nanoflann.hpp"

namespace reglib
{
    class RegistrationRefinement2;
    class RegistrationRefinement2Instance{
        public:
        RegistrationRefinement2 * refinement;
        Eigen::Matrix4d result;
        Eigen::Matrix4d lastRematch;
        Eigen::Matrix4d lastModel;
        double meandist;

        bool timestopped;
        double score;
        double stop;
        double lastModelNoise;


        double stopD2;


        unsigned int d_nrp;
        double * dp;
        double * dn;
        double * di;
        Tree3d * trees3d;

        DistanceWeightFunction2PPR3 *   func;

        unsigned int s_nrp;
        double * sp;
        double * sn;
        double * si;
        unsigned int * matches;
        double * rangew;
        double * Qp;
        double * Qn;

        int capacity;
        size_t *  ret_indexes;
        double * out_dists_sqr;

        double * residuals;

        void transform(Eigen::Matrix4d p);
        void computeResiduals();
        int rematch(bool force = false);
        int model(bool force = false);
        int refine();
        int update();

        RegistrationRefinement2Instance( RegistrationRefinement2 * ref, Eigen::MatrixXd guess);
        ~RegistrationRefinement2Instance();

        FusionResults getTransform();
    };

    class RegistrationRefinement2 : public Registration
	{
		public:

		MatchType type;
		bool use_PPR_weight;
		bool use_features;
		bool normalize_matchweights;
        bool use_timer;

        unsigned int d_nrp;
        double * dp;
        double * dn;
        double * di;

        Tree3d * trees3d;
        ArrayData3D<double> * a3d;

        RegistrationRefinement2();
        ~RegistrationRefinement2();

		void setDst(std::vector<superpoint> & dst_);
		
		FusionResults getTransform(Eigen::MatrixXd guess);

        double getChange(Eigen::Matrix4d & change, double meandist);
        double getChange(Eigen::Matrix4d & before, Eigen::Matrix4d & after, double meandist);
	};
}

#endif
