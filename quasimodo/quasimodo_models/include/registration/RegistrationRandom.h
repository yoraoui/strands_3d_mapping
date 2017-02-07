#ifndef RegistrationRandom_H
#define RegistrationRandom_H

#include "Registration.h"
#include <time.h>

#include "RegistrationRefinement.h"

namespace reglib
{
	class RegistrationRandom : public Registration
	{
		public:

        Registration * refinement;
        Registration * refinement2;

		unsigned int steprx;
		unsigned int stepry;
		unsigned int steprz;
		double start_rx;
		double start_ry;
		double start_rz;
		double stop_rx;
		double stop_ry;
		double stop_rz;

		unsigned int steptx;
		unsigned int stepty;
		unsigned int steptz;
		double start_tx;
		double start_ty;
		double start_tz;
		double stop_tx;
		double stop_ty;
		double stop_tz;

		unsigned int src_meantype;
		unsigned int dst_meantype;

		virtual Eigen::Affine3d getMean(std::vector<superpoint> & data, int type);
		virtual void setSrc(std::vector<superpoint> & src_);
		virtual void setDst(std::vector<superpoint> & dst_);


		RegistrationRandom(unsigned int steps = 4);
		~RegistrationRandom();

		bool issame(FusionResults fr1, FusionResults fr2, int stepxsmall);
		
		FusionResults getTransform(Eigen::MatrixXd guess);
	};
}

#endif
