#ifndef MassRegistrationRecursive_H
#define MassRegistrationRecursive_H

#include "MassRegistration.h"

namespace reglib
{
    class MassRegistrationRecursive : public MassRegistration
	{
		public:

		double per_split;
		MassRegistration * massreg;
		std::vector<Model * > models;

		MassRegistrationRecursive(MassRegistration * massreg = 0, double per_split = 10);
        ~MassRegistrationRecursive();
		void addModel(Model * model);

		MassFusionResults getTransforms(std::vector<Eigen::Matrix4d> poses, double start, double stop);
        MassFusionResults getTransforms(std::vector<Eigen::Matrix4d> poses);
	};

}

#endif
