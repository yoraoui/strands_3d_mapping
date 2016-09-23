#ifndef reglibReprojectionResult_H
#define reglibReprojectionResult_H

#include <Eigen/Dense>


namespace reglib
{
class ReprojectionResult{
	public:
	unsigned long	src_ind;
	unsigned long	dst_ind;
	double			angle;
	double			residualZ;
	double			residualR;
	double			residualG;
	double			residualB;

	double noiseZ;
	double noiseRGB;

	ReprojectionResult();
	ReprojectionResult(unsigned long si, unsigned long di, double rz, double a, double rr, double rg, double rb, double nz, double nrgb);
	~ReprojectionResult();
	void print();
};
}

#endif
