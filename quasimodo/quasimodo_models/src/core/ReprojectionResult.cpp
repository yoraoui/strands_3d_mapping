#include "../../include/core/ReprojectionResult.h"

namespace reglib
{

ReprojectionResult::ReprojectionResult(){}
ReprojectionResult::ReprojectionResult(unsigned long si, unsigned long di, double a, double rz, double rE2, double rr, double rg, double rb, double nz, double nrgb){
	src_ind		= si;
	dst_ind		= di;
	angle		= a;

	residualZ	= rz;
	residualE2	= rE2;
	residualR	= rr;
	residualG	= rg;
	residualB	= rb;
}
ReprojectionResult::~ReprojectionResult(){}

void ReprojectionResult::print(){
	printf("src_ind	= %i, dst_ind = %i, angle = %f, residualZ = %f, residualR = %f, residualG = %f, residualB = %f\n",src_ind, dst_ind, angle, residualZ, residualR, residualG, residualB);
}
}

