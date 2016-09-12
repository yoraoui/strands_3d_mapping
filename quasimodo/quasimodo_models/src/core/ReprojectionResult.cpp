#include "../../include/core/ReprojectionResult.h"

namespace reglib
{

ReprojectionResult::ReprojectionResult(){}
ReprojectionResult::ReprojectionResult(unsigned long si, unsigned long di, double a, double rz, double rr, double rg, double rb, double nz, double nrgb){
	src_ind		= si;
	dst_ind		= di;
	angle		= a;

	residualZ	= rz;
	residualR	= rr;
	residualG	= rg;
	residualB	= rb;
}
ReprojectionResult::~ReprojectionResult(){}

}

