#include "core/OcclusionScore.h"

namespace reglib
{

OcclusionScore::OcclusionScore(){score = 0;occlusions = 0;}
OcclusionScore::OcclusionScore(	double score_ ,double occlusions_){score = score_;occlusions = occlusions_;}
OcclusionScore::~OcclusionScore(){}

void OcclusionScore::add(OcclusionScore oc){
	score += oc.score;
	occlusions += oc.occlusions;
}

void OcclusionScore::print(){printf("score: %5.5f occlusions: %5.5f\n",score,occlusions);}

}
