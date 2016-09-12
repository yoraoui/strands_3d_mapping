#ifndef reglibOcclusionScore_H
#define reglibOcclusionScore_H

#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <unordered_map>

#include "OcclusionScore.h"

namespace reglib
{

class OcclusionScore{
	public:
	double score;
	double occlusions;

	OcclusionScore();
	OcclusionScore(	double score_ ,double occlusions_);
	~OcclusionScore();

	void add(OcclusionScore oc);
	void print();
};

}

#endif
