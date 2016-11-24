#ifndef __QUASIMODO_POINT_TYPE__H
#define __QUASIMODO_POINT_TYPE__H

#define PCL_NO_PRECOMPILE
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

struct QuasimodoPointType
{
	//PCL_ADD_POINT4D; // This adds the members x,y,z which can also be accessed using the point (which is float[4])
	//PCL_ADD_RGB;//Add color to the point
	PCL_ADD_NORMAL4D; // This adds the member normal[3] which can also be accessed using the point (which is float[4])
	union
	{
		struct
		{
			float ce_x;
			float ce_y;
			float de_x;
			float de_y;
		};
		float data_t[4];
	};

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (QuasimodoPointType,
//								   (float, x, x)
//								   (float, y, y)
//								   (float, z, z)
								   (float, normal_x, normal_x)
								   (float, normal_y, normal_y)
								   (float, normal_z, normal_z)
								   //(uint32_t, rgba, rgba)
								   (float, ce_x, ce_x)
								   (float, ce_y, ce_y)
								   (float, de_x, de_x)
								   (float, de_y, de_y)
								   )
#endif
