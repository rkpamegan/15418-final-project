#include "mesh.h"
#include "cudaRemesh.h"

void CudaRemesher::setup(Mesh mesh) {

}
//isotropic_remesh: improves mesh quality through local operations.
// Do note that this requires a working implementation of EdgeSplit, EdgeFlip, and EdgeCollapse
void CudaRemesher::isotropic_remesh(Isotropic_Remesh_Params const &params) {

	//A2Go2: Isotropic Remeshing
	// Optional! Only one of {A2Go1, A2Go2, A2Go3} is required!

	// Compute the mean edge length. This will be the "target length".

    // Repeat the four main steps for `outer_iterations` iterations:

    // -> Split edges much longer than the target length.
	//     ("much longer" means > target length * params.longer_factor)

    // -> Collapse edges much shorter than the target length.
	//     ("much shorter" means < target length * params.shorter_factor)

    // -> Flip each edge if it improves vertex degree.

    // -> Finally, apply some tangential smoothing to the vertex positions.
	//     This means move every vertex in the plane of its normal,
	//     toward the centroid of its neighbors, by params.smoothing_step of
	//     the total distance (so, smoothing_step of 1 would move all the way,
	//     smoothing_step of 0 would not move).
	// -> Repeat the tangential smoothing part params.smoothing_iterations times.

	//NOTE: many of the steps in this function will be modifying the element
	//      lists they are looping over. Take care to avoid use-after-free
	//      or infinite-loop problems.

}