#include "mesh.h"
#include "cudaRemesh.h"

#include <stdio.h>
#include <iostream>

#include "test.h"

int main() {
    // CudaRemesher* remesher = new CudaRemesher();
    // Mesh mesh = Mesh::from_indexed_faces({	
	// 	Vec3{-1.0f, 1.0f, 1.0f}, 	Vec3{-1.0f, 1.0f, -1.0f},
	// 	Vec3{-1.0f, -1.0f, -1.0f}, 	Vec3{-1.0f, -1.0f, 1.0f},
	// 	Vec3{1.0f, -1.0f, -1.0f}, 	Vec3{1.0f, -1.0f, 1.0f},
	// 	Vec3{1.0f, 1.0f, -1.0f}, 	Vec3{1.0f, 1.0f, 1.0f}
	// },{
	// 	{3, 0, 1, 2}, 
	// 	{5, 3, 2, 4}, 
	// 	{7, 5, 4, 6}, 
	// 	{0, 7, 6, 1}, 
	// 	{0, 3, 5, 7}, 
	// 	{6, 4, 2, 1} });
	// // std::printf("finished mesh creation\n");
    // remesher->setup(mesh);
	// Isotropic_Remesh_Params params{
	// 	1, 1.5f, 0.5f, 1, 1.0f
	// };
	// remesher->isotropic_remesh(params);
	test_converge();
    return 0;

}