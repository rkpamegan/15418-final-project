#include "mesh.h"
#include "cudaRemesh.h"
#include "test.h"

int test_split_edge()
{
    CudaRemesher* remesher = new CudaRemesher();
    Mesh mesh = Mesh::from_indexed_faces({
        Vec3{0.0f, 1.0f, 0.0f}, Vec3{1.0f, 1.0f, 0.0f},
        Vec3{0.0f, 0.0f, 0.0f}, Vec3{1.0f, 0.0f, 0.0f}, Vec3{20.0f, 0.0f, 0.0f}
    }, {
        {0, 2, 3, 1}, {1, 3, 4}
    });
    mesh.describe();
    remesher->setup(mesh);
    Isotropic_Remesh_Params params{
		1, 1.5f, 0.5f, 1, 1.0f
	};
    remesher->isotropic_remesh(params);

    return 0;
}

int test_smooth_vertex() {
    CudaRemesher* remesher = new CudaRemesher();
    Mesh mesh = Mesh::from_indexed_faces({	
		Vec3{-1.0f, 1.0f, 1.0f}, 	Vec3{-1.0f, 1.0f, -1.0f},
		Vec3{-1.0f, -1.0f, -1.0f}, 	Vec3{-1.0f, -1.0f, 1.0f},
		Vec3{1.0f, -1.0f, -1.0f}, 	Vec3{1.0f, -1.0f, 1.0f},
		Vec3{1.0f, 1.0f, -1.0f}, 	Vec3{1.0f, 1.0f, 1.0f}
	},{
		{3, 0, 1, 2}, 
		{5, 3, 2, 4}, 
		{7, 5, 4, 6}, 
		{0, 7, 6, 1}, 
		{0, 3, 5, 7}, 
		{6, 4, 2, 1} });
	// std::printf("finished mesh creation\n");
    remesher->setup(mesh);
	Isotropic_Remesh_Params params{
		1, 1.5f, 0.5f, 1, 0.1f
	};
	remesher->isotropic_remesh(params);

    return 0;
}