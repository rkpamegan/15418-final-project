#include "mesh.h"
#include "cudaRemesh.h"
#include "test.h"

int test_split_edge()
{
    CudaRemesher* remesher = new CudaRemesher();
    // Closed tetrahedron with one vertex (v3) stretched far away,
    // so the 3 edges incident to v3 are flagged for splitting.
    Mesh mesh = Mesh::from_indexed_faces({
        Vec3{0.0f, 0.0f, 0.0f},
        Vec3{3.0f, 0.0f, 0.0f},
        Vec3{0.0f, 3.0f, 0.0f},
        Vec3{0.0f, 0.0f, 30.0f}
    }, {
        {0, 2, 1}, // bottom (winding outward, -z normal)
        {0, 1, 3},
        {1, 2, 3},
        {2, 0, 3}
    });
    mesh.describe();
    remesher->setup(mesh);
    // collapse_factor = 0.1 keeps short bottom edges (~3) above threshold so only split fires.
    Isotropic_Remesh_Params params{
		1, 1.5f, 0.1f, 1, 1.0f
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