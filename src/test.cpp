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

int test_collapse_edge()
{
    CudaRemesher* remesher = new CudaRemesher();
    // Triangular bipyramid (closed, no boundary), but asymmetric:
    // v3 is placed almost on top of v2, so ONLY the v2-v3 equator edge is
    // very short. After collapsing it once we get a valid tetrahedron.
    // (Collapsing all 3 equator edges would degenerate the entire mesh.)
    Mesh mesh = Mesh::from_indexed_faces({
        Vec3{ 0.0f,    0.0f,    1.0f},     // v0 top apex
        Vec3{ 0.0f,    0.0f,   -1.0f},     // v1 bottom apex
        Vec3{ 1.0f,    0.0f,    0.0f},     // v2 equator
        Vec3{ 1.0001f, 0.0f,    0.0f},     // v3 equator -- almost identical to v2
        Vec3{-0.5f,    0.866f,  0.0f}      // v4 equator
    }, {
        {0, 2, 3}, {0, 3, 4}, {0, 4, 2},
        {1, 3, 2}, {1, 4, 3}, {1, 2, 4}
    });
    mesh.describe();
    remesher->setup(mesh);
    // Edge lengths: v2-v3 ~0.0001 (short), other equator edges ~1.73, apex edges ~1.41.
    // avg ~1.33; collapse_factor=0.5 -> threshold ~0.66 -> only v2-v3 collapses.
    // split_factor=3.0 -> threshold ~3.99 -> nothing splits.
    Isotropic_Remesh_Params params{
        1, 3.0f, 0.5f, 1, 1.0f
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