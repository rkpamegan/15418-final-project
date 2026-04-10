#include "mesh.h"
#include "cudaRemesh.h"

CudaRemesher::CudaRemesher() {
	cudaDeviceVertices = NULL;
	cudaDeviceHalfedges = NULL;
	cudaDeviceFaces = NULL;
	cudaDeviceEdges = NULL;
}

CudaRemesher::~CudaRemesher() {
	cudaFree(cudaDeviceVertices);
	cudaFree(cudaDeviceEdges);
	cudaFree(cudaDeviceHalfedges);
	cudaFree(cudaDeviceFaces);
}

void CudaRemesher::setup(Mesh mesh) {
	cudaMalloc(&cudaDeviceVertices, sizeof(Mesh::VertexRef) * mesh.vertices.size());
	cudaMalloc(&cudaDeviceEdges, sizeof(Mesh::EdgeRef) * mesh.edges.size());
	cudaMalloc(&cudaDeviceHalfedges, sizeof(Mesh::HalfedgeRef) * mesh.halfedges.size());
	cudaMalloc(&cudaDeviceFaces, sizeof(Mesh::FaceRef) * mesh.faces.size());

	cudaMemcpy(cudaDeviceVertices, mesh.vertices, sizeof(Mesh::VertexRef) * mesh.vertices.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaDeviceEdges, mesh.edges, sizeof(Mesh::EdgeRef) * mesh.edges.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaDeviceHalfedges, mesh.halfedges, sizeof(Mesh::HalfedgeRef) * mesh.halfedges.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaDeviceFaces, mesh.faces, sizeof(Mesh::FaceRef) * mesh.faces.size(), cudaMemcpyHostToDevice);
}

/**
 * Ideas for graph coloring:
 * 	1. 	While graph not colored,
 * 		a.	Choose from uncolored objects with some probability
 * 		b.	Make object's color the lowest color not found in neighborhood.
 *  2.
 *
 */ 
void CudaRemesher::color_vertices() {

}

void CudaRemesher::color_edges() {

}

void CudaRemesher::color_mesh() {
	color_vertices();
	color_edges();
}

__global__ void CudaRemesher::kernelSplitEdge() {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	// if index > numEdges, return
	EdgeRef e = cudaDeviceEdges[index];
	// Phase 1: collect existing elements
	HalfedgeRef h = e->halfedge;
	HalfedgeRef t = h->twin;
	VertexRef v1 = h->vertex;
	VertexRef v2 = t->vertex;

	// Phase 2: Allocate new elements, set data
	VertexRef vm = emplace_vertex();
	vm->position = (v1->position + v2->position) / 2.0f;
	interpolate_data({v1, v2}, vm); //set bone_weights

	EdgeRef e2 = emplace_edge();
	e2->sharp = e->sharp; //copy sharpness flag

	HalfedgeRef h2 = emplace_halfedge();
	interpolate_data({h, h->next}, h2); //set corner_uv, corner_normal

	HalfedgeRef t2 = emplace_halfedge();
	interpolate_data({t, t->next}, t2); //set corner_uv, corner_normal

	// Phase 3: Reassign connectivity (careful about ordering so you don't overwrite values you may need later!)

	vm->halfedge = h2;

	e2->halfedge = h2;

	assert(e->halfedge == h); //unchanged

	//n.b. h remains on the same face so even if h->face->halfedge == h, no fixup needed (t, similarly)

	h2->twin = t;
	h2->next = h->next;
	h2->vertex = vm;
	h2->edge = e2;
	h2->face = h->face;

	t2->twin = h;
	t2->next = t->next;
	t2->vertex = vm;
	t2->edge = e;
	t2->face = t->face;
	
	h->twin = t2;
	h->next = h2;
	assert(h->vertex == v1); // unchanged
	assert(h->edge == e); // unchanged
	//h->face unchanged

	t->twin = h2;
	t->next = t2;
	assert(t->vertex == v2); // unchanged
	t->edge = e2;
	//t->face unchanged


	// Phase 4: Delete unused elements
    erase_face(f_not_used);
    erase_halfedge(h_not_used);

	// Phase 5: Return the correct iterator
	return vm;

}
//isotropic_remesh: improves mesh quality through local operations.
// Do note that this requires a working implementation of EdgeSplit, EdgeFlip, and EdgeCollapse
void CudaRemesher::isotropic_remesh(Isotropic_Remesh_Params const &params) {

	// Compute the mean edge length. This will be the "target length".

	/**
	 * 	1. 	Color mesh vertices such that no two adjacent vertices have the same color
	 * 	2. 	Color mesh edges such that no two edges which share an incident vertex have the same color
	 * 	3. 	For each edge color c_e}:
	 * 		a. 	For each edge with color c_e:
	 * 			i. 	Split edges much longer than the target length.
	 * 				("much longer" means > target length * params.longer_factor)
	 * 			ii.	Collapse edges much shorter than the target length.
	 *	4.	For each color c_v:
	 *		a. 	For each vertex of color c_v:
	 *			i. 	Apply some tangential smoothing to the vertex positions.
	 *				This means move every vertex in the plane of its normal,
	 *				toward the centroid of its neighbors, by params.smoothing_step of
	 *				the total distance (so, smoothing_step of 1 would move all the way,
	 *				smoothing_step of 0 would not move). 
	 *			ii.	Repeat the tangential smoothing part params.smoothing_iterations times.
	 *	5. Repeat steps 1-4 `num_iters` times.
	 */

	//NOTE: many of the steps in this function will be modifying the element
	//      lists they are looping over. Take care to avoid use-after-free
	//      or infinite-loop problems.

}