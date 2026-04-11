/**
 * TODO: Find workaround for emplace_x function calls, as they can cause data races
 * TODO: Graph coloring algorithm for vertices/edges
 */
#include <cuda.h>
// #include <cuda_runtime.h>
#include "mesh.h"
#include "cudaRemesh.h"
#include <stdint.h>

CudaRemesher::CudaRemesher() {
	cudaDeviceVertices = NULL;
	cudaDeviceHalfedges = NULL;
	cudaDeviceFaces = NULL;
	cudaDeviceEdges = NULL;

	numVertices = 0;
	numEdges = 0;
	numHalfedges = 0;
	numFaces = 0;
}

CudaRemesher::~CudaRemesher() {
	if (cudaDeviceVertices) {
		cudaFree(cudaDeviceVertices);
		cudaFree(cudaDeviceEdges);
		cudaFree(cudaDeviceHalfedges);
		cudaFree(cudaDeviceFaces);
	}
}

void CudaRemesher::setup(Mesh mesh) {
	cudaMalloc(&cudaDeviceVertices, sizeof(Mesh::Vertex) * mesh.vertices.size());
	cudaMalloc(&cudaDeviceEdges, sizeof(Mesh::Edge) * mesh.edges.size());
	cudaMalloc(&cudaDeviceHalfedges, sizeof(Mesh::Halfedge) * mesh.halfedges.size());
	cudaMalloc(&cudaDeviceFaces, sizeof(Mesh::Face) * mesh.faces.size());

	cudaMemcpy(cudaDeviceVertices, mesh.vertices.data(), sizeof(Mesh::Vertex) * mesh.vertices.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaDeviceEdges, mesh.edges.data(), sizeof(Mesh::Edge) * mesh.edges.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaDeviceHalfedges, mesh.halfedges.data(), sizeof(Mesh::Halfedge) * mesh.halfedges.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaDeviceFaces, mesh.faces.data(), sizeof(Mesh::Face) * mesh.faces.size(), cudaMemcpyHostToDevice);

	cudaMalloc((void**) &numVertices, sizeof(uint32_t));
	cudaMalloc((void**) &numEdges, sizeof(uint32_t));
	cudaMalloc((void**) &numHalfedges, sizeof(uint32_t));
	cudaMalloc((void**) &numFaces, sizeof(uint32_t));

	uint32_t v_size = mesh.vertices.size();
	cudaMemcpy((void*) &numVertices, (void*) &v_size, 1, cudaMemcpyHostToDevice);
	uint32_t e_size = mesh.edges.size();
	cudaMemcpy((void*) &numEdges, (void*) &e_size, 1, cudaMemcpyHostToDevice);
	uint32_t h_size = mesh.halfedges.size();
	cudaMemcpy((void*) &numHalfedges, (void*) &h_size, 1, cudaMemcpyHostToDevice);
	uint32_t f_size = mesh.faces.size();
	cudaMemcpy((void*) &numFaces, (void*) &f_size, 1, cudaMemcpyHostToDevice);
}

/**
 * Ideas for graph coloring:
 * 	1. 	While graph not colored,
 * 		a.	Choose from uncolored objects with some probability
 * 		b.	Make object's color the lowest color not found in neighborhood.
 *  2.
 *
 */ 
__global__ void color_vertices() {

}

__global__ void color_edges() {

}

void CudaRemesher::color_mesh() {
	// color_vertices<<<>>>color_vertices();
	// color_edges();
}

__global__ void kernel_smooth_vertex() {
	
}
__global__ void kernel_flip_edge() {

}

__global__ void kernel_collapse_edge() {

}

__global__ void kernel_split_edge() {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	return;
	/*
	// if index > numEdges, return
	if (index > 0) return;

	////////////////////////// bisect edge //////////////////////////
	HalfedgeRef h = e->halfedge;
	HalfedgeRef t = h->twin;
	VertexRef v1 = h->vertex;
	VertexRef v2 = t->vertex;

	// Phase 2: Allocate new elements, set data
	VertexRef v1 = emplace_vertex();
	v1->position = (v1->position + v2->position) / 2.0f;
	interpolate_data({v1, v2}, v1); //set bone_weights

	EdgeRef e2 = emplace_edge();
	e2->sharp = e->sharp; //copy sharpness flag

	HalfedgeRef h2 = emplace_halfedge();
	interpolate_data({h, h->next}, h2); //set corner_uv, corner_normal

	HalfedgeRef t2 = emplace_halfedge();
	interpolate_data({t, t->next}, t2); //set corner_uv, corner_normal

	// The following elements aren't necessary for the bisect_edge, but they are here to demonstrate phase 4
    FaceRef f_not_used = emplace_face();
    HalfedgeRef h_not_used = emplace_halfedge();

	// Phase 3: Reassign connectivity (careful about ordering so you don't overwrite values you may need later!)

	v1->halfedge = h2;

	e2->halfedge = h2;

	assert(e->halfedge == h); //unchanged

	//n.b. h remains on the same face so even if h->face->halfedge == h, no fixup needed (t, similarly)

	h2->twin = t;
	h2->next = h->next;
	h2->vertex = v1;
	h2->edge = e2;
	h2->face = h->face;

	t2->twin = h;
	t2->next = t->next;
	t2->vertex = v1;
	t2->edge = e;
	t2->face = t->face;
	
	h->twin = t2;
	h->next = h2;
	assert(h->vertex == v1);
	assert(h->edge == e);

	t->twin = h2;
	t->next = t2;
	assert(t->vertex == v2);
	t->edge = e2;

	////////////////////////// split edge //////////////////////////
	HalfedgeRef h = e->halfedge->next;
	HalfedgeRef tp = h->twin;
	HalfedgeRef t = tp->next;

	FaceRef f1 = h->face;
	FaceRef f2 = t->face;

	std::vector<HalfedgeCRef> f1_halfedges;
	HalfedgeRef temp_h1 = f1->halfedge;
	do {
		f1_halfedges.emplace_back(temp_h1);
		temp_h1 = temp_h1->next;
	} while (temp_h1 != f1->halfedge);

	std::vector<HalfedgeCRef> f2_halfedges;
	HalfedgeRef temp_h2 = f2->halfedge;
	do {
		f2_halfedges.emplace_back(temp_h2);
		temp_h2 = temp_h2->next;
	} while (temp_h2 != f2->halfedge);
	// To keep simple, we want to always assume f1 is not a boundary face,
	// and f2 may or may not be a boundary face. The edge must have at least
	// one non-boundary face.
	if (f1->boundary)
	{
		std::swap(h, t);
		std::swap(f1, f2);
	}
	
	HalfedgeRef hn = h->next;
	HalfedgeRef tn = t->next;

	HalfedgeRef hp = h->next;
	while (hp->next != h) hp = hp->next;

	tp = t->next; // reset so we are sure we have the correct tp
	while (tp->next != t) tp = tp->next;

	VertexRef v2 = h->next->next->vertex;

	// Phase 2: create
	EdgeRef e2 = emplace_edge(false);
	HalfedgeRef h2 = emplace_halfedge();
	HalfedgeRef t2 = emplace_halfedge();
	FaceRef f3 = emplace_face(false);
	interpolate_data(f1_halfedges, h2);
	interpolate_data(f1_halfedges, t2);
	// Phase 3: connect
	e2->halfedge = h2;

	f3->halfedge = h2;

	h2->twin = t2;
	h2->next = h;
	h2->vertex = v2;
	h2->edge = e2;
	h2->face = f3;

	t2->twin = h2;
	t2->next = hn->next;
	t2->vertex = v1;
	t2->edge = e2;
	t2->face = f1;

	hn->next = h2;
	hn->face = f3;

	hp->next = t2;

	h->face = f3;

	f1->halfedge = t2;
	
	if (!f2->boundary) // do same thing on f2 side if its not a boundary
	{
		// more collect
		VertexRef v3 = t->next->next->vertex;

		// more create
		EdgeRef e3 = emplace_edge(false);
		HalfedgeRef h3 = emplace_halfedge();
		HalfedgeRef t3 = emplace_halfedge();
		FaceRef f4 = emplace_face(false);
		interpolate_data(f2_halfedges, h3);
		interpolate_data(f2_halfedges, t3);
		// more connect
		e3->halfedge = h3;

		f4->halfedge = t;

		h3->twin = t3;
		h3->next = t;
		h3->vertex = v3;
		h3->edge = e3;
		h3->face = f4;

		t3->twin = h3;
		t3->next = tn->next;
		t3->vertex = v1;
		t3->edge = e3;
		t3->face = f2;

		tn->face = f4;
		tn->next = h3;

		t->face = f4;

		tp->next = t3;
		
		f2->halfedge = t3;
		 
	}
	// (void)e; //this line avoids 'unused parameter' warnings. You can delete it as you fill in the function.
    // return v1;
	*/
}

//isotropic_remesh: improves mesh quality through local operations.
// Do note that this requires a working implementation of EdgeSplit, EdgeFlip, and EdgeCollapse
void CudaRemesher::isotropic_remesh(Isotropic_Remesh_Params const &params) {
	dim3 blockDim(256, 0);
	dim3 gridDim(1, 0);

	kernel_split_edge<<<blockDim, gridDim>>>();

	// Compute the mean edge length. This will be the "target length".

	/**
	 * 	1. 	Color mesh vertices zsuch that no two adjacent vertices have the same color
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