/**
 * TODO: Find workaround for emplace_x function calls, as they can cause data races
 * TODO: Graph coloring algorithm for vertices/edges
 */

#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "thrust/scan.h"
#include "thrust/reduce.h"
#include "thrust/functional.h"

#include "mesh.h"
#include "cudaRemesh.h"

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
__global__ void kernel_color_vertices(Mesh::Vertex* vertices) { }
__global__ void kernel_color_edges(Mesh::Edge* edges) { }

__global__ void kernel_smooth_vertex() { }

/**
 * Populates a mask of size numEdges with the edges
 * which should be flipped
 */
__global__ void kernel_get_flip_edges(Mesh::Edge* edges, int* mask) { }
__global__ void kernel_flip_edge(Mesh::Edge* edges, int color) { }

/**
 * Populates a mask of size numEdges with the edges
 * which should be collapsed
 */
__global__ void kernel_get_collapse_edges(Mesh::Edge* edges, int* mask) { }
__global__ void kernel_collapse_edge(Mesh::Edge* edges, int color) { }

/**
 * Populates a mask of size numEdges with the edges
 * which should be split
 */
__global__ void kernel_get_split_edges(Mesh::Edge* edges, int* mask) { }
__global__ void kernel_split_edge(Mesh::Edge* edges, int color) { }

//isotropic_remesh: improves mesh quality through local operations.
// Do note that this requires a working implementation of EdgeSplit, EdgeFlip, and EdgeCollapse
void CudaRemesher::isotropic_remesh(Isotropic_Remesh_Params const &params) {
	// dim3 blockDim(256, 0);
	// dim3 gridDim(1, 0);
	
	// kernel_get_split_edges<<<blockDim, gridDim>>>();

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

	for (int t = 0; t < params.num_iters; t++) {
		std::printf("iteration %d of remeshing\n", t);
		// kernel_color_edges<<<>>>(edge_color_mask);

		// color_mask holds color of corresponding edge
		// get max color in color_mask
		int max_color = *thrust::max_element(thrust::device, edge_color_mask, edge_color_mask + numEdges);
		// kernel_get_flip_edges<<<>>>(cudaDeviceEdges, edge_op_mask);
		for (int c = 0; c <= max_color; c++) {
			std::printf("Flipping edges of color %d\n", c);
			// flips all edges with color c if flipping them increases regular-ness
			// kernel_flip_edges<<<>>>(cudaDeviceEdges, c);
		}

		// vertex incidence may change after flipping, so we need to recolor
		// kernel_color_edges<<<>>>(edge_color_mask);
		// kernel_get_split_edges<<<>>>(cudaDeviceEdges, edge_op_mask);
		max_color = *thrust::max_element(thrust::device, edge_color_mask, edge_color_mask + numEdges);

		// TODO: allocate space for all elements that will be produced during splits

		for (int c = 0; c <= max_color; c++) {
			std::printf("Splitting edges of color %d\n", c);
			// flips all edges with color c if they are sufficiently larger than average
			// kernel_split_edges<<<>>>(cudaDeviceEdges, c);
		}

		// TODO: recalculate element counts

		// similar to post-flip recoloring
		// kernel_color_edges<<<>>>(edge_color_mask);
		// kernel_get_collapse_edges<<<>>>(cudaDeviceEdges, edge_op_mask);
		max_color = *thrust::max_element(thrust::device, edge_color_mask, edge_color_mask + numEdges);
		
		for (int c = 0; c <= max_color; c++) {
			std::printf("Collapsing edges of color %d\n", c);
			// collapses all edges with color c if they are sufficiently smaller than average 
			// kernel_collapse_edges<<<>>>(cudaDeviceEdges, c);
		}
		
		// kernel_color_vertices<<<>>>(cudaDeviceVertices, vertex_color_mask);
		max_color = *thrust::max_element(thrust::device, vertex_color_mask, vertex_color_mask + numVertices);
		for (int c = 0; c <= max_color; c++) {
			std::printf("Smoothing vertices of color %d\n", c);
			// kernel_smooth_vertex<<<>>>(cudaDeviceVertices, c);
		}
	}
	//NOTE: many of the steps in this function will be modifying the element
	//      lists they are looping over. Take care to avoid use-after-free
	//      or infinite-loop problems.

}