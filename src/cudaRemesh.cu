/**
 * TODO: Find workaround for emplace_x function calls, as they can cause data races
 * TODO: Graph coloring algorithm for vertices/edges
 * TODO: Function for copying mesh elements back to original mesh
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

		cudaFree((void**) &numVertices);
		cudaFree((void**) &numEdges);
		cudaFree((void**) &numHalfedges);
		cudaFree((void**) &numFaces);

		cudaFree(edge_color_mask);
		cudaFree(edge_op_mask);
		cudaFree(vertex_color_mask);
	}
}

void CudaRemesher::setup(Mesh _mesh) {
	mesh = _mesh;
	cudaMalloc(&cudaDeviceVertices, sizeof(Mesh::Vertex) * _mesh.vertices.size());
	cudaMalloc(&cudaDeviceEdges, sizeof(Mesh::Edge) * _mesh.edges.size());
	cudaMalloc(&cudaDeviceHalfedges, sizeof(Mesh::Halfedge) * _mesh.halfedges.size());
	cudaMalloc(&cudaDeviceFaces, sizeof(Mesh::Face) * _mesh.faces.size());

	cudaMemcpy(cudaDeviceVertices, _mesh.vertices.data(), sizeof(Mesh::Vertex) * _mesh.vertices.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaDeviceEdges, _mesh.edges.data(), sizeof(Mesh::Edge) * _mesh.edges.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaDeviceHalfedges, _mesh.halfedges.data(), sizeof(Mesh::Halfedge) * _mesh.halfedges.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaDeviceFaces, _mesh.faces.data(), sizeof(Mesh::Face) * _mesh.faces.size(), cudaMemcpyHostToDevice);

	cudaMalloc((void**) &numVertices, sizeof(uint32_t));
	cudaMalloc((void**) &numEdges, sizeof(uint32_t));
	cudaMalloc((void**) &numHalfedges, sizeof(uint32_t));
	cudaMalloc((void**) &numFaces, sizeof(uint32_t));

	uint32_t v_size = _mesh.vertices.size();
	cudaMemcpy((void*) &numVertices, (void*) &v_size, 1, cudaMemcpyHostToDevice);
	uint32_t e_size = _mesh.edges.size();
	cudaMemcpy((void*) &numEdges, (void*) &e_size, 1, cudaMemcpyHostToDevice);
	uint32_t h_size = _mesh.halfedges.size();
	cudaMemcpy((void*) &numHalfedges, (void*) &h_size, 1, cudaMemcpyHostToDevice);
	uint32_t f_size = _mesh.faces.size();
	cudaMemcpy((void*) &numFaces, (void*) &f_size, 1, cudaMemcpyHostToDevice);

	cudaMalloc(&edge_lengths, sizeof(float) * numEdges);
	cudaMalloc(&edge_color_mask, sizeof(int) * numEdges);
	cudaMalloc(&edge_op_mask, sizeof(int) * numEdges);
	cudaMalloc(&vertex_color_mask, sizeof(int) * numVertices);
}

/**
 * Ideas for graph coloring:
 * 	1. 	While graph not colored,
 * 		a.	Choose from uncolored objects with some probability
 * 		b.	Make object's color the lowest color not found in neighborhood.
 *  2.
 *
 */ 
__global__ void kernel_color_vertices(Mesh::Vertex* vertices, uint32_t num_vertices,int* color_mask) { }
__global__ void kernel_color_edges(Mesh::Edge* edges, uint32_t num_edges, int* color_mask) { }

__global__ void kernel_smooth_vertex(Mesh::Vertex* vertices, uint32_t num_vertices, int color) { }

/**
 * Populates a mask of size numEdges with the edges
 * which should be flipped
 */
__global__ void kernel_get_flip_edges(Mesh::Edge* edges, uint32_t num_edges, int* op_mask) { }
__global__ void kernel_flip_edge(Mesh::Edge* edges, uint32_t num_edges, int color) { }


__device__ void kernel_get_edge_lengths(Mesh::Edge* edges, float* lengths, uint32_t num_edges)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index > num_edges) return;

	Mesh::Edge* e = &edges[index];
	Mesh::Vertex* const v1 = e->halfedge->vertex;
	Mesh::Vertex* const v2 = e->halfedge->twin->vertex;

	float dx = (v1->position.x - v2->position.x);
	float dy = (v1->position.y - v2->position.y);
	float dz = (v1->position.z - v2->position.z);

	lengths[index] = std::sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Populates a mask of size numEdges with the edges
 * which should be collapsed
 */
__global__ void kernel_get_collapse_edges(float* lengths, uint32_t num_edges, float avg_len, float collapse_factor, int* op_mask) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index > num_edges) return;
	
	op_mask[index] = lengths[index] < avg_len * collapse_factor;
}
__global__ void kernel_collapse_edge(Mesh::Edge* edges, uint32_t num_edges, int color) { }

/**
 * Populates a mask of size numEdges with the edges
 * which should be split
 */
__global__ void kernel_get_split_edges(float* lengths, uint32_t num_edges, float avg_len, float split_factor, int* op_mask) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index > num_edges) return;

	op_mask[index] = lengths[index] > avg_len * split_factor;
}

__global__ void kernel_split_edge(Mesh::Edge* edges, uint32_t num_edges, int color) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index > num_edges) return;

	Mesh::Edge e = edges[index];
	if (e.color != color) return;

}

//isotropic_remesh: improves mesh quality through local operations.
// Do note that this requires a working implementation of EdgeSplit, EdgeFlip, and EdgeCollapse
void CudaRemesher::isotropic_remesh(Isotropic_Remesh_Params const &params) {
	dim3 blockDim;
	dim3 gridDim;
	
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

	//NOTE: many of the steps in this function will be modifying the element
	//      lists they are looping over. Take care to avoid use-after-free
	//      or infinite-loop problems.

	for (int t = 0; t < params.num_iters; t++) {
		std::printf("iteration %d of remeshing\n", t);
		blockDim = dim3(256);
		gridDim = dim3((numEdges + blockDim.x - 1 ) / blockDim.x);
		kernel_color_edges<<<gridDim, blockDim>>>(cudaDeviceEdges, numEdges, edge_color_mask);
		
		// color_mask holds color of corresponding edge
		// get max color in color_mask
		int max_color = *thrust::max_element(thrust::device, edge_color_mask, edge_color_mask + numEdges);
		
		
		kernel_get_flip_edges<<<gridDim, blockDim>>>(cudaDeviceEdges, numEdges, edge_op_mask);
		for (int c = 0; c <= max_color; c++) {
			std::printf("Flipping edges of color %d\n", c);
			// flips all edges with color c if flipping them increases regular-ness
			kernel_flip_edge<<<blockDim, gridDim>>>(cudaDeviceEdges, numEdges, c);
		}
			
		kernel_get_edge_lengths<<<gridDim, blockDim>>>(cudaDeviceEdges, edge_lengths, numEdges);
		float avg_len = thrust::reduce(thrust::device, edge_lengths, edge_lengths + numEdges, 0.0f, thrust::plus<float>());
		// vertex incidence may change after flipping, so we need to recolor
		kernel_color_edges<<<gridDim, blockDim>>>(cudaDeviceEdges, numEdges, edge_color_mask);
		kernel_get_split_edges<<<gridDim, blockDim>>>(edge_lengths, numEdges, avg_len, params.split_factor, edge_op_mask);
		max_color = *thrust::max_element(thrust::device, edge_color_mask, edge_color_mask + numEdges);

		// TODO: allocate space for all elements that will be produced during splits

		for (int c = 0; c <= max_color; c++) {
			std::printf("Splitting edges of color %d\n", c);
			// flips all edges with color c if they are sufficiently larger than average
			kernel_split_edge<<<gridDim, blockDim>>>(cudaDeviceEdges, numEdges, c);
		}

		// TODO: recalculate element counts

		// similar to post-flip recoloring
		kernel_color_edges<<<gridDim, blockDim>>>(cudaDeviceEdges, numEdges, edge_color_mask);

		kernel_get_edge_lengths<<<gridDim, blockDim>>>(cudaDeviceEdges, edge_lengths, numEdges);
		float avg_len = thrust::reduce(thrust::device, edge_lengths, edge_lengths + numEdges, 0.0f, thrust::plus<float>());

		kernel_get_collapse_edges<<<gridDim, blockDim>>>(edge_lengths, numEdges, avg_len, params.collapse_factor, edge_op_mask);
		max_color = *thrust::max_element(thrust::device, edge_color_mask, edge_color_mask + numEdges);
		
		for (int c = 0; c <= max_color; c++) {
			std::printf("Collapsing edges of color %d\n", c);
			// collapses all edges with color c if they are sufficiently smaller than average 
			kernel_collapse_edge<<<gridDim, blockDim>>>(cudaDeviceEdges, numEdges, c);
		}
		
		gridDim = dim3((numVertices + blockDim.x - 1) / blockDim.x);
		// kernel_color_vertices<<<>>>(cudaDeviceVertices, vertex_color_mask);
		max_color = *thrust::max_element(thrust::device, vertex_color_mask, vertex_color_mask + numVertices);
		for (int c = 0; c <= max_color; c++) {
			std::printf("Smoothing vertices of color %d\n", c);
			kernel_smooth_vertex<<<gridDim, blockDim>>>(cudaDeviceVertices, numVertices, c);
		}
	}
	

}