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

#include <cstdlib>

#include "mesh.h"
#include "cudaRemesh.h"
#include "vec3.h"

CudaRemesher::CudaRemesher() {
	cudaDeviceVertices = NULL;
	cudaDeviceHalfedges = NULL;
	cudaDeviceFaces = NULL;
	cudaDeviceEdges = NULL;

	numVertices = 0;
	numEdges = 0;
	numHalfedges = 0;
	numFaces = 0;

	float* edge_lengths = NULL;
    int* edge_color_mask = NULL;
	int* edge_op_mask = NULL;
	int* vertex_color_mask = NULL;
	Vec3* vertex_pos = NULL;
	Vec3* vertex_normals = NULL;
}

CudaRemesher::~CudaRemesher() {
	if (mesh) update_mesh();

	if (cudaDeviceVertices) {
		cudaFree(cudaDeviceVertices);
		cudaFree(cudaDeviceEdges);
		cudaFree(cudaDeviceHalfedges);
		cudaFree(cudaDeviceFaces);

		cudaFree((void**) &numVertices);
		cudaFree((void**) &numEdges);
		cudaFree((void**) &numHalfedges);
		cudaFree((void**) &numFaces);

		cudaFree(edge_lengths);
		cudaFree(edge_color_mask);
		cudaFree(edge_op_mask);
		cudaFree(vertex_color_mask);
		cudaFree(vertex_pos);
		cudaFree(vertex_normals);
		cudaFree(vertex_priorities);
		cudaFree(edge_priorities);
		cudaFree(d_coloring_done);
	}
}

void CudaRemesher::setup(Mesh &_mesh) {
	mesh = &_mesh;
	cudaMalloc(&cudaDeviceVertices, sizeof(Mesh::Vertex) * _mesh.vertices.size());
	cudaMalloc(&cudaDeviceEdges, sizeof(Mesh::Edge) * _mesh.edges.size());
	cudaMalloc(&cudaDeviceHalfedges, sizeof(Mesh::Halfedge) * _mesh.halfedges.size());
	cudaMalloc(&cudaDeviceFaces, sizeof(Mesh::Face) * _mesh.faces.size());
	std::printf("malloc'd elements\n");

	cudaMemcpy(cudaDeviceVertices, _mesh.vertices.data(), sizeof(Mesh::Vertex) * _mesh.vertices.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaDeviceEdges, _mesh.edges.data(), sizeof(Mesh::Edge) * _mesh.edges.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaDeviceHalfedges, _mesh.halfedges.data(), sizeof(Mesh::Halfedge) * _mesh.halfedges.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaDeviceFaces, _mesh.faces.data(), sizeof(Mesh::Face) * _mesh.faces.size(), cudaMemcpyHostToDevice);
	std::printf("memcpy elements\n");

	numVertices = _mesh.vertices.size();
	numEdges = _mesh.edges.size();
	numHalfedges = _mesh.halfedges.size();
	numFaces = _mesh.faces.size();
	std::printf("setup: numVertices=%u, numEdges=%u, numHalfedges=%u, numFaces=%u\n",
		numVertices, numEdges, numHalfedges, numFaces);

	cudaMalloc(&edge_lengths, sizeof(float) * numEdges);
	cudaMalloc(&edge_color_mask, sizeof(int) * numEdges);
	cudaMalloc(&edge_op_mask, sizeof(int) * numEdges);
	cudaMalloc(&vertex_color_mask, sizeof(int) * numVertices);
	cudaMalloc(&vertex_pos, sizeof(Vec3) * numVertices);
	cudaMalloc(&vertex_normals, sizeof(Vec3) * numVertices);
	std::printf("malloc'd masks\n");

	// Generate random priorities for graph coloring
	std::vector<int> h_priorities(numVertices);
	for (uint32_t i = 0; i < numVertices; i++) {
		h_priorities[i] = rand();
	}
	cudaMalloc(&vertex_priorities, sizeof(int) * numVertices);
	cudaMemcpy(vertex_priorities, h_priorities.data(), sizeof(int) * numVertices, cudaMemcpyHostToDevice);

	// Generate random priorities for edge coloring
	std::vector<int> h_edge_priorities(numEdges);
	for (uint32_t i = 0; i < numEdges; i++) {
		h_edge_priorities[i] = rand();
	}
	cudaMalloc(&edge_priorities, sizeof(int) * numEdges);
	cudaMemcpy(edge_priorities, h_edge_priorities.data(), sizeof(int) * numEdges, cudaMemcpyHostToDevice);

	cudaMalloc(&d_coloring_done, sizeof(bool));
}

// Updates mesh fields to the remeshed values
void CudaRemesher::update_mesh() {
	mesh->vertices.resize(numVertices);
	mesh->edges.resize(numEdges);
	mesh->halfedges.resize(numHalfedges);
	mesh->faces.resize(numFaces);

	cudaMemcpy(mesh->vertices.data(), cudaDeviceVertices, sizeof(Mesh::Vertex) * numVertices, cudaMemcpyDeviceToHost);
	cudaMemcpy(mesh->edges.data(), cudaDeviceEdges, sizeof(Mesh::Edge) * numEdges, cudaMemcpyDeviceToHost);
	cudaMemcpy(mesh->halfedges.data(), cudaDeviceHalfedges, sizeof(Mesh::Halfedge) * numHalfedges, cudaMemcpyDeviceToHost);
	cudaMemcpy(mesh->faces.data(), cudaDeviceFaces, sizeof(Mesh::Face) * numFaces, cudaMemcpyDeviceToHost);
}

/**
 * Ideas for graph coloring:
 * 	Jones-Plassmann parallel graph coloring:
 * 	Each vertex has a random priority. Each round, a vertex colors itself
 * 	only if it has the highest priority among all its uncolored neighbors.
 * 	It picks the smallest color not used by any already-colored neighbor.
 */ 
__global__ void kernel_color_vertices(
	Mesh::Vertex* vertices, Mesh::Halfedge* halfedges,
	uint32_t num_vertices, int* color_mask, int* priorities, bool* done)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= num_vertices) return;
	if (color_mask[idx] != -1) return; // already colored

	// Check if this vertex has higher priority than all uncolored neighbors
	bool is_local_max = true;
	// Track which colors are used by colored neighbors (bitmask, supports up to 32 colors)
	uint32_t used_colors = 0;

	// Walk around the vertex using halfedges to find all neighbors
	uint32_t start_he = vertices[idx].halfedge_idx;
	if (start_he == INVALID_IDX) {
		color_mask[idx] = 0; // isolated vertex
		return;
	}

	uint32_t he = start_he;
	do {
		// he goes out from vertex idx, twin goes back, twin->next goes out from neighbor
		uint32_t twin_he = halfedges[he].twin_idx;
		if (twin_he == INVALID_IDX) break;

		uint32_t neighbor = halfedges[twin_he].vertex_idx;

		if (color_mask[neighbor] == -1) {
			// neighbor is uncolored — check priority
			if (priorities[neighbor] > priorities[idx]) {
				is_local_max = false;
				break;
			}
			// tie-break by index
			if (priorities[neighbor] == priorities[idx] && neighbor > idx) {
				is_local_max = false;
				break;
			}
		} else {
			// neighbor is colored — record its color
			if (color_mask[neighbor] < 32) {
				used_colors |= (1u << color_mask[neighbor]);
			}
		}

		// move to next outgoing halfedge around vertex
		he = halfedges[twin_he].next_idx;
		if (he == INVALID_IDX) break;
	} while (he != start_he);

	if (is_local_max) {
		// pick smallest color not used by neighbors
		int color = 0;
		while (used_colors & (1u << color)) color++;
		color_mask[idx] = color;
	} else {
		*done = false; // still have uncolored vertices
	}
}
__global__ void kernel_color_edges(
	Mesh::Edge* edges, Mesh::Halfedge* halfedges, Mesh::Vertex* vertices,
	uint32_t num_edges, int* color_mask, int* priorities, bool* done)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= num_edges) return;
	if (color_mask[idx] != -1) return; // already colored

	bool is_local_max = true;
	uint32_t used_colors = 0;

	// An edge has two endpoints. Find all edges incident to each endpoint.
	uint32_t h_idx = edges[idx].halfedge_idx;
	if (h_idx == INVALID_IDX) {
		color_mask[idx] = 0;
		return;
	}

	// Two endpoints of this edge
	uint32_t v1 = halfedges[h_idx].vertex_idx;
	uint32_t twin_idx = halfedges[h_idx].twin_idx;
	uint32_t v2 = (twin_idx != INVALID_IDX) ? halfedges[twin_idx].vertex_idx : INVALID_IDX;

	// Walk around v1 to find all incident edges (adjacent to this edge)
	uint32_t start_he = vertices[v1].halfedge_idx;
	if (start_he != INVALID_IDX) {
		uint32_t he = start_he;
		do {
			uint32_t neighbor_edge = halfedges[he].edge_idx;
			if (neighbor_edge != idx && neighbor_edge != INVALID_IDX) {
				if (color_mask[neighbor_edge] == -1) {
					if (priorities[neighbor_edge] > priorities[idx] ||
						(priorities[neighbor_edge] == priorities[idx] && neighbor_edge > idx)) {
						is_local_max = false;
						break;
					}
				} else if (color_mask[neighbor_edge] < 32) {
					used_colors |= (1u << color_mask[neighbor_edge]);
				}
			}
			uint32_t tw = halfedges[he].twin_idx;
			if (tw == INVALID_IDX) break;
			he = halfedges[tw].next_idx;
			if (he == INVALID_IDX) break;
		} while (he != start_he);
	}

	// Walk around v2 to find all incident edges
	if (is_local_max && v2 != INVALID_IDX) {
		start_he = vertices[v2].halfedge_idx;
		if (start_he != INVALID_IDX) {
			uint32_t he = start_he;
			do {
				uint32_t neighbor_edge = halfedges[he].edge_idx;
				if (neighbor_edge != idx && neighbor_edge != INVALID_IDX) {
					if (color_mask[neighbor_edge] == -1) {
						if (priorities[neighbor_edge] > priorities[idx] ||
							(priorities[neighbor_edge] == priorities[idx] && neighbor_edge > idx)) {
							is_local_max = false;
							break;
						}
					} else if (color_mask[neighbor_edge] < 32) {
						used_colors |= (1u << color_mask[neighbor_edge]);
					}
				}
				uint32_t tw = halfedges[he].twin_idx;
				if (tw == INVALID_IDX) break;
				he = halfedges[tw].next_idx;
				if (he == INVALID_IDX) break;
			} while (he != start_he);
		}
	}

	if (is_local_max) {
		int color = 0;
		while (used_colors & (1u << color)) color++;
		color_mask[idx] = color;
	} else {
		*done = false;
	}
}

/**
 * Computes the normal of every vertex in the mesh based on its neighbors
 * Original code provided by 15-362 course staff
 */
__global__ void kernel_get_vertex_normals(
	Mesh::Vertex* vertices,
	Mesh::Halfedge* halfedges,
	Mesh::Face* faces,
	Vec3* vertex_normals,
	uint32_t num_vertices,
	uint32_t num_halfedges
) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= num_vertices) return;

	Vec3 n = Vec3(0.0f, 0.0f, 0.0f);
	Vec3 pi = vertices[index].position;
	uint32_t h_idx = vertices[index].halfedge_idx;
	uint32_t curr_idx = h_idx;

	//walk clockwise around the vertex:
	do {
		Mesh::Halfedge h = halfedges[curr_idx];
		Vec3 pk = vertices[halfedges[h.next_idx].vertex_idx].position;
		h = halfedges[halfedges[h.twin_idx].next_idx];
		Vec3 pj = vertices[halfedges[h.next_idx].vertex_idx].position;
		//pi,pk,pj is a ccw-oriented triangle covering the area of h->face incident on the vertex
		if (!faces[h.face_idx].boundary) n += cross(pj - pi, pk - pi);
	} while (curr_idx != h_idx);
	vertex_normals[index] = n.unit();
}

__global__ void kernel_smooth_vertex(
	Mesh::Vertex* vertices,
	Mesh::Edge* edges,
	Mesh::Halfedge* halfedges,
	Mesh::Face* faces,
	int* vertex_color_mask,
	Vec3* vertex_normals,
	Vec3* vertex_pos,
	uint32_t num_vertices,
	uint32_t num_edges,
	uint32_t num_halfedges,
	uint32_t num_faces,
	float smoothing_factor,
	int color
) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= num_vertices) return;
	if (vertex_color_mask[index] != color) return;

	Mesh::Vertex v = vertices[index];

	Vec3 center;

	uint32_t h_idx = v.halfedge_idx;
	uint32_t curr_idx = h_idx;

	uint32_t count = 0;
	do {
		Mesh::Halfedge h = halfedges[curr_idx];

		Mesh::Vertex neighbor = vertices[halfedges[h.twin_idx].vertex_idx];
		center += neighbor.position;
		count++;

		curr_idx = halfedges[h.twin_idx].next_idx;
	} while (curr_idx != h_idx);
	
	center /= count;

	center = v.position + smoothing_factor * (center - v.position);
	Vec3 normal = vertex_normals[index];
	center = center - dot(normal, center) * normal;
	std::printf("vertex %d: (%f %f %f) -> (%f %f %f)\n", index, v.position.x, v.position.y, v.position.z, center.x, center.y, center.z);
	vertex_pos[index] = center;
}

__global__ void kernel_update_vertex_pos(
	Mesh::Vertex* vertices,
	Vec3* vertex_pos,
	uint32_t num_vertices
) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= num_vertices) return;

	vertices[index].position = vertex_pos[index];
}

/**
 * Computes the degree of a vertex by walking around it
 */
__device__ uint32_t vertex_degree(Mesh::Vertex* vertices, Mesh::Halfedge* halfedges, uint32_t v_idx) {
	uint32_t start_he = vertices[v_idx].halfedge_idx;
	if (start_he == INVALID_IDX) return 0;
	uint32_t he = start_he;
	uint32_t deg = 0;
	do {
		deg++;
		uint32_t tw = halfedges[he].twin_idx;
		if (tw == INVALID_IDX) { deg++; break; } // boundary vertex
		he = halfedges[tw].next_idx;
		if (he == INVALID_IDX) break;
	} while (he != start_he);
	return deg;
}

/**
 * Populates a mask of size num Edges with the edges
 * which should be flipped
 */
__global__ void kernel_get_flip_edges(
	Mesh::Edge* edges, Mesh::Halfedge* halfedges, Mesh::Vertex* vertices, Mesh::Face* faces,
	uint32_t num_edges, int* op_mask)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= num_edges) { return; }

	op_mask[idx] = 0;

	uint32_t h_idx = edges[idx].halfedge_idx;
	if (h_idx == INVALID_IDX) return;

	uint32_t t_idx = halfedges[h_idx].twin_idx;
	if (t_idx == INVALID_IDX) return; // boundary edge, can't flip

	// Skip edges that touch a boundary face
	if (faces[halfedges[h_idx].face_idx].boundary) return;
	if (faces[halfedges[t_idx].face_idx].boundary) return;

	// 4 vertices of the diamond:
	//       B
	//      /|\
	//     / | \
	//    A  |  C
	//     \ | /
	//      \|/
	//       D
	// h = B->D, twin = D->B
	uint32_t vB = halfedges[h_idx].vertex_idx;
	uint32_t vD = halfedges[t_idx].vertex_idx;
	uint32_t vA = halfedges[halfedges[halfedges[h_idx].next_idx].next_idx].vertex_idx;
	uint32_t vC = halfedges[halfedges[halfedges[t_idx].next_idx].next_idx].vertex_idx;

	int degA = vertex_degree(vertices, halfedges, vA);
	int degB = vertex_degree(vertices, halfedges, vB);
	int degC = vertex_degree(vertices, halfedges, vC);
	int degD = vertex_degree(vertices, halfedges, vD);

	// deviation from ideal degree 6
	int dev_before = abs(degA - 6) + abs(degB - 6) + abs(degC - 6) + abs(degD - 6);
	// after flip: B,D lose 1 edge; A,C gain 1 edge
	int dev_after = abs(degA + 1 - 6) + abs(degB - 1 - 6) + abs(degC + 1 - 6) + abs(degD - 1 - 6);

	op_mask[idx] = (dev_after < dev_before) ? 1 : 0;
}
__global__ void kernel_flip_edge(
	Mesh::Edge* edges, Mesh::Halfedge* halfedges, Mesh::Vertex* vertices, Mesh::Face* faces,
	uint32_t num_edges, int* edge_color_mask, int* op_mask, int color)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= num_edges) return;
	if (edge_color_mask[idx] != color) return;
	if (op_mask[idx] != 1) return;

	uint32_t h_idx = edges[idx].halfedge_idx;
	if (h_idx == INVALID_IDX) return;
	uint32_t t_idx = halfedges[h_idx].twin_idx;
	if (t_idx == INVALID_IDX) return; // boundary edge

	// Gather the 6 halfedges
	uint32_t hn_idx = halfedges[h_idx].next_idx;   // h_next
	uint32_t hp_idx = halfedges[hn_idx].next_idx;   // h_prev (= h_next.next)
	uint32_t tn_idx = halfedges[t_idx].next_idx;    // t_next
	uint32_t tp_idx = halfedges[tn_idx].next_idx;   // t_prev (= t_next.next)

	// The 4 vertices
	uint32_t v0 = halfedges[h_idx].vertex_idx;
	uint32_t v1 = halfedges[t_idx].vertex_idx;
	uint32_t v2 = halfedges[hp_idx].vertex_idx;
	uint32_t v3 = halfedges[tp_idx].vertex_idx;

	// The 2 faces
	uint32_t f0 = halfedges[h_idx].face_idx;
	uint32_t f1 = halfedges[t_idx].face_idx;

	// --- Rewire ---
	// After flip:
	//   Face f0: h(v2->v3) -> t_prev(v3->v1) -> h_next(v1->v2)
	//   Face f1: t(v3->v2) -> h_prev(v2->v0) -> t_next(v0->v3)

	// Update vertices of h and t
	halfedges[h_idx].vertex_idx = v2;
	halfedges[t_idx].vertex_idx = v3;

	// Update next pointers
	halfedges[h_idx].next_idx = tp_idx;
	halfedges[tp_idx].next_idx = hn_idx;
	halfedges[hn_idx].next_idx = h_idx;

	halfedges[t_idx].next_idx = hp_idx;
	halfedges[hp_idx].next_idx = tn_idx;
	halfedges[tn_idx].next_idx = t_idx;

	// Update face assignments (t_prev moves to f0, h_prev moves to f1)
	halfedges[h_idx].face_idx = f0;
	halfedges[tp_idx].face_idx = f0;
	halfedges[hn_idx].face_idx = f0;

	halfedges[t_idx].face_idx = f1;
	halfedges[hp_idx].face_idx = f1;
	halfedges[tn_idx].face_idx = f1;

	// Update face halfedge pointers
	faces[f0].halfedge_idx = h_idx;
	faces[f1].halfedge_idx = t_idx;

	// Update vertex halfedge pointers (v0 and v1 might have pointed to h or t)
	vertices[v0].halfedge_idx = tn_idx;
	vertices[v1].halfedge_idx = hn_idx;
	vertices[v2].halfedge_idx = h_idx;
	vertices[v3].halfedge_idx = t_idx;
}


__global__ void kernel_get_edge_lengths(
	Mesh::Edge* edges, 
	Mesh::Halfedge* halfedges, 
	Mesh::Vertex* vertices, 
	float* lengths,
	uint32_t num_edges)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= num_edges) return;

	Mesh::Edge e = edges[index];
	Mesh::Halfedge h = halfedges[e.halfedge_idx];
	Mesh::Halfedge h_twin = halfedges[h.twin_idx];
	Mesh::Vertex v1 = vertices[h.vertex_idx];
	Mesh::Vertex v2 = vertices[h_twin.vertex_idx];

	float dx = (v1.position.x - v2.position.x);
	float dy = (v1.position.y - v2.position.y);
	float dz = (v1.position.z - v2.position.z);

	float length = std::sqrt(dx * dx + dy * dy + dz * dz);
	lengths[index] = length;
	std::printf("edge %d is length %f\n", index, length);
}

/**
 * Populates a mask of size numEdges with the edges
 * which should be collapsed
 */
__global__ void kernel_get_collapse_edges(
	Mesh::Edge* edges, Mesh::Halfedge* halfedges, Mesh::Face* faces,
	float* lengths, uint32_t num_edges, float avg_len, float collapse_factor, int* op_mask) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= num_edges) return;
	op_mask[index] = 0;

	uint32_t h_idx = edges[index].halfedge_idx;
	if (h_idx == INVALID_IDX) return;
	uint32_t t_idx = halfedges[h_idx].twin_idx;
	if (t_idx == INVALID_IDX) return; // boundary edge
	if (faces[halfedges[h_idx].face_idx].boundary) return;
	if (faces[halfedges[t_idx].face_idx].boundary) return;

	op_mask[index] = lengths[index] < avg_len * collapse_factor;
}
/**
 * Collapse edge: merge two endpoints into one (midpoint).
 * The edge and its two adjacent faces are removed (marked invalid).
 *
 * Before:                After:
 *       A                    A
 *      /|\                  / \
 *     / | \                /   \
 *    B--+--C     →        M     (B,C merged into M at B's slot)
 *     \ | /                \   /
 *      \|/                  \ /
 *       D                    D
 */
__global__ void kernel_collapse_edge(
	Mesh::Vertex* vertices,
	Mesh::Edge* edges,
	Mesh::Halfedge* halfedges,
	Mesh::Face* faces,
	int* edge_color_mask,
	int* op_mask,
	uint32_t num_edges,
	int color
) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= num_edges) return;
	if (edge_color_mask[idx] != color) return;
	if (op_mask[idx] != 1) return;

	uint32_t h_idx = edges[idx].halfedge_idx;
	if (h_idx == INVALID_IDX) return;
	uint32_t t_idx = halfedges[h_idx].twin_idx;
	if (t_idx == INVALID_IDX) return; // boundary edge

	// 6 halfedges of the two triangles
	uint32_t hn_idx = halfedges[h_idx].next_idx;   // C→A
	uint32_t hp_idx = halfedges[hn_idx].next_idx;   // A→B
	uint32_t tn_idx = halfedges[t_idx].next_idx;    // B→D
	uint32_t tp_idx = halfedges[tn_idx].next_idx;   // D→C

	// 4 vertices
	uint32_t vB = halfedges[h_idx].vertex_idx;
	uint32_t vC = halfedges[t_idx].vertex_idx;
	uint32_t vA = halfedges[hp_idx].vertex_idx;
	uint32_t vD = halfedges[tp_idx].vertex_idx;

	// 2 faces to remove
	uint32_t f0 = halfedges[h_idx].face_idx;
	uint32_t f1 = halfedges[t_idx].face_idx;

	// Edges on the boundary of the diamond (to be merged)
	uint32_t ehn = halfedges[hn_idx].edge_idx; // edge C-A
	uint32_t ehp = halfedges[hp_idx].edge_idx; // edge A-B
	uint32_t etn = halfedges[tn_idx].edge_idx; // edge B-D
	uint32_t etp = halfedges[tp_idx].edge_idx; // edge D-C

	// Twin halfedges of the 4 outer halfedges
	uint32_t hn_twin = halfedges[hn_idx].twin_idx;
	uint32_t hp_twin = halfedges[hp_idx].twin_idx;
	uint32_t tn_twin = halfedges[tn_idx].twin_idx;
	uint32_t tp_twin = halfedges[tp_idx].twin_idx;

	// Move B to midpoint of B and C
	vertices[vB].position = (vertices[vB].position + vertices[vC].position) * 0.5f;

	// Rewire all halfedges that pointed to C → now point to B
	// Walk around C and redirect
	uint32_t start_he = vertices[vC].halfedge_idx;
	if (start_he != INVALID_IDX) {
		uint32_t he = start_he;
		do {
			halfedges[he].vertex_idx = vB;
			uint32_t tw = halfedges[he].twin_idx;
			if (tw == INVALID_IDX) break;
			he = halfedges[tw].next_idx;
			if (he == INVALID_IDX) break;
		} while (he != start_he);
	}

	// Merge twin pairs: make outer halfedges twins of each other
	// hn and hp's twins become direct twins (removing face f0)
	if (hn_twin != INVALID_IDX) halfedges[hn_twin].twin_idx = hp_twin;
	if (hp_twin != INVALID_IDX) halfedges[hp_twin].twin_idx = hn_twin;
	// Merge one edge: keep ehp, mark ehn as invalid
	if (hn_twin != INVALID_IDX) halfedges[hn_twin].edge_idx = ehp;
	edges[ehn].halfedge_idx = INVALID_IDX;

	// tn and tp's twins become direct twins (removing face f1)
	if (tn_twin != INVALID_IDX) halfedges[tn_twin].twin_idx = tp_twin;
	if (tp_twin != INVALID_IDX) halfedges[tp_twin].twin_idx = tn_twin;
	// Merge one edge: keep etn, mark etp as invalid
	if (tp_twin != INVALID_IDX) halfedges[tp_twin].edge_idx = etn;
	edges[etp].halfedge_idx = INVALID_IDX;

	// Mark the collapsed edge as invalid
	edges[idx].halfedge_idx = INVALID_IDX;

	// Mark vertex C as invalid
	vertices[vC].halfedge_idx = INVALID_IDX;

	// Mark the 6 inner halfedges as invalid
	halfedges[h_idx].vertex_idx = INVALID_IDX;
	halfedges[t_idx].vertex_idx = INVALID_IDX;
	halfedges[hn_idx].vertex_idx = INVALID_IDX;
	halfedges[hp_idx].vertex_idx = INVALID_IDX;
	halfedges[tn_idx].vertex_idx = INVALID_IDX;
	halfedges[tp_idx].vertex_idx = INVALID_IDX;

	// Mark the 2 faces as invalid
	faces[f0].halfedge_idx = INVALID_IDX;
	faces[f1].halfedge_idx = INVALID_IDX;

	// Update vertex halfedge pointers for A, B, D
	vertices[vA].halfedge_idx = (hp_twin != INVALID_IDX) ? hp_twin : hn_twin;
	vertices[vD].halfedge_idx = (tn_twin != INVALID_IDX) ? tn_twin : tp_twin;
	vertices[vB].halfedge_idx = (tn_twin != INVALID_IDX) ? tn_twin : hp_twin;
}

/**
 * Populates a mask of size numEdges with the edges
 * which should be split
 */
__global__ void kernel_get_split_edges(
	Mesh::Edge* edges, Mesh::Halfedge* halfedges, Mesh::Face* faces,
	float* lengths, uint32_t num_edges, float avg_len, float split_factor, int* op_mask) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= num_edges) return;
	op_mask[index] = 0;

	uint32_t h_idx = edges[index].halfedge_idx;
	if (h_idx == INVALID_IDX) return;
	uint32_t t_idx = halfedges[h_idx].twin_idx;
	if (t_idx == INVALID_IDX) return; // boundary edge, can't split here
	if (faces[halfedges[h_idx].face_idx].boundary) return;
	if (faces[halfedges[t_idx].face_idx].boundary) return;

	std::printf("edge %u: length = %f, cmp = %f\n", index, lengths[index], avg_len * split_factor);
	op_mask[index] = lengths[index] > avg_len * split_factor;
	if (op_mask[index]) std::printf("edge %u should be split\n", index);
}

/**
 * Split edge at its midpoint. Creates 1 vertex, 3 edges, 6 halfedges, 2 faces.
 * split_offsets is the exclusive prefix sum of op_mask, so each thread knows
 * where to write its new elements.
 *
 * Before:                After:
 *       A                     A
 *      / \                   /|\
 *     / f0\                 / | \
 *    /     \               /  |  \
 *   B-------C             B---M---C
 *    \ f1  /               \  |  /
 *     \   /                 \ | /
 *      \ /                   \|/
 *       D                     D
 */
__global__ void kernel_split_edge(
	Mesh::Vertex* vertices,
	Mesh::Edge* edges,
	Mesh::Halfedge* halfedges,
	Mesh::Face* faces,
	int* edge_color_mask,
	int* op_mask,
	int* split_offsets,
	uint32_t num_edges,
	uint32_t base_v, uint32_t base_e, uint32_t base_h, uint32_t base_f,
	int color
) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= num_edges) return;
	if (edge_color_mask[idx] != color) return;
	if (op_mask[idx] != 1) return;

	uint32_t h_idx = edges[idx].halfedge_idx;
	if (h_idx == INVALID_IDX) return;
	uint32_t t_idx = halfedges[h_idx].twin_idx;
	if (t_idx == INVALID_IDX) return; // skip boundary edges

	// Original 6 halfedges:
	// Face f0: h(B→C), hn(C→A), hp(A→B)
	// Face f1: t(C→B), tn(B→D), tp(D→C)
	uint32_t hn_idx = halfedges[h_idx].next_idx;   // C→A
	uint32_t hp_idx = halfedges[hn_idx].next_idx;   // A→B
	uint32_t tn_idx = halfedges[t_idx].next_idx;    // B→D
	uint32_t tp_idx = halfedges[tn_idx].next_idx;   // D→C

	// 4 vertices
	uint32_t vB = halfedges[h_idx].vertex_idx;
	uint32_t vC = halfedges[t_idx].vertex_idx;
	uint32_t vA = halfedges[hp_idx].vertex_idx;
	uint32_t vD = halfedges[tp_idx].vertex_idx;

	// 2 original faces
	uint32_t f0 = halfedges[h_idx].face_idx;
	uint32_t f1 = halfedges[t_idx].face_idx;

	// Compute new element indices using prefix sum offset
	int off = split_offsets[idx];

	uint32_t vM       = base_v + off;

	uint32_t eMA_idx  = base_e + off * 3;
	uint32_t eMC_idx  = base_e + off * 3 + 1;
	uint32_t eMD_idx  = base_e + off * 3 + 2;

	uint32_t nMA_idx  = base_h + off * 6;       // M→A
	uint32_t nAM_idx  = base_h + off * 6 + 1;   // A→M
	uint32_t nMC_idx  = base_h + off * 6 + 2;   // M→C
	uint32_t nCM_idx  = base_h + off * 6 + 3;   // C→M
	uint32_t nDM_idx  = base_h + off * 6 + 4;   // D→M
	uint32_t nMD_idx  = base_h + off * 6 + 5;   // M→D

	uint32_t f2       = base_f + off * 2;        // face AMC
	uint32_t f3       = base_f + off * 2 + 1;    // face MDC

	// --- Create new vertex M at midpoint ---
	vertices[vM].position = (vertices[vB].position + vertices[vC].position) * 0.5f;
	vertices[vM].halfedge_idx = t_idx; // t becomes M→B after rewire
	vertices[vM].id = vM;

	// --- Create 3 new edges ---
	edges[eMA_idx].halfedge_idx = nMA_idx;
	edges[eMA_idx].id = eMA_idx;
	edges[eMA_idx].sharp = false;

	edges[eMC_idx].halfedge_idx = nMC_idx;
	edges[eMC_idx].id = eMC_idx;
	edges[eMC_idx].sharp = false;

	edges[eMD_idx].halfedge_idx = nMD_idx;
	edges[eMD_idx].id = eMD_idx;
	edges[eMD_idx].sharp = false;

	// --- Create 2 new faces ---
	faces[f2].halfedge_idx = nAM_idx;
	faces[f2].id = f2;
	faces[f2].boundary = false;

	faces[f3].halfedge_idx = nMD_idx;
	faces[f3].id = f3;
	faces[f3].boundary = false;

	// --- Create 6 new halfedges ---
	// nMA: M→A (in face f0: ABM)
	halfedges[nMA_idx].vertex_idx = vM;
	halfedges[nMA_idx].next_idx = hp_idx;
	halfedges[nMA_idx].twin_idx = nAM_idx;
	halfedges[nMA_idx].edge_idx = eMA_idx;
	halfedges[nMA_idx].face_idx = f0;
	halfedges[nMA_idx].id = nMA_idx;

	// nAM: A→M (in face f2: AMC)
	halfedges[nAM_idx].vertex_idx = vA;
	halfedges[nAM_idx].next_idx = nMC_idx;
	halfedges[nAM_idx].twin_idx = nMA_idx;
	halfedges[nAM_idx].edge_idx = eMA_idx;
	halfedges[nAM_idx].face_idx = f2;
	halfedges[nAM_idx].id = nAM_idx;

	// nMC: M→C (in face f2: AMC)
	halfedges[nMC_idx].vertex_idx = vM;
	halfedges[nMC_idx].next_idx = hn_idx;
	halfedges[nMC_idx].twin_idx = nCM_idx;
	halfedges[nMC_idx].edge_idx = eMC_idx;
	halfedges[nMC_idx].face_idx = f2;
	halfedges[nMC_idx].id = nMC_idx;

	// nCM: C→M (in face f3: MDC)
	halfedges[nCM_idx].vertex_idx = vC;
	halfedges[nCM_idx].next_idx = nMD_idx;
	halfedges[nCM_idx].twin_idx = nMC_idx;
	halfedges[nCM_idx].edge_idx = eMC_idx;
	halfedges[nCM_idx].face_idx = f3;
	halfedges[nCM_idx].id = nCM_idx;

	// nDM: D→M (in face f1: MBD)
	halfedges[nDM_idx].vertex_idx = vD;
	halfedges[nDM_idx].next_idx = t_idx;
	halfedges[nDM_idx].twin_idx = nMD_idx;
	halfedges[nDM_idx].edge_idx = eMD_idx;
	halfedges[nDM_idx].face_idx = f1;
	halfedges[nDM_idx].id = nDM_idx;

	// nMD: M→D (in face f3: MDC)
	halfedges[nMD_idx].vertex_idx = vM;
	halfedges[nMD_idx].next_idx = tp_idx;
	halfedges[nMD_idx].twin_idx = nDM_idx;
	halfedges[nMD_idx].edge_idx = eMD_idx;
	halfedges[nMD_idx].face_idx = f3;
	halfedges[nMD_idx].id = nMD_idx;

	// --- Modify existing halfedges ---
	// h (B→M, was B→C): update next
	halfedges[h_idx].next_idx = nMA_idx;
	// face stays f0, vertex stays vB, twin stays t_idx, edge stays idx

	// t (M→B, was C→B): update vertex and next
	halfedges[t_idx].vertex_idx = vM;
	halfedges[t_idx].next_idx = tn_idx;
	// face stays f1, twin stays h_idx, edge stays idx

	// hn (C→A): move to face f2, update next
	halfedges[hn_idx].next_idx = nAM_idx;
	halfedges[hn_idx].face_idx = f2;

	// hp (A→B): next stays h_idx (unchanged)

	// tn (B→D): update next
	halfedges[tn_idx].next_idx = nDM_idx;
	// face stays f1

	// tp (D→C): move to face f3, update next
	halfedges[tp_idx].next_idx = nCM_idx;
	halfedges[tp_idx].face_idx = f3;

	// --- Update face halfedge pointers ---
	faces[f0].halfedge_idx = h_idx;
	faces[f1].halfedge_idx = t_idx;

	// --- Update vertex halfedge pointers ---
	// vC's halfedge might have been t_idx (C→B), but now t leaves from M
	vertices[vC].halfedge_idx = nCM_idx;
	// vB, vA, vD still have valid outgoing halfedges
}

//isotropic_remesh: improves mesh quality through local operations.
// Do note that this requires a working implementation of EdgeSplit, EdgeFlip, and EdgeCollapse
void CudaRemesher::isotropic_remesh(Isotropic_Remesh_Params const &params) {
	dim3 blockDim;
	dim3 gridDim;

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
	 *			ii.	Repeat the tangential smoothing part params.smoothing_iters times.
	 *	5. Repeat steps 1-4 `num_iters` times.
	 */

	//NOTE: many of the steps in this function will be modifying the element
	//      lists they are looping over. Take care to avoid use-after-free
	//      or infinite-loop problems.

	for (int t = 0; t < params.num_iters; t++) {
		std::printf("iteration %d of remeshing\n", t);
		blockDim = dim3(256);
		gridDim = dim3((numEdges + blockDim.x - 1 ) / blockDim.x);
		cudaMemset(edge_color_mask, -1, sizeof(int) * numEdges);
		bool h_done = false;
		while (!h_done) {
			h_done = true;
			cudaMemcpy(d_coloring_done, &h_done, sizeof(bool), cudaMemcpyHostToDevice);
			kernel_color_edges<<<gridDim, blockDim>>>(cudaDeviceEdges, cudaDeviceHalfedges, cudaDeviceVertices, numEdges, edge_color_mask, edge_priorities, d_coloring_done);
			cudaMemcpy(&h_done, d_coloring_done, sizeof(bool), cudaMemcpyDeviceToHost);
		}
		
		// color_mask holds color of corresponding edge
		// get max color in color_mask
		int* cuda_max_color = thrust::max_element(thrust::device, edge_color_mask, edge_color_mask + numEdges);
		int max_color;
		cudaMemcpy(&max_color, cuda_max_color, sizeof(int), cudaMemcpyDeviceToHost);
		
		kernel_get_flip_edges<<<gridDim, blockDim>>>(cudaDeviceEdges, cudaDeviceHalfedges, cudaDeviceVertices, cudaDeviceFaces, numEdges, edge_op_mask);
		for (int c = 0; c <= max_color; c++) {
			std::printf("Flipping edges of color %d\n", c);
			// flips all edges with color c if flipping them increases regular-ness
			kernel_flip_edge<<<gridDim, blockDim>>>(cudaDeviceEdges, cudaDeviceHalfedges, cudaDeviceVertices, cudaDeviceFaces, numEdges, edge_color_mask, edge_op_mask, c);
		}

		kernel_get_edge_lengths<<<gridDim, blockDim>>>(cudaDeviceEdges, cudaDeviceHalfedges, cudaDeviceVertices, edge_lengths, numEdges);
		cudaDeviceSynchronize();

		float avg_len = thrust::reduce(thrust::device, edge_lengths, edge_lengths + numEdges, 0.0f, thrust::plus<float>()) / std::max(1U, numEdges);
		std::printf("average length is %f\n", avg_len);

		// Recolor edges after flip (connectivity changed)
		cudaMemset(edge_color_mask, -1, sizeof(int) * numEdges);
		h_done = false;
		while (!h_done) {
			h_done = true;
			cudaMemcpy(d_coloring_done, &h_done, sizeof(bool), cudaMemcpyHostToDevice);
			kernel_color_edges<<<gridDim, blockDim>>>(cudaDeviceEdges, cudaDeviceHalfedges, cudaDeviceVertices, numEdges, edge_color_mask, edge_priorities, d_coloring_done);
			cudaMemcpy(&h_done, d_coloring_done, sizeof(bool), cudaMemcpyDeviceToHost);
		}

		// === SPLIT ===
		kernel_get_split_edges<<<gridDim, blockDim>>>(cudaDeviceEdges, cudaDeviceHalfedges, cudaDeviceFaces, edge_lengths, numEdges, avg_len, params.split_factor, edge_op_mask);
		cudaDeviceSynchronize();

		// Compute prefix sum of op_mask to get per-edge offset
		cudaMalloc(&split_offsets, sizeof(int) * numEdges);
		thrust::exclusive_scan(thrust::device, edge_op_mask, edge_op_mask + numEdges, split_offsets);

		// Total number of splits = last offset + last op_mask value
		int last_offset, last_mask;
		cudaMemcpy(&last_offset, split_offsets + numEdges - 1, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&last_mask, edge_op_mask + numEdges - 1, sizeof(int), cudaMemcpyDeviceToHost);
		int total_splits = last_offset + last_mask;
		std::printf("total splits = %d\n", total_splits);

		if (total_splits > 0) {
			// Compute new counts
			uint32_t newV = numVertices + total_splits;
			uint32_t newE = numEdges + total_splits * 3;
			uint32_t newH = numHalfedges + total_splits * 6;
			uint32_t newF = numFaces + total_splits * 2;

			// Reallocate arrays to accommodate new elements
			Mesh::Vertex* newVertices;
			Mesh::Edge* newEdges;
			Mesh::Halfedge* newHalfedges;
			Mesh::Face* newFaces;
			cudaMalloc(&newVertices, sizeof(Mesh::Vertex) * newV);
			cudaMalloc(&newEdges, sizeof(Mesh::Edge) * newE);
			cudaMalloc(&newHalfedges, sizeof(Mesh::Halfedge) * newH);
			cudaMalloc(&newFaces, sizeof(Mesh::Face) * newF);
			cudaMemcpy(newVertices, cudaDeviceVertices, sizeof(Mesh::Vertex) * numVertices, cudaMemcpyDeviceToDevice);
			cudaMemcpy(newEdges, cudaDeviceEdges, sizeof(Mesh::Edge) * numEdges, cudaMemcpyDeviceToDevice);
			cudaMemcpy(newHalfedges, cudaDeviceHalfedges, sizeof(Mesh::Halfedge) * numHalfedges, cudaMemcpyDeviceToDevice);
			cudaMemcpy(newFaces, cudaDeviceFaces, sizeof(Mesh::Face) * numFaces, cudaMemcpyDeviceToDevice);
			cudaFree(cudaDeviceVertices);
			cudaFree(cudaDeviceEdges);
			cudaFree(cudaDeviceHalfedges);
			cudaFree(cudaDeviceFaces);
			cudaDeviceVertices = newVertices;
			cudaDeviceEdges = newEdges;
			cudaDeviceHalfedges = newHalfedges;
			cudaDeviceFaces = newFaces;

			cuda_max_color = thrust::max_element(thrust::device, edge_color_mask, edge_color_mask + numEdges);
			cudaMemcpy(&max_color, cuda_max_color, sizeof(int), cudaMemcpyDeviceToHost);

			for (int c = 0; c <= max_color; c++) {
				std::printf("Splitting edges of color %d\n", c);
				kernel_split_edge<<<gridDim, blockDim>>>(
					cudaDeviceVertices, cudaDeviceEdges, cudaDeviceHalfedges, cudaDeviceFaces,
					edge_color_mask, edge_op_mask, split_offsets,
					numEdges, numVertices, numEdges, numHalfedges, numFaces, c);
				cudaDeviceSynchronize();
			}

			// Update counts
			numVertices = newV;
			numEdges = newE;
			numHalfedges = newH;
			numFaces = newF;

			// Reallocate edge-sized arrays for new count
			cudaFree(edge_lengths);
			cudaFree(edge_color_mask);
			cudaFree(edge_op_mask);
			cudaFree(edge_priorities);
			cudaMalloc(&edge_lengths, sizeof(float) * numEdges);
			cudaMalloc(&edge_color_mask, sizeof(int) * numEdges);
			cudaMalloc(&edge_op_mask, sizeof(int) * numEdges);
			// Regenerate edge priorities for new edges
			std::vector<int> h_ep(numEdges);
			for (uint32_t i = 0; i < numEdges; i++) h_ep[i] = rand();
			cudaMalloc(&edge_priorities, sizeof(int) * numEdges);
			cudaMemcpy(edge_priorities, h_ep.data(), sizeof(int) * numEdges, cudaMemcpyHostToDevice);

			// Update gridDim for new edge count
			gridDim = dim3((numEdges + blockDim.x - 1) / blockDim.x);
		}
		cudaFree(split_offsets);
		split_offsets = NULL;

		// === COLLAPSE ===
		// Recompute edge lengths (split may have changed the mesh)
		kernel_get_edge_lengths<<<gridDim, blockDim>>>(cudaDeviceEdges, cudaDeviceHalfedges, cudaDeviceVertices, edge_lengths, numEdges);
		cudaDeviceSynchronize();

		avg_len = thrust::reduce(thrust::device, edge_lengths, edge_lengths + numEdges, 0.0f, thrust::plus<float>()) / std::max(1U, numEdges);
		std::printf("average length after split is %f\n", avg_len);

		// Recolor edges (split changed connectivity)
		cudaMemset(edge_color_mask, -1, sizeof(int) * numEdges);
		h_done = false;
		while (!h_done) {
			h_done = true;
			cudaMemcpy(d_coloring_done, &h_done, sizeof(bool), cudaMemcpyHostToDevice);
			kernel_color_edges<<<gridDim, blockDim>>>(cudaDeviceEdges, cudaDeviceHalfedges, cudaDeviceVertices, numEdges, edge_color_mask, edge_priorities, d_coloring_done);
			cudaMemcpy(&h_done, d_coloring_done, sizeof(bool), cudaMemcpyDeviceToHost);
		}

		kernel_get_collapse_edges<<<gridDim, blockDim>>>(cudaDeviceEdges, cudaDeviceHalfedges, cudaDeviceFaces, edge_lengths, numEdges, avg_len, params.collapse_factor, edge_op_mask);
		cudaDeviceSynchronize();

		cuda_max_color = thrust::max_element(thrust::device, edge_color_mask, edge_color_mask + numEdges);
		cudaMemcpy(&max_color, cuda_max_color, sizeof(int), cudaMemcpyDeviceToHost);

		for (int c = 0; c <= max_color; c++) {
			std::printf("Collapsing edges of color %d\n", c);
			kernel_collapse_edge<<<gridDim, blockDim>>>(
				cudaDeviceVertices, cudaDeviceEdges, cudaDeviceHalfedges, cudaDeviceFaces,
				edge_color_mask, edge_op_mask, numEdges, c);
			cudaDeviceSynchronize();
		}

		gridDim = dim3((numVertices + blockDim.x - 1) / blockDim.x);
		// Color vertices using Jones-Plassmann algorithm
		cudaMemset(vertex_color_mask, -1, sizeof(int) * numVertices); // reset all to -1 (uncolored)
		h_done = false;
		while (!h_done) {
			h_done = true;
			cudaMemcpy(d_coloring_done, &h_done, sizeof(bool), cudaMemcpyHostToDevice);
			kernel_color_vertices<<<gridDim, blockDim>>>(cudaDeviceVertices, cudaDeviceHalfedges, numVertices, vertex_color_mask, vertex_priorities, d_coloring_done);
			cudaMemcpy(&h_done, d_coloring_done, sizeof(bool), cudaMemcpyDeviceToHost);
		}

		cuda_max_color = thrust::max_element(thrust::device, vertex_color_mask, vertex_color_mask + numVertices);
		cudaMemcpy(&max_color, cuda_max_color, sizeof(int), cudaMemcpyDeviceToHost);
		for (int i = 0; i < params.smoothing_iters; i++) {
			std::printf("iteration %d of vertex smoothing\n", i);
			for (int c = 0; c <= max_color; c++) {
				// smooth all vertices of each color
				std::printf("Smoothing vertices of color %d\n", c);
				kernel_smooth_vertex<<<gridDim, blockDim>>>(cudaDeviceVertices, cudaDeviceEdges,
					cudaDeviceHalfedges, cudaDeviceFaces, vertex_color_mask, vertex_normals, vertex_pos,
					numVertices, numEdges, numHalfedges, numFaces, params.smoothing_step, c);
					cudaDeviceSynchronize();
			}
			// update vertex positions
			kernel_update_vertex_pos<<<gridDim, blockDim>>>(cudaDeviceVertices, vertex_pos, numVertices);
			cudaDeviceSynchronize();
		}
	}
}