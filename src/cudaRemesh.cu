#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "thrust/scan.h"
#include "thrust/reduce.h"
#include "thrust/functional.h"

#include <cstdlib>
#include <chrono>

#include "mesh.h"
#include "cudaRemesh.h"
#include "vec3.h"

// Set VERBOSE=1 to re-enable chatty per-element/per-color logging.
#ifndef VERBOSE
#define VERBOSE 0
#endif

#if VERBOSE
#define VPRINTF(...) do { std::printf(__VA_ARGS__); std::fflush(stdout); } while(0)
#define DPRINTF(...) std::printf(__VA_ARGS__)
#else
#define VPRINTF(...) do {} while(0)
#define DPRINTF(...) do {} while(0)
#endif

#define CUDA_CHECK(label) do { \
	cudaError_t _s = cudaDeviceSynchronize(); \
	cudaError_t _l = cudaGetLastError(); \
	if (_s != cudaSuccess || _l != cudaSuccess) { \
		if (verbose) { \
			std::printf("[CUDA ERROR @ %s] sync=%s last=%s\n", label, cudaGetErrorString(_s), cudaGetErrorString(_l)); \
			std::fflush(stdout); \
		} \
		std::abort(); \
	} else { \
		VPRINTF("[ok @ %s]\n", label); \
	} \
} while(0)

void cuda_clear_last_error() { cudaGetLastError(); }

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
	VPRINTF("malloc'd elements\n");

	cudaMemcpy(cudaDeviceVertices, _mesh.vertices.data(), sizeof(Mesh::Vertex) * _mesh.vertices.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaDeviceEdges, _mesh.edges.data(), sizeof(Mesh::Edge) * _mesh.edges.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaDeviceHalfedges, _mesh.halfedges.data(), sizeof(Mesh::Halfedge) * _mesh.halfedges.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaDeviceFaces, _mesh.faces.data(), sizeof(Mesh::Face) * _mesh.faces.size(), cudaMemcpyHostToDevice);
	VPRINTF("memcpy elements\n");

	numVertices = _mesh.vertices.size();
	numEdges = _mesh.edges.size();
	numHalfedges = _mesh.halfedges.size();
	numFaces = _mesh.faces.size();

	cudaMalloc(&edge_lengths, sizeof(float) * numEdges);
	cudaMalloc(&edge_color_mask, sizeof(int) * numEdges);
	cudaMalloc(&edge_op_mask, sizeof(int) * numEdges);
	cudaMalloc(&vertex_color_mask, sizeof(int) * numVertices);
	cudaMalloc(&vertex_pos, sizeof(Vec3) * numVertices);
	cudaMalloc(&vertex_normals, sizeof(Vec3) * numVertices);
	VPRINTF("malloc'd masks\n");

	std::vector<int> h_priorities(numVertices);
	for (uint32_t i = 0; i < numVertices; i++) h_priorities[i] = rand();
	cudaMalloc(&vertex_priorities, sizeof(int) * numVertices);
	cudaMemcpy(vertex_priorities, h_priorities.data(), sizeof(int) * numVertices, cudaMemcpyHostToDevice);

	std::vector<int> h_edge_priorities(numEdges);
	for (uint32_t i = 0; i < numEdges; i++) h_edge_priorities[i] = rand();
	cudaMalloc(&edge_priorities, sizeof(int) * numEdges);
	cudaMemcpy(edge_priorities, h_edge_priorities.data(), sizeof(int) * numEdges, cudaMemcpyHostToDevice);

	cudaMalloc(&d_coloring_done, sizeof(bool));
}

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

// Grid-stride loop pattern: each thread processes multiple elements when
// total threads < N.  The do{}while(0) wrapper lets us use break instead of
// return for early exits inside the loop body.

__global__ void kernel_color_vertices(
	Mesh::Vertex* vertices, Mesh::Halfedge* halfedges,
	uint32_t num_vertices, int* color_mask, int* priorities, bool* done)
{
	for (int idx = blockDim.x * blockIdx.x + threadIdx.x;
	     (uint32_t)idx < num_vertices;
	     idx += blockDim.x * gridDim.x)
	{ do {
		if (color_mask[idx] != -1) break;

		bool is_local_max = true;
		uint32_t used_colors = 0;

		uint32_t start_he = vertices[idx].halfedge_idx;
		if (start_he == INVALID_IDX) { color_mask[idx] = 0; break; }
		if (halfedges[start_he].vertex_idx == INVALID_IDX) { color_mask[idx] = 0; break; }

		uint32_t he = start_he;
		int guard = 0;
		do {
			uint32_t twin_he = halfedges[he].twin_idx;
			if (twin_he == INVALID_IDX) break;
			uint32_t neighbor = halfedges[twin_he].vertex_idx;
			if (neighbor == INVALID_IDX) break;
			if (color_mask[neighbor] == -1) {
				if (priorities[neighbor] > priorities[idx]) { is_local_max = false; break; }
				if (priorities[neighbor] == priorities[idx] && neighbor > (uint32_t)idx) { is_local_max = false; break; }
			} else {
				if (color_mask[neighbor] < 32) used_colors |= (1u << color_mask[neighbor]);
			}
			he = halfedges[twin_he].next_idx;
			if (he == INVALID_IDX) break;
			if (++guard > 1024) break;
		} while (he != start_he);

		if (is_local_max) {
			int color = 0;
			while (used_colors & (1u << color)) color++;
			color_mask[idx] = color;
		} else {
			*done = false;
		}
	} while(0); }
}

__global__ void kernel_color_edges(
	Mesh::Edge* edges, Mesh::Halfedge* halfedges, Mesh::Vertex* vertices,
	uint32_t num_edges, int* color_mask, int* priorities, bool* done)
{
	for (int idx = blockDim.x * blockIdx.x + threadIdx.x;
	     (uint32_t)idx < num_edges;
	     idx += blockDim.x * gridDim.x)
	{ do {
		if (color_mask[idx] != -1) break;

		bool is_local_max = true;
		uint32_t used_colors = 0;

		uint32_t h_idx = edges[idx].halfedge_idx;
		if (h_idx == INVALID_IDX) { color_mask[idx] = 0; break; }

		uint32_t v1 = halfedges[h_idx].vertex_idx;
		uint32_t twin_idx = halfedges[h_idx].twin_idx;
		uint32_t v2 = (twin_idx != INVALID_IDX) ? halfedges[twin_idx].vertex_idx : INVALID_IDX;
		if (v1 == INVALID_IDX) { color_mask[idx] = 0; break; }

		uint32_t start_he = vertices[v1].halfedge_idx;
		if (start_he != INVALID_IDX && halfedges[start_he].vertex_idx != INVALID_IDX) {
			uint32_t he = start_he;
			int guard1 = 0;
			do {
				uint32_t neighbor_edge = halfedges[he].edge_idx;
				if (neighbor_edge != (uint32_t)idx && neighbor_edge != INVALID_IDX) {
					if (color_mask[neighbor_edge] == -1) {
						if (priorities[neighbor_edge] > priorities[idx] ||
							(priorities[neighbor_edge] == priorities[idx] && neighbor_edge > (uint32_t)idx)) {
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
				if (++guard1 > 1024) break;
			} while (he != start_he);
		}

		if (is_local_max && v2 != INVALID_IDX) {
			start_he = vertices[v2].halfedge_idx;
			if (start_he != INVALID_IDX && halfedges[start_he].vertex_idx != INVALID_IDX) {
				uint32_t he = start_he;
				int guard2 = 0;
				do {
					uint32_t neighbor_edge = halfedges[he].edge_idx;
					if (neighbor_edge != (uint32_t)idx && neighbor_edge != INVALID_IDX) {
						if (color_mask[neighbor_edge] == -1) {
							if (priorities[neighbor_edge] > priorities[idx] ||
								(priorities[neighbor_edge] == priorities[idx] && neighbor_edge > (uint32_t)idx)) {
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
					if (++guard2 > 1024) break;
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
	} while(0); }
}

__global__ void kernel_get_vertex_normals(
	Mesh::Vertex* vertices,
	Mesh::Halfedge* halfedges,
	Mesh::Face* faces,
	Vec3* vertex_normals,
	uint32_t num_vertices,
	uint32_t num_halfedges
) {
	for (int index = blockDim.x * blockIdx.x + threadIdx.x;
	     (uint32_t)index < num_vertices;
	     index += blockDim.x * gridDim.x)
	{ do {
		Vec3 n = Vec3(0.0f, 0.0f, 0.0f);
		Vec3 pi = vertices[index].position;
		uint32_t h_idx = vertices[index].halfedge_idx;
		if (h_idx == INVALID_IDX) { vertex_normals[index] = Vec3(0.0f, 0.0f, 0.0f); break; }
		if (halfedges[h_idx].vertex_idx == INVALID_IDX) { vertex_normals[index] = Vec3(0.0f, 0.0f, 0.0f); break; }

		uint32_t curr_idx = h_idx;
		int guard = 0;
		do {
			Mesh::Halfedge h = halfedges[curr_idx];
			uint32_t hn = h.next_idx;
			if (hn == INVALID_IDX) break;
			uint32_t hnn = halfedges[hn].next_idx;
			if (hnn == INVALID_IDX) break;
			uint32_t pk_v = halfedges[hn].vertex_idx;
			uint32_t pj_v = halfedges[hnn].vertex_idx;
			if (pk_v != INVALID_IDX && pj_v != INVALID_IDX && !faces[h.face_idx].boundary) {
				Vec3 pk = vertices[pk_v].position;
				Vec3 pj = vertices[pj_v].position;
				n += cross(pk - pi, pj - pi);
			}
			uint32_t tw = h.twin_idx;
			if (tw == INVALID_IDX) break;
			curr_idx = halfedges[tw].next_idx;
			if (curr_idx == INVALID_IDX) break;
			if (++guard > 1024) break;
		} while (curr_idx != h_idx);

		float len = std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
		if (len > 1e-12f) n = n * (1.0f / len);
		else n = Vec3(0.0f, 0.0f, 0.0f);
		vertex_normals[index] = n;
	} while(0); }
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
	for (int index = blockDim.x * blockIdx.x + threadIdx.x;
	     (uint32_t)index < num_vertices;
	     index += blockDim.x * gridDim.x)
	{ do {
		if (vertex_color_mask[index] != color) break;

		Mesh::Vertex v = vertices[index];
		Vec3 center;

		uint32_t h_idx = v.halfedge_idx;
		if (h_idx == INVALID_IDX) break;
		if (halfedges[h_idx].vertex_idx == INVALID_IDX) break;
		uint32_t curr_idx = h_idx;

		uint32_t count = 0;
		int sguard = 0;
		do {
			Mesh::Halfedge h = halfedges[curr_idx];
			uint32_t tw = h.twin_idx;
			if (tw == INVALID_IDX) break;
			uint32_t nb_idx = halfedges[tw].vertex_idx;
			if (nb_idx == INVALID_IDX) break;
			Mesh::Vertex neighbor = vertices[nb_idx];
			center += neighbor.position;
			count++;
			curr_idx = halfedges[tw].next_idx;
			if (curr_idx == INVALID_IDX) break;
			if (++sguard > 1024) break;
		} while (curr_idx != h_idx);

		if (count == 0) break;
		center /= count;

		center = v.position + smoothing_factor * (center - v.position);
		Vec3 normal = vertex_normals[index];
		center = center - dot(normal, center) * normal;
		DPRINTF("vertex %d: (%f %f %f) -> (%f %f %f)\n", index, v.position.x, v.position.y, v.position.z, center.x, center.y, center.z);
		vertex_pos[index] = center;
	} while(0); }
}

__global__ void kernel_update_vertex_pos(
	Mesh::Vertex* vertices,
	Vec3* vertex_pos,
	uint32_t num_vertices
) {
	for (int index = blockDim.x * blockIdx.x + threadIdx.x;
	     (uint32_t)index < num_vertices;
	     index += blockDim.x * gridDim.x)
	{
		vertices[index].position = vertex_pos[index];
	}
}

__global__ void kernel_init_vertex_pos(
	Mesh::Vertex* vertices,
	Vec3* vertex_pos,
	uint32_t num_vertices
) {
	for (int index = blockDim.x * blockIdx.x + threadIdx.x;
	     (uint32_t)index < num_vertices;
	     index += blockDim.x * gridDim.x)
	{
		vertex_pos[index] = vertices[index].position;
	}
}

__device__ uint32_t vertex_degree(Mesh::Vertex* vertices, Mesh::Halfedge* halfedges, uint32_t v_idx) {
	uint32_t start_he = vertices[v_idx].halfedge_idx;
	if (start_he == INVALID_IDX) return 0;
	if (halfedges[start_he].vertex_idx == INVALID_IDX) return 0;
	uint32_t he = start_he;
	uint32_t deg = 0;
	int dguard = 0;
	do {
		deg++;
		uint32_t tw = halfedges[he].twin_idx;
		if (tw == INVALID_IDX) { deg++; break; }
		he = halfedges[tw].next_idx;
		if (he == INVALID_IDX) break;
		if (++dguard > 1024) break;
	} while (he != start_he);
	return deg;
}

__global__ void kernel_get_flip_edges(
	Mesh::Edge* edges, Mesh::Halfedge* halfedges, Mesh::Vertex* vertices, Mesh::Face* faces,
	uint32_t num_edges, int* op_mask)
{
	for (int idx = blockDim.x * blockIdx.x + threadIdx.x;
	     (uint32_t)idx < num_edges;
	     idx += blockDim.x * gridDim.x)
	{ do {
		op_mask[idx] = 0;

		uint32_t h_idx = edges[idx].halfedge_idx;
		if (h_idx == INVALID_IDX) break;
		if (halfedges[h_idx].vertex_idx == INVALID_IDX) break;

		uint32_t t_idx = halfedges[h_idx].twin_idx;
		if (t_idx == INVALID_IDX) break;
		if (halfedges[t_idx].vertex_idx == INVALID_IDX) break;

		if (faces[halfedges[h_idx].face_idx].boundary) break;
		if (faces[halfedges[t_idx].face_idx].boundary) break;

		uint32_t vB = halfedges[h_idx].vertex_idx;
		uint32_t vD = halfedges[t_idx].vertex_idx;
		uint32_t hn = halfedges[h_idx].next_idx;
		uint32_t tn = halfedges[t_idx].next_idx;
		if (hn == INVALID_IDX || tn == INVALID_IDX) break;
		uint32_t hnn = halfedges[hn].next_idx;
		uint32_t tnn = halfedges[tn].next_idx;
		if (hnn == INVALID_IDX || tnn == INVALID_IDX) break;
		uint32_t vA = halfedges[hnn].vertex_idx;
		uint32_t vC = halfedges[tnn].vertex_idx;
		if (vA == INVALID_IDX || vC == INVALID_IDX || vB == INVALID_IDX || vD == INVALID_IDX) break;

		int degA = vertex_degree(vertices, halfedges, vA);
		int degB = vertex_degree(vertices, halfedges, vB);
		int degC = vertex_degree(vertices, halfedges, vC);
		int degD = vertex_degree(vertices, halfedges, vD);

		int dev_before = abs(degA - 6) + abs(degB - 6) + abs(degC - 6) + abs(degD - 6);
		int dev_after  = abs(degA + 1 - 6) + abs(degB - 1 - 6) + abs(degC + 1 - 6) + abs(degD - 1 - 6);

		op_mask[idx] = (dev_after < dev_before) ? 1 : 0;
	} while(0); }
}

__global__ void kernel_flip_edge(
	Mesh::Edge* edges, Mesh::Halfedge* halfedges, Mesh::Vertex* vertices, Mesh::Face* faces,
	uint32_t num_edges, int* edge_color_mask, int* op_mask, int color)
{
	for (int idx = blockDim.x * blockIdx.x + threadIdx.x;
	     (uint32_t)idx < num_edges;
	     idx += blockDim.x * gridDim.x)
	{ do {
		if (edge_color_mask[idx] != color) break;
		if (op_mask[idx] != 1) break;

		uint32_t h_idx = edges[idx].halfedge_idx;
		if (h_idx == INVALID_IDX) break;
		if (halfedges[h_idx].vertex_idx == INVALID_IDX) break;
		uint32_t t_idx = halfedges[h_idx].twin_idx;
		if (t_idx == INVALID_IDX) break;
		if (halfedges[t_idx].vertex_idx == INVALID_IDX) break;

		uint32_t hn_idx = halfedges[h_idx].next_idx;
		uint32_t hp_idx = halfedges[hn_idx].next_idx;
		uint32_t tn_idx = halfedges[t_idx].next_idx;
		uint32_t tp_idx = halfedges[tn_idx].next_idx;

		uint32_t v0 = halfedges[h_idx].vertex_idx;
		uint32_t v1 = halfedges[t_idx].vertex_idx;
		uint32_t v2 = halfedges[hp_idx].vertex_idx;
		uint32_t v3 = halfedges[tp_idx].vertex_idx;

		uint32_t f0 = halfedges[h_idx].face_idx;
		uint32_t f1 = halfedges[t_idx].face_idx;

		halfedges[h_idx].vertex_idx = v2;
		halfedges[t_idx].vertex_idx = v3;

		halfedges[h_idx].next_idx = tp_idx;
		halfedges[tp_idx].next_idx = hn_idx;
		halfedges[hn_idx].next_idx = h_idx;

		halfedges[t_idx].next_idx = hp_idx;
		halfedges[hp_idx].next_idx = tn_idx;
		halfedges[tn_idx].next_idx = t_idx;

		halfedges[h_idx].face_idx  = f0;
		halfedges[tp_idx].face_idx = f0;
		halfedges[hn_idx].face_idx = f0;

		halfedges[t_idx].face_idx  = f1;
		halfedges[hp_idx].face_idx = f1;
		halfedges[tn_idx].face_idx = f1;

		faces[f0].halfedge_idx = h_idx;
		faces[f1].halfedge_idx = t_idx;

		vertices[v0].halfedge_idx = tn_idx;
		vertices[v1].halfedge_idx = hn_idx;
		vertices[v2].halfedge_idx = h_idx;
		vertices[v3].halfedge_idx = t_idx;
	} while(0); }
}

__global__ void kernel_get_edge_lengths(
	Mesh::Edge* edges,
	Mesh::Halfedge* halfedges,
	Mesh::Vertex* vertices,
	float* lengths,
	uint32_t num_edges)
{
	for (int index = blockDim.x * blockIdx.x + threadIdx.x;
	     (uint32_t)index < num_edges;
	     index += blockDim.x * gridDim.x)
	{ do {
		Mesh::Edge e = edges[index];
		if (e.halfedge_idx == INVALID_IDX) { lengths[index] = 0.0f; break; }
		Mesh::Halfedge h = halfedges[e.halfedge_idx];
		if (h.twin_idx == INVALID_IDX || h.vertex_idx == INVALID_IDX) { lengths[index] = 0.0f; break; }
		Mesh::Halfedge h_twin = halfedges[h.twin_idx];
		if (h_twin.vertex_idx == INVALID_IDX) { lengths[index] = 0.0f; break; }
		Mesh::Vertex v1 = vertices[h.vertex_idx];
		Mesh::Vertex v2 = vertices[h_twin.vertex_idx];

		float dx = v1.position.x - v2.position.x;
		float dy = v1.position.y - v2.position.y;
		float dz = v1.position.z - v2.position.z;
		lengths[index] = std::sqrt(dx*dx + dy*dy + dz*dz);
		DPRINTF("edge %d is length %f\n", index, lengths[index]);
	} while(0); }
}

__global__ void kernel_get_collapse_edges(
	Mesh::Edge* edges, Mesh::Halfedge* halfedges, Mesh::Face* faces,
	float* lengths, uint32_t num_edges, float avg_len, float collapse_factor, int* op_mask)
{
	for (int index = blockDim.x * blockIdx.x + threadIdx.x;
	     (uint32_t)index < num_edges;
	     index += blockDim.x * gridDim.x)
	{ do {
		op_mask[index] = 0;

		uint32_t h_idx = edges[index].halfedge_idx;
		if (h_idx == INVALID_IDX) break;
		if (halfedges[h_idx].vertex_idx == INVALID_IDX) break;
		uint32_t t_idx = halfedges[h_idx].twin_idx;
		if (t_idx == INVALID_IDX) break;
		if (halfedges[t_idx].vertex_idx == INVALID_IDX) break;
		if (faces[halfedges[h_idx].face_idx].boundary) break;
		if (faces[halfedges[t_idx].face_idx].boundary) break;

		op_mask[index] = lengths[index] < avg_len * collapse_factor;
	} while(0); }
}

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
	for (int idx = blockDim.x * blockIdx.x + threadIdx.x;
	     (uint32_t)idx < num_edges;
	     idx += blockDim.x * gridDim.x)
	{ do {
		if (edge_color_mask[idx] != color) break;
		if (op_mask[idx] != 1) break;

		uint32_t h_idx = edges[idx].halfedge_idx;
		if (h_idx == INVALID_IDX) break;
		if (halfedges[h_idx].vertex_idx == INVALID_IDX) break;
		uint32_t t_idx = halfedges[h_idx].twin_idx;
		if (t_idx == INVALID_IDX) break;
		if (halfedges[t_idx].vertex_idx == INVALID_IDX) break;

		uint32_t hn_idx = halfedges[h_idx].next_idx;
		if (hn_idx == INVALID_IDX) break;
		uint32_t hp_idx = halfedges[hn_idx].next_idx;
		if (hp_idx == INVALID_IDX) break;
		uint32_t tn_idx = halfedges[t_idx].next_idx;
		if (tn_idx == INVALID_IDX) break;
		uint32_t tp_idx = halfedges[tn_idx].next_idx;
		if (tp_idx == INVALID_IDX) break;

		if (halfedges[hn_idx].vertex_idx == INVALID_IDX) break;
		if (halfedges[hp_idx].vertex_idx == INVALID_IDX) break;
		if (halfedges[tn_idx].vertex_idx == INVALID_IDX) break;
		if (halfedges[tp_idx].vertex_idx == INVALID_IDX) break;

		uint32_t vB = halfedges[h_idx].vertex_idx;
		uint32_t vC = halfedges[t_idx].vertex_idx;
		uint32_t vA = halfedges[hp_idx].vertex_idx;
		uint32_t vD = halfedges[tp_idx].vertex_idx;
		if (vA == INVALID_IDX || vB == INVALID_IDX || vC == INVALID_IDX || vD == INVALID_IDX) break;

		uint32_t f0 = halfedges[h_idx].face_idx;
		uint32_t f1 = halfedges[t_idx].face_idx;

		uint32_t ehn = halfedges[hn_idx].edge_idx;
		uint32_t ehp = halfedges[hp_idx].edge_idx;
		uint32_t etn = halfedges[tn_idx].edge_idx;
		uint32_t etp = halfedges[tp_idx].edge_idx;

		uint32_t hn_twin = halfedges[hn_idx].twin_idx;
		uint32_t hp_twin = halfedges[hp_idx].twin_idx;
		uint32_t tn_twin = halfedges[tn_idx].twin_idx;
		uint32_t tp_twin = halfedges[tp_idx].twin_idx;

		vertices[vB].position = (vertices[vB].position + vertices[vC].position) * 0.5f;

		uint32_t start_he = vertices[vC].halfedge_idx;
		if (start_he != INVALID_IDX && halfedges[start_he].vertex_idx == vC) {
			uint32_t he = start_he;
			int cguard = 0;
			do {
				halfedges[he].vertex_idx = vB;
				uint32_t tw = halfedges[he].twin_idx;
				if (tw == INVALID_IDX) break;
				he = halfedges[tw].next_idx;
				if (he == INVALID_IDX) break;
				if (++cguard > 1024) break;
			} while (he != start_he);
		}

		if (hn_twin != INVALID_IDX) halfedges[hn_twin].twin_idx = hp_twin;
		if (hp_twin != INVALID_IDX) halfedges[hp_twin].twin_idx = hn_twin;
		if (hn_twin != INVALID_IDX) halfedges[hn_twin].edge_idx = ehp;
		if (hn_twin != INVALID_IDX)      edges[ehp].halfedge_idx = hn_twin;
		else if (hp_twin != INVALID_IDX) edges[ehp].halfedge_idx = hp_twin;
		else                              edges[ehp].halfedge_idx = INVALID_IDX;
		edges[ehn].halfedge_idx = INVALID_IDX;

		if (tn_twin != INVALID_IDX) halfedges[tn_twin].twin_idx = tp_twin;
		if (tp_twin != INVALID_IDX) halfedges[tp_twin].twin_idx = tn_twin;
		if (tp_twin != INVALID_IDX) halfedges[tp_twin].edge_idx = etn;
		if (tn_twin != INVALID_IDX)      edges[etn].halfedge_idx = tn_twin;
		else if (tp_twin != INVALID_IDX) edges[etn].halfedge_idx = tp_twin;
		else                              edges[etn].halfedge_idx = INVALID_IDX;
		edges[etp].halfedge_idx = INVALID_IDX;

		edges[idx].halfedge_idx = INVALID_IDX;
		vertices[vC].halfedge_idx = INVALID_IDX;

		halfedges[h_idx].vertex_idx  = INVALID_IDX;
		halfedges[t_idx].vertex_idx  = INVALID_IDX;
		halfedges[hn_idx].vertex_idx = INVALID_IDX;
		halfedges[hp_idx].vertex_idx = INVALID_IDX;
		halfedges[tn_idx].vertex_idx = INVALID_IDX;
		halfedges[tp_idx].vertex_idx = INVALID_IDX;

		faces[f0].halfedge_idx = INVALID_IDX;
		faces[f1].halfedge_idx = INVALID_IDX;

		vertices[vA].halfedge_idx = (hn_twin != INVALID_IDX) ? hn_twin : hp_twin;
		vertices[vD].halfedge_idx = (tn_twin != INVALID_IDX) ? tn_twin : tp_twin;
		vertices[vB].halfedge_idx = (hp_twin != INVALID_IDX) ? hp_twin : tp_twin;
	} while(0); }
}

__global__ void kernel_get_split_edges(
	Mesh::Edge* edges, Mesh::Halfedge* halfedges, Mesh::Face* faces,
	float* lengths, uint32_t num_edges, float avg_len, float split_factor, int* op_mask)
{
	for (int index = blockDim.x * blockIdx.x + threadIdx.x;
	     (uint32_t)index < num_edges;
	     index += blockDim.x * gridDim.x)
	{ do {
		op_mask[index] = 0;

		uint32_t h_idx = edges[index].halfedge_idx;
		if (h_idx == INVALID_IDX) break;
		if (halfedges[h_idx].vertex_idx == INVALID_IDX) break;
		uint32_t t_idx = halfedges[h_idx].twin_idx;
		if (t_idx == INVALID_IDX) break;
		if (halfedges[t_idx].vertex_idx == INVALID_IDX) break;
		if (faces[halfedges[h_idx].face_idx].boundary) break;
		if (faces[halfedges[t_idx].face_idx].boundary) break;

		DPRINTF("edge %u: length = %f, cmp = %f\n", index, lengths[index], avg_len * split_factor);
		op_mask[index] = lengths[index] > avg_len * split_factor;
		if (op_mask[index]) DPRINTF("edge %u should be split\n", index);
	} while(0); }
}

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
	for (int idx = blockDim.x * blockIdx.x + threadIdx.x;
	     (uint32_t)idx < num_edges;
	     idx += blockDim.x * gridDim.x)
	{ do {
		if (edge_color_mask[idx] != color) break;
		if (op_mask[idx] != 1) break;

		uint32_t h_idx = edges[idx].halfedge_idx;
		if (h_idx == INVALID_IDX) break;
		if (halfedges[h_idx].vertex_idx == INVALID_IDX) break;
		uint32_t t_idx = halfedges[h_idx].twin_idx;
		if (t_idx == INVALID_IDX) break;
		if (halfedges[t_idx].vertex_idx == INVALID_IDX) break;

		uint32_t hn_idx = halfedges[h_idx].next_idx;
		uint32_t hp_idx = halfedges[hn_idx].next_idx;
		uint32_t tn_idx = halfedges[t_idx].next_idx;
		uint32_t tp_idx = halfedges[tn_idx].next_idx;

		uint32_t vB = halfedges[h_idx].vertex_idx;
		uint32_t vC = halfedges[t_idx].vertex_idx;
		uint32_t vA = halfedges[hp_idx].vertex_idx;
		uint32_t vD = halfedges[tp_idx].vertex_idx;

		uint32_t f0 = halfedges[h_idx].face_idx;
		uint32_t f1 = halfedges[t_idx].face_idx;

		int off = split_offsets[idx];

		uint32_t vM      = base_v + off;
		uint32_t eMA_idx = base_e + off * 3;
		uint32_t eMC_idx = base_e + off * 3 + 1;
		uint32_t eMD_idx = base_e + off * 3 + 2;
		uint32_t nMA_idx = base_h + off * 6;
		uint32_t nAM_idx = base_h + off * 6 + 1;
		uint32_t nMC_idx = base_h + off * 6 + 2;
		uint32_t nCM_idx = base_h + off * 6 + 3;
		uint32_t nDM_idx = base_h + off * 6 + 4;
		uint32_t nMD_idx = base_h + off * 6 + 5;
		uint32_t f2      = base_f + off * 2;
		uint32_t f3      = base_f + off * 2 + 1;

		vertices[vM].position   = (vertices[vB].position + vertices[vC].position) * 0.5f;
		vertices[vM].halfedge_idx = t_idx;
		vertices[vM].id          = vM;

		edges[eMA_idx].halfedge_idx = nMA_idx; edges[eMA_idx].id = eMA_idx; edges[eMA_idx].sharp = false;
		edges[eMC_idx].halfedge_idx = nMC_idx; edges[eMC_idx].id = eMC_idx; edges[eMC_idx].sharp = false;
		edges[eMD_idx].halfedge_idx = nMD_idx; edges[eMD_idx].id = eMD_idx; edges[eMD_idx].sharp = false;

		faces[f2].halfedge_idx = nAM_idx; faces[f2].id = f2; faces[f2].boundary = false;
		faces[f3].halfedge_idx = nMD_idx; faces[f3].id = f3; faces[f3].boundary = false;

		halfedges[nMA_idx].vertex_idx = vM;  halfedges[nMA_idx].next_idx = hp_idx;  halfedges[nMA_idx].twin_idx = nAM_idx; halfedges[nMA_idx].edge_idx = eMA_idx; halfedges[nMA_idx].face_idx = f0;  halfedges[nMA_idx].id = nMA_idx;
		halfedges[nAM_idx].vertex_idx = vA;  halfedges[nAM_idx].next_idx = nMC_idx; halfedges[nAM_idx].twin_idx = nMA_idx; halfedges[nAM_idx].edge_idx = eMA_idx; halfedges[nAM_idx].face_idx = f2;  halfedges[nAM_idx].id = nAM_idx;
		halfedges[nMC_idx].vertex_idx = vM;  halfedges[nMC_idx].next_idx = hn_idx;  halfedges[nMC_idx].twin_idx = nCM_idx; halfedges[nMC_idx].edge_idx = eMC_idx; halfedges[nMC_idx].face_idx = f2;  halfedges[nMC_idx].id = nMC_idx;
		halfedges[nCM_idx].vertex_idx = vC;  halfedges[nCM_idx].next_idx = nMD_idx; halfedges[nCM_idx].twin_idx = nMC_idx; halfedges[nCM_idx].edge_idx = eMC_idx; halfedges[nCM_idx].face_idx = f3;  halfedges[nCM_idx].id = nCM_idx;
		halfedges[nDM_idx].vertex_idx = vD;  halfedges[nDM_idx].next_idx = t_idx;   halfedges[nDM_idx].twin_idx = nMD_idx; halfedges[nDM_idx].edge_idx = eMD_idx; halfedges[nDM_idx].face_idx = f1;  halfedges[nDM_idx].id = nDM_idx;
		halfedges[nMD_idx].vertex_idx = vM;  halfedges[nMD_idx].next_idx = tp_idx;  halfedges[nMD_idx].twin_idx = nDM_idx; halfedges[nMD_idx].edge_idx = eMD_idx; halfedges[nMD_idx].face_idx = f3;  halfedges[nMD_idx].id = nMD_idx;

		halfedges[h_idx].next_idx  = nMA_idx;
		halfedges[t_idx].vertex_idx = vM;
		halfedges[t_idx].next_idx  = tn_idx;
		halfedges[hn_idx].next_idx = nAM_idx;
		halfedges[hn_idx].face_idx = f2;
		halfedges[tn_idx].next_idx = nDM_idx;
		halfedges[tp_idx].next_idx = nCM_idx;
		halfedges[tp_idx].face_idx = f3;

		faces[f0].halfedge_idx = h_idx;
		faces[f1].halfedge_idx = t_idx;
		vertices[vC].halfedge_idx = nCM_idx;
	} while(0); }
}

void CudaRemesher::isotropic_remesh(Isotropic_Remesh_Params const &params) {
	dim3 blockDim;
	dim3 gridDim;

	uint32_t nb = params.num_blocks; // 0 = auto (full parallelism)
	// Helper lambdas: compute gridDim for edge/vertex kernels respecting nb
	auto edge_grid = [&]() -> uint32_t {
		return nb ? nb : (numEdges + blockDim.x - 1) / blockDim.x;
	};
	auto vert_grid = [&]() -> uint32_t {
		return nb ? nb : (numVertices + blockDim.x - 1) / blockDim.x;
	};

	using clk = std::chrono::steady_clock;
	auto t_total_begin = clk::now();
	double ms_color = 0, ms_flip = 0, ms_split = 0, ms_collapse = 0, ms_smooth = 0;

	for (int t = 0; t < params.num_iters; t++) {
		std::printf("iteration %d of remeshing\n", t);
		blockDim = dim3(params.block_size);
		gridDim = dim3(edge_grid());

		cudaMemset(edge_color_mask, -1, sizeof(int) * numEdges);

		bool h_done = false;
		cudaDeviceSynchronize(); auto _ts = clk::now();
		while (!h_done) {
			h_done = true;
			cudaMemcpy(d_coloring_done, &h_done, sizeof(bool), cudaMemcpyHostToDevice);
			kernel_color_edges<<<gridDim, blockDim>>>(cudaDeviceEdges, cudaDeviceHalfedges, cudaDeviceVertices, numHalfedges, numEdges, edge_color_mask, edge_priorities, d_coloring_done);
			CUDA_CHECK("color_edges_top", verbose);
			cudaMemcpy(&h_done, d_coloring_done, sizeof(bool), cudaMemcpyDeviceToHost);
		}
		ms_color += std::chrono::duration<double, std::milli>(clk::now() - _ts).count();

		int* cuda_max_color = thrust::max_element(thrust::device, edge_color_mask, edge_color_mask + numEdges);
		int max_color;
		cudaMemcpy(&max_color, cuda_max_color, sizeof(int), cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize(); _ts = clk::now();
		kernel_get_flip_edges<<<gridDim, blockDim>>>(cudaDeviceEdges, cudaDeviceHalfedges, cudaDeviceVertices, cudaDeviceFaces, numEdges, edge_op_mask);
		CUDA_CHECK("get_flip_edges", verbose);
		for (int c = 0; c <= max_color; c++) {
			VPRINTF("Flipping edges of color %d\n", c);
			kernel_flip_edge<<<gridDim, blockDim>>>(cudaDeviceEdges, cudaDeviceHalfedges, cudaDeviceVertices, cudaDeviceFaces, numEdges, edge_color_mask, edge_op_mask, c);
			CUDA_CHECK("flip_edge", verbose);
		}
		ms_flip += std::chrono::duration<double, std::milli>(clk::now() - _ts).count();

		kernel_get_edge_lengths<<<gridDim, blockDim>>>(cudaDeviceEdges, cudaDeviceHalfedges, cudaDeviceVertices, edge_lengths, numEdges);
		CUDA_CHECK("get_edge_lengths_1", verbose);

		float avg_len = thrust::reduce(thrust::device, edge_lengths, edge_lengths + numEdges, 0.0f, thrust::plus<float>()) / std::max(1U, numEdges);
		if (verbose) std::printf("average length is %f\n", avg_len);

		cudaMemset(edge_color_mask, -1, sizeof(int) * numEdges);
		h_done = false;
		cudaDeviceSynchronize(); _ts = clk::now();
		while (!h_done) {
			h_done = true;
			cudaMemcpy(d_coloring_done, &h_done, sizeof(bool), cudaMemcpyHostToDevice);
			kernel_color_edges<<<gridDim, blockDim>>>(cudaDeviceEdges, cudaDeviceHalfedges, cudaDeviceVertices, numHalfedges, numEdges, edge_color_mask, edge_priorities, d_coloring_done);
			CUDA_CHECK("color_edges_pre_split", verbose);
			cudaMemcpy(&h_done, d_coloring_done, sizeof(bool), cudaMemcpyDeviceToHost);
		}
		ms_color += std::chrono::duration<double, std::milli>(clk::now() - _ts).count();

		// === SPLIT ===
		cudaDeviceSynchronize(); _ts = clk::now();
		kernel_get_split_edges<<<gridDim, blockDim>>>(cudaDeviceEdges, cudaDeviceHalfedges, cudaDeviceFaces, edge_lengths, numEdges, avg_len, params.split_factor, edge_op_mask);
		CUDA_CHECK("get_split_edges", verbose);

		cudaMalloc(&split_offsets, sizeof(int) * numEdges);
		thrust::exclusive_scan(thrust::device, edge_op_mask, edge_op_mask + numEdges, split_offsets);
		CUDA_CHECK("exclusive_scan_split", verbose);

		int last_offset, last_mask;
		cudaMemcpy(&last_offset, split_offsets + numEdges - 1, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&last_mask, edge_op_mask + numEdges - 1, sizeof(int), cudaMemcpyDeviceToHost);
		int total_splits = last_offset + last_mask;
		if (verbose) std::printf("total splits = %d\n", total_splits);

		if (total_splits > 0) {
			uint32_t newV = numVertices + total_splits;
			uint32_t newE = numEdges + total_splits * 3;
			uint32_t newH = numHalfedges + total_splits * 6;
			uint32_t newF = numFaces + total_splits * 2;

			Mesh::Vertex*   newVertices;
			Mesh::Edge*     newEdges;
			Mesh::Halfedge* newHalfedges;
			Mesh::Face*     newFaces;
			cudaMalloc(&newVertices,  sizeof(Mesh::Vertex)   * newV);
			cudaMalloc(&newEdges,     sizeof(Mesh::Edge)     * newE);
			cudaMalloc(&newHalfedges, sizeof(Mesh::Halfedge) * newH);
			cudaMalloc(&newFaces,     sizeof(Mesh::Face)     * newF);
			cudaMemcpy(newVertices,  cudaDeviceVertices,  sizeof(Mesh::Vertex)   * numVertices,  cudaMemcpyDeviceToDevice);
			cudaMemcpy(newEdges,     cudaDeviceEdges,     sizeof(Mesh::Edge)     * numEdges,     cudaMemcpyDeviceToDevice);
			cudaMemcpy(newHalfedges, cudaDeviceHalfedges, sizeof(Mesh::Halfedge) * numHalfedges, cudaMemcpyDeviceToDevice);
			cudaMemcpy(newFaces,     cudaDeviceFaces,     sizeof(Mesh::Face)     * numFaces,     cudaMemcpyDeviceToDevice);
			cudaFree(cudaDeviceVertices);  cudaDeviceVertices  = newVertices;
			cudaFree(cudaDeviceEdges);     cudaDeviceEdges     = newEdges;
			cudaFree(cudaDeviceHalfedges); cudaDeviceHalfedges = newHalfedges;
			cudaFree(cudaDeviceFaces);     cudaDeviceFaces     = newFaces;

			cuda_max_color = thrust::max_element(thrust::device, edge_color_mask, edge_color_mask + numEdges);
			cudaMemcpy(&max_color, cuda_max_color, sizeof(int), cudaMemcpyDeviceToHost);

			for (int c = 0; c <= max_color; c++) {
				VPRINTF("Splitting edges of color %d\n", c);
				kernel_split_edge<<<gridDim, blockDim>>>(
					cudaDeviceVertices, cudaDeviceEdges, cudaDeviceHalfedges, cudaDeviceFaces,
					edge_color_mask, edge_op_mask, split_offsets,
					numEdges, numVertices, numEdges, numHalfedges, numFaces, c);
				cudaDeviceSynchronize();
			}

			numVertices  = newV;
			numEdges     = newE;
			numHalfedges = newH;
			numFaces     = newF;

			cudaFree(edge_lengths);    cudaFree(edge_color_mask); cudaFree(edge_op_mask); cudaFree(edge_priorities);
			cudaMalloc(&edge_lengths,    sizeof(float) * numEdges);
			cudaMalloc(&edge_color_mask, sizeof(int)   * numEdges);
			cudaMalloc(&edge_op_mask,    sizeof(int)   * numEdges);
			std::vector<int> h_ep(numEdges);
			for (uint32_t i = 0; i < numEdges; i++) h_ep[i] = rand();
			cudaMalloc(&edge_priorities, sizeof(int) * numEdges);
			cudaMemcpy(edge_priorities, h_ep.data(), sizeof(int) * numEdges, cudaMemcpyHostToDevice);

			cudaFree(vertex_color_mask); cudaFree(vertex_pos); cudaFree(vertex_normals); cudaFree(vertex_priorities);
			cudaMalloc(&vertex_color_mask, sizeof(int)  * numVertices);
			cudaMalloc(&vertex_pos,        sizeof(Vec3) * numVertices);
			cudaMalloc(&vertex_normals,    sizeof(Vec3) * numVertices);
			std::vector<int> h_vp(numVertices);
			for (uint32_t i = 0; i < numVertices; i++) h_vp[i] = rand();
			cudaMalloc(&vertex_priorities, sizeof(int) * numVertices);
			cudaMemcpy(vertex_priorities, h_vp.data(), sizeof(int) * numVertices, cudaMemcpyHostToDevice);

			gridDim = dim3(edge_grid());
		}
		cudaFree(split_offsets);
		split_offsets = NULL;
		ms_split += std::chrono::duration<double, std::milli>(clk::now() - _ts).count();

		// === COLLAPSE ===
		kernel_get_edge_lengths<<<gridDim, blockDim>>>(cudaDeviceEdges, cudaDeviceHalfedges, cudaDeviceVertices, edge_lengths, numEdges);
		CUDA_CHECK("get_edge_lengths_2", verbose);

		avg_len = thrust::reduce(thrust::device, edge_lengths, edge_lengths + numEdges, 0.0f, thrust::plus<float>()) / std::max(1U, numEdges);
		CUDA_CHECK("reduce_avg_len_2", verbose);
		if (verbose) std::printf("average length after split is %f\n", avg_len);

		cudaMemset(edge_color_mask, -1, sizeof(int) * numEdges);
		h_done = false;
		cudaDeviceSynchronize(); _ts = clk::now();
		while (!h_done) {
			h_done = true;
			cudaMemcpy(d_coloring_done, &h_done, sizeof(bool), cudaMemcpyHostToDevice);
			kernel_color_edges<<<gridDim, blockDim>>>(cudaDeviceEdges, cudaDeviceHalfedges, cudaDeviceVertices, numHalfedges, numEdges, edge_color_mask, edge_priorities, d_coloring_done);
			cudaMemcpy(&h_done, d_coloring_done, sizeof(bool), cudaMemcpyDeviceToHost);
		}
		ms_color += std::chrono::duration<double, std::milli>(clk::now() - _ts).count();

		cudaDeviceSynchronize(); _ts = clk::now();
		kernel_get_collapse_edges<<<gridDim, blockDim>>>(cudaDeviceEdges, cudaDeviceHalfedges, cudaDeviceFaces, edge_lengths, numEdges, avg_len, params.collapse_factor, edge_op_mask);
		CUDA_CHECK("get_collapse_edges", verbose);

		cuda_max_color = thrust::max_element(thrust::device, edge_color_mask, edge_color_mask + numEdges);
		cudaMemcpy(&max_color, cuda_max_color, sizeof(int), cudaMemcpyDeviceToHost);

		for (int c = 0; c <= max_color; c++) {
			VPRINTF("Collapsing edges of color %d\n", c);
			kernel_collapse_edge<<<gridDim, blockDim>>>(
				cudaDeviceVertices, cudaDeviceEdges, cudaDeviceHalfedges, cudaDeviceFaces,
				edge_color_mask, edge_op_mask, numEdges, c);
			CUDA_CHECK("collapse_edge", verbose);
		}
		ms_collapse += std::chrono::duration<double, std::milli>(clk::now() - _ts).count();

		// === SMOOTH ===
		gridDim = dim3(vert_grid());
		cudaMemset(vertex_color_mask, -1, sizeof(int) * numVertices);
		h_done = false;
		cudaDeviceSynchronize(); _ts = clk::now();
		while (!h_done) {
			h_done = true;
			cudaMemcpy(d_coloring_done, &h_done, sizeof(bool), cudaMemcpyHostToDevice);
			kernel_color_vertices<<<gridDim, blockDim>>>(cudaDeviceVertices, cudaDeviceHalfedges, numVertices, vertex_color_mask, vertex_priorities, d_coloring_done);
			CUDA_CHECK("color_vertices", verbose);
			cudaMemcpy(&h_done, d_coloring_done, sizeof(bool), cudaMemcpyDeviceToHost);
		}

		cuda_max_color = thrust::max_element(thrust::device, vertex_color_mask, vertex_color_mask + numVertices);
		CUDA_CHECK("max_element_v", verbose);
		cudaMemcpy(&max_color, cuda_max_color, sizeof(int), cudaMemcpyDeviceToHost);
		ms_color += std::chrono::duration<double, std::milli>(clk::now() - _ts).count();

		cudaDeviceSynchronize(); _ts = clk::now();
		for (int i = 0; i < params.smoothing_iters; i++) {
			VPRINTF("iteration %d of vertex smoothing\n", i);
			kernel_init_vertex_pos<<<gridDim, blockDim>>>(cudaDeviceVertices, vertex_pos, numVertices);
			CUDA_CHECK("init_vertex_pos");
			kernel_get_vertex_normals<<<gridDim, blockDim>>>(cudaDeviceVertices, cudaDeviceHalfedges,
				cudaDeviceFaces, vertex_normals, numVertices, numHalfedges);
			CUDA_CHECK("get_vertex_normals", verbose);
			for (int c = 0; c <= max_color; c++) {
				VPRINTF("Smoothing vertices of color %d\n", c);
				kernel_smooth_vertex<<<gridDim, blockDim>>>(cudaDeviceVertices, cudaDeviceEdges,
					cudaDeviceHalfedges, cudaDeviceFaces, vertex_color_mask, vertex_normals, vertex_pos,
					numVertices, numEdges, numHalfedges, numFaces, params.smoothing_step, c);
				CUDA_CHECK("smooth_vertex", verbose);
			}
			kernel_update_vertex_pos<<<gridDim, blockDim>>>(cudaDeviceVertices, vertex_pos, numVertices);
			CUDA_CHECK("update_vertex_pos", verbose);
		}
		ms_smooth += std::chrono::duration<double, std::milli>(clk::now() - _ts).count();
	}

	cudaDeviceSynchronize();
	double ms_total = std::chrono::duration<double, std::milli>(clk::now() - t_total_begin).count();
	double ms_phases = ms_color + ms_flip + ms_split + ms_collapse + ms_smooth;
	std::printf("\n=== Timing (ms) over %u outer iter(s) ===\n", params.num_iters);
	std::printf("  color    %10.2f  (%5.1f%%)\n", ms_color,    100.0 * ms_color    / ms_total);
	std::printf("  flip     %10.2f  (%5.1f%%)\n", ms_flip,     100.0 * ms_flip     / ms_total);
	std::printf("  split    %10.2f  (%5.1f%%)\n", ms_split,    100.0 * ms_split    / ms_total);
	std::printf("  collapse %10.2f  (%5.1f%%)\n", ms_collapse, 100.0 * ms_collapse / ms_total);
	std::printf("  smooth   %10.2f  (%5.1f%%)\n", ms_smooth,   100.0 * ms_smooth   / ms_total);
	std::printf("  -------- ----------\n");
	std::printf("  measured %10.2f  (%5.1f%%)\n", ms_phases,   100.0 * ms_phases   / ms_total);
	std::printf("  total    %10.2f  (100.0%%)\n", ms_total);
}
