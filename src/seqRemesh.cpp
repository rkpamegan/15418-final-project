#include "mesh.h"
#include "vec3.h"
#include <iostream>
// void Remesher::setup(Mesh &_mesh) {
// 	mesh = &_mesh;

// 	Vec3* vertex_pos = (Vec3*) malloc(sizeof(Vec3) * vertices.size());
// 	Vec3* vertex_normals = (Vec3*) malloc(sizeof(Vec3) * vertices.size());

// 	// // Generate random priorities for graph coloring
// 	// std::vector<int> h_priorities(numVertices);
// 	// for (uint32_t i = 0; i < numVertices; i++) {
// 	// 	h_priorities[i] = rand();
// 	// }
// 	// cudaMalloc(&vertex_priorities, sizeof(int) * numVertices);
// 	// cudaMemcpy(vertex_priorities, h_priorities.data(), sizeof(int) * numVertices, cudaMemcpyHostToDevice);

// 	// // Generate random priorities for edge coloring
// 	// std::vector<int> h_edge_priorities(numEdges);
// 	// for (uint32_t i = 0; i < numEdges; i++) {
// 	// 	h_edge_priorities[i] = rand();
// 	// }
// 	// cudaMalloc(&edge_priorities, sizeof(int) * numEdges);
// 	// cudaMemcpy(edge_priorities, h_edge_priorities.data(), sizeof(int) * numEdges, cudaMemcpyHostToDevice);

// 	// cudaMalloc(&d_coloring_done, sizeof(bool));

// 	// cudaError_t err = cudaGetLastError();
// 	// if (err != cudaSuccess) printf("error copying data: %s\n", cudaGetErrorString(err));

// 	std::printf("setup: numVertices=%lu, numEdges=%lu, numHalfedges=%lu, numFaces=%lu\n",
// 		vertices.size(), edges.size(), halfedges.size(), faces.size());
// }


uint32_t Mesh::vertex_degree(uint32_t v) {
	uint32_t start_he = vertices[v].halfedge_idx;
	if (start_he == INVALID_IDX) return 0;
	if (halfedges[start_he].vertex_idx == INVALID_IDX) return 0;
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

float Mesh::edge_length(uint32_t e) {
	uint32_t v1 = halfedges[edges[e].halfedge_idx].vertex_idx;
	uint32_t v2 = halfedges[halfedges[edges[e].halfedge_idx].twin_idx].vertex_idx;
	return (vertices[v1].position - vertices[v2].position).norm();
}

void Mesh::flip_edge(uint32_t e) {
	uint32_t h_idx = edges[e].halfedge_idx;
	if (h_idx == INVALID_IDX) return;
	if (halfedges[h_idx].vertex_idx == INVALID_IDX) return;
	uint32_t t_idx = halfedges[h_idx].twin_idx;
	if (t_idx == INVALID_IDX) return; // boundary edge
	if (halfedges[t_idx].vertex_idx == INVALID_IDX) return;

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

void Mesh::flip_edges() {
	for (uint32_t e = 0; e < edges.size(); e++) {
		uint32_t v1 = halfedges[edges[e].halfedge_idx].vertex_idx;
		uint32_t v2 = halfedges[halfedges[edges[e].halfedge_idx].twin_idx].vertex_idx;

		uint32_t v3 = halfedges[halfedges[edges[e].halfedge_idx].next_idx].vertex_idx;
		uint32_t v4 = halfedges[halfedges[halfedges[edges[e].halfedge_idx].twin_idx].next_idx].vertex_idx;

		int deg1 = vertex_degree(v1);
		int deg2 = vertex_degree(v2);
		int deg3 = vertex_degree(v3);
		int deg4 = vertex_degree(v4);
	
		int dev1 = abs(deg1 - 6) + abs(deg2 - 6) + abs(deg3 - 6) + abs(deg4 - 6);
		int dev2 = abs(deg1 - 1 - 6) + abs(deg2 - 1 - 6) + abs(deg3 + 1 - 6) + abs(deg4 + 1 - 6);

		if (dev2 < dev1) {
			flip_edge(e);
		}
	}
}

void Mesh::split_edge(uint32_t e) {
	uint32_t h_idx = edges[e].halfedge_idx;
	if (h_idx == INVALID_IDX) return;
	if (halfedges[h_idx].vertex_idx == INVALID_IDX) return;
	uint32_t t_idx = halfedges[h_idx].twin_idx;
	if (t_idx == INVALID_IDX) return; // skip boundary edges
	if (halfedges[t_idx].vertex_idx == INVALID_IDX) return;

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


	uint32_t vM       = emplace_vertex();

	uint32_t eMA_idx  = emplace_edge(false);
	uint32_t eMC_idx  = emplace_edge(false);
	uint32_t eMD_idx  = emplace_edge(false);

	uint32_t nMA_idx  = emplace_halfedge();       // M→A
	uint32_t nAM_idx  = emplace_halfedge();   // A→M
	uint32_t nMC_idx  = emplace_halfedge();   // M→C
	uint32_t nCM_idx  = emplace_halfedge();   // C→M
	uint32_t nDM_idx  = emplace_halfedge();   // D→M
	uint32_t nMD_idx  = emplace_halfedge();   // M→D

	uint32_t f2       = emplace_face(false);        // face AMC
	uint32_t f3       = emplace_face(false);    // face MDC

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

uint32_t Mesh::split_edges(float avg_len, float split_factor) {
	uint32_t num_edges = edges.size();
	uint32_t count = 0;
	for (uint32_t e = 0; e < num_edges; e++) {
		float length = edge_length(e);

		if (length > avg_len * split_factor) {
			split_edge(e);
			count++;
		}
	}
	return count;
}

void Mesh::collapse_edge(uint32_t e) {
	uint32_t h_idx = edges[e].halfedge_idx;
	if (h_idx == INVALID_IDX) return;
	if (halfedges[h_idx].vertex_idx == INVALID_IDX) return;
	uint32_t t_idx = halfedges[h_idx].twin_idx;
	if (t_idx == INVALID_IDX) return; // boundary edge
	if (halfedges[t_idx].vertex_idx == INVALID_IDX) return;

	// 6 halfedges of the two triangles
	uint32_t hn_idx = halfedges[h_idx].next_idx;   // C→A
	if (hn_idx == INVALID_IDX) return;
	uint32_t hp_idx = halfedges[hn_idx].next_idx;   // A→B
	if (hp_idx == INVALID_IDX) return;
	uint32_t tn_idx = halfedges[t_idx].next_idx;    // B→D
	if (tn_idx == INVALID_IDX) return;
	uint32_t tp_idx = halfedges[tn_idx].next_idx;   // D→C
	if (tp_idx == INVALID_IDX) return;
	// If any of these inner halfedges was invalidated by a prior collapse this kernel,
	// our pre-collapse view is stale; skip to avoid corrupting connectivity further.
	if (halfedges[hn_idx].vertex_idx == INVALID_IDX) return;
	if (halfedges[hp_idx].vertex_idx == INVALID_IDX) return;
	if (halfedges[tn_idx].vertex_idx == INVALID_IDX) return;
	if (halfedges[tp_idx].vertex_idx == INVALID_IDX) return;

	// 4 vertices
	uint32_t vB = halfedges[h_idx].vertex_idx;
	uint32_t vC = halfedges[t_idx].vertex_idx;
	uint32_t vA = halfedges[hp_idx].vertex_idx;
	uint32_t vD = halfedges[tp_idx].vertex_idx;
	if (vA == INVALID_IDX || vB == INVALID_IDX || vC == INVALID_IDX || vD == INVALID_IDX) return;

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

	// Merge twin pairs: make outer halfedges twins of each other
	// hn and hp's twins become direct twins (removing face f0)
	if (hn_twin != INVALID_IDX) halfedges[hn_twin].twin_idx = hp_twin;
	if (hp_twin != INVALID_IDX) halfedges[hp_twin].twin_idx = hn_twin;
	// Merge one edge: keep ehp, mark ehn as invalid.
	// IMPORTANT: edges[ehp].halfedge_idx may still point to hp_idx (now invalid).
	// Redirect it to a surviving halfedge of this edge.
	if (hn_twin != INVALID_IDX) halfedges[hn_twin].edge_idx = ehp;
	if (hn_twin != INVALID_IDX)      edges[ehp].halfedge_idx = hn_twin;
	else if (hp_twin != INVALID_IDX) edges[ehp].halfedge_idx = hp_twin;
	else                              edges[ehp].halfedge_idx = INVALID_IDX;
	edges[ehn].halfedge_idx = INVALID_IDX;

	// tn and tp's twins become direct twins (removing face f1)
	if (tn_twin != INVALID_IDX) halfedges[tn_twin].twin_idx = tp_twin;
	if (tp_twin != INVALID_IDX) halfedges[tp_twin].twin_idx = tn_twin;
	// Merge one edge: keep etn, mark etp as invalid.
	// edges[etn].halfedge_idx may still point to tn_idx (now invalid). Redirect.
	if (tp_twin != INVALID_IDX) halfedges[tp_twin].edge_idx = etn;
	if (tn_twin != INVALID_IDX)      edges[etn].halfedge_idx = tn_twin;
	else if (tp_twin != INVALID_IDX) edges[etn].halfedge_idx = tp_twin;
	else                              edges[etn].halfedge_idx = INVALID_IDX;
	edges[etp].halfedge_idx = INVALID_IDX;

	// Mark the collapsed edge as invalid
	edges[e].halfedge_idx = INVALID_IDX;

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
	// Convention: vertex.halfedge must be an outgoing halfedge (halfedges[h].vertex_idx == that vertex).
	// hn_twin source = A; hp_twin source = B; tn_twin source = D; tp_twin source = C (rewritten to B).
	vertices[vA].halfedge_idx = (hn_twin != INVALID_IDX) ? hn_twin : hp_twin;
	vertices[vD].halfedge_idx = (tn_twin != INVALID_IDX) ? tn_twin : tp_twin;
	vertices[vB].halfedge_idx = (hp_twin != INVALID_IDX) ? hp_twin : tp_twin;
}

void Mesh::collapse_edges(float avg_len, float collapse_factor) {
	uint32_t e = 0;
	while (e < edges.size()) {
		uint32_t h_idx = edges[e].halfedge_idx;
		if (h_idx == INVALID_IDX) { e++; continue; }
		if (halfedges[h_idx].vertex_idx == INVALID_IDX) { e++; continue; }
		uint32_t t_idx = halfedges[h_idx].twin_idx;
		if (t_idx == INVALID_IDX) { e++; continue; } // boundary edge
		if (halfedges[t_idx].vertex_idx == INVALID_IDX) { e++; continue; }
		if (faces[halfedges[h_idx].face_idx].boundary) { e++; continue; }
		if (faces[halfedges[t_idx].face_idx].boundary) { e++; continue; }

		float length = edge_length(e);
		if (length < avg_len * collapse_factor)
		{
			collapse_edge(e);
		}
		e++;
	}
}

void Mesh::smooth_vertices(std::vector<Vec3>& vertex_pos, std::vector<Vec3>& vertex_normals, float smoothing_factor) {
	for (uint32_t i = 0; i < vertices.size(); i++) {
		printf("%u\n", i);
		Mesh::Vertex v = vertices[i];

		Vec3 center;
		
		uint32_t h_idx = v.halfedge_idx;
		if (h_idx == INVALID_IDX) return; // collapsed/isolated vertex — skip
		// stale start_he: vertex points to halfedge that was invalidated by prior collapse
		if (halfedges[h_idx].vertex_idx == INVALID_IDX) return;
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
		
		if (count == 0) return;
		center /= count;
		
		center = v.position + smoothing_factor * (center - v.position);
		Vec3 normal = vertex_normals[i];
		center = center - dot(normal, center) * normal;
		vertex_pos[i] = center;
	}
}

void Mesh::get_vertex_normals(std::vector<Vec3>& vertex_normals) {
	for (uint32_t v = 0; v < vertices.size(); v++) {
		Vec3 n = Vec3(0.0f, 0.0f, 0.0f);
		Vec3 pi = vertices[v].position;
		uint32_t h_idx = vertices[v].halfedge_idx;
		if (h_idx == INVALID_IDX) { vertex_normals[v] = Vec3(0.0f, 0.0f, 0.0f); return; }
		if (halfedges[h_idx].vertex_idx == INVALID_IDX) { vertex_normals[v] = Vec3(0.0f, 0.0f, 0.0f); return; }
		
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
			// advance to next outgoing halfedge around the vertex
			uint32_t tw = h.twin_idx;
			if (tw == INVALID_IDX) break;
			curr_idx = halfedges[tw].next_idx;
			if (curr_idx == INVALID_IDX) break;
			if (++guard > 1024) break;
		} while (curr_idx != h_idx);
		printf("%u h idx was = %u\n", v, h_idx);
		
		float len = std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
		if (len > 1e-12f) n = n * (1.0f / len);
		else n = Vec3(0.0f, 0.0f, 0.0f);
		vertex_normals[v] = n;
	}
}

void Mesh::update_vertex_pos(std::vector<Vec3>& vertex_pos) {
	for (uint32_t i = 0; i < vertices.size(); i++) {
		vertices[i].position = vertex_pos[i];
	}
}

//isotropic_remesh: improves mesh quality through local operations.
void Mesh::isotropic_remesh(Isotropic_Remesh_Params const &params) {
	/**
	 * 	1.	Split edges much longer than the target length.
	 * 			("much longer" means > target length * params.split_factor)
	 * 	2.	Collapse edges much shorter than the target length.
	 *	3.	Apply some tangential smoothing to the vertex positions.
	 *		This means move every vertex in the plane of its normal,
	 *		toward the centroid of its neighbors, by params.smoothing_step of
	 *		the total distance (so, smoothing_step of 1 would move all the way,
	 *		smoothing_step of 0 would not move). 
	 *	4.	Repeat the tangential smoothing part params.smoothing_iters times.
	 *	5. Repeat steps 1-4 `num_iters` times.
	 */

	//NOTE: many of the steps in this function will be modifying the element
	//      lists they are looping over. Take care to avoid use-after-free
	//      or infinite-loop problems.
	std::vector<Vec3> vertex_normals(vertices.size());
	std::vector<Vec3> vertex_pos(vertices.size());
	for (uint32_t t = 0; t < params.num_iters; t++) {
		std::printf("iteration %d of remeshing\n", t);
		std::printf("flipping edges\n");
		flip_edges();
		
		float avg_len = 0.0f;
		for (uint32_t e = 0; e < edges.size(); e++) {
			avg_len += edge_length(e) / edges.size();
		}
		std::printf("splitting edges\n");
		uint32_t num_splits = split_edges(avg_len, params.split_factor);
		vertex_normals.resize(vertex_normals.size() + num_splits);
		vertex_pos.resize(vertex_pos.size() + num_splits);
		std::printf("collapsing edges\n");
		avg_len = 0.0f;
		for (uint32_t e = 0; e < edges.size(); e++) {
			avg_len += edge_length(e) / edges.size();
		}
		collapse_edges(avg_len, params.collapse_factor);
		std::optional<std::pair<uint32_t, std::string>> res = validate();
		std::string res_str = res == std::nullopt ? "no problems" : res.value().second;
		std::cout << res_str << std::endl;
		
		
		// if (verbose) std::printf("smoothing vertices\n");
		// for (uint32_t i = 0; i < params.smoothing_iters; i++) {
		// 	get_vertex_normals(vertex_normals);
		// 	smooth_vertices(vertex_pos, vertex_normals, params.smoothing_step);
		// 	update_vertex_pos(vertex_pos);
		// }
	}
}