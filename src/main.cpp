#include "mesh.h"
#include "cudaRemesh.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include "test.h"
#include "vec3.h"

// Note that element ids may not necessarily map to vector index after a remesh
Mesh* mesh_from_file(std::string filename) {
	std::ifstream file(filename);
	
	if (!file.is_open()) {
		return nullptr;
	}
	// per-type maps from raw id -> index in respective element array
	// (raw ids overlap across element types so a single global map is wrong)
	std::unordered_map<uint32_t, uint32_t> h_id_to_idx;
	std::unordered_map<uint32_t, uint32_t> v_id_to_idx;
	std::unordered_map<uint32_t, uint32_t> e_id_to_idx;
	std::unordered_map<uint32_t, uint32_t> f_id_to_idx;
	uint32_t h_idx = 0;
	uint32_t v_idx = 0;
	uint32_t e_idx = 0;
	uint32_t f_idx = 0;
	Mesh* mesh = new Mesh();
	
	std::string line;
	// loop through all objects and store the raw ids of the elements they refer to
	while (getline(file, line)) {
		// skip blank / too-short lines (need at least "[x0]")
		if (line.size() < 4 || line[0] != '[') continue;
		size_t rbr = line.find(']');
		if (rbr == std::string::npos || rbr <= 2) continue;
		uint32_t id = std::stoul(line.substr(2, rbr - 2));
		switch(line[1]) {
			case 'h':
			{
				// of form hA tB nC vD eG fH where A, B, C, D, G, H are ids of corresponding elements
				size_t t_pos = line.find('t');
				size_t n_pos = line.find('n');
				size_t v_pos = line.find('v');
				size_t e_pos = line.find('e');
				size_t f_pos = line.find('f');
				uint32_t t_id = std::stoul(line.substr(t_pos+1,n_pos-t_pos-1));
				uint32_t n_id = std::stoul(line.substr(n_pos+1,v_pos-n_pos-1));
				uint32_t v_id = std::stoul(line.substr(v_pos+1,e_pos-e_pos-1));
				uint32_t e_id = std::stoul(line.substr(e_pos+1,f_pos-v_pos-1));
				uint32_t f_id = std::stoul(line.substr(f_pos+1,line.length()-f_pos-1));

				Mesh::Halfedge h;
				h.twin_idx = t_id;
				h.next_idx = n_id;
				h.vertex_idx = v_id;
				h.edge_idx = e_id;
				h.face_idx = f_id;
				h.id = h_idx++;
				mesh->halfedges.emplace_back(h);
				h_id_to_idx.insert({id, h.id});
				break;
			}
			case 'v':
			{
				// of form vA hB @ {x, y, z} where A is vertex id, B is halfedge id
				size_t h_pos = line.find('h');
				uint32_t h_id = std::stoul(line.substr(h_pos+1, line.find('@')-h_pos-1));
				size_t lbr_pos = line.find('{');
				size_t c1_pos = line.find(',');
				float x = std::stof(line.substr(lbr_pos+1,c1_pos-lbr_pos-1));
				line = line.substr(c1_pos+1);
				size_t c2_pos = line.find(',');
				size_t rbr_pos = line.find('}');
				float y = std::stof(line.substr(0, c2_pos));
				float z = std::stof(line.substr(c2_pos+1,rbr_pos-c2_pos-1));
				Mesh::Vertex v;
				v.position = Vec3{x, y, z};
				v.halfedge_idx = h_id;
				v.id = v_idx++;
				mesh->vertices.emplace_back(v);
				v_id_to_idx.insert({id, v.id});
				break;
			}
			case 'e':
			{
				// of form eA hB where A, B are corresponding ids
				size_t h_pos = line.find('h');
				uint32_t h_id = std::stoul(line.substr(h_pos+1, line.length()-h_pos-1));
				
				Mesh::Edge e;
				e.halfedge_idx = h_id;
				e.id = e_idx++;
				mesh->edges.emplace_back(e);
				e_id_to_idx.insert({id, e.id});
				break;
			}
			case 'f':
			{
				// of form fA hB bX where A, B are corresponding ids, X in {0, 1} is if face is boundary 
				size_t h_pos = line.find('h');
				size_t b_pos = line.find('b');
				uint32_t h_id = std::stoul(line.substr(h_pos+1, b_pos-h_pos-1));
				int boundary = std::stoi(line.substr(b_pos+1));

				Mesh::Face f;
				f.boundary = boundary;
				f.halfedge_idx = h_id;
				f.id = f_idx++;
				mesh->faces.emplace_back(f);
				f_id_to_idx.insert({id, f.id});
				break;
			}
		}
	}
	file.close();
	
	// loop through the elements again, using the per-type maps to change raw IDs to vector indices.
	// Use references so the writes actually persist into the mesh.
	for (size_t i = 0; i < mesh->halfedges.size(); i++) {
		Mesh::Halfedge& h = mesh->halfedges[i];
		h.twin_idx = h_id_to_idx[h.twin_idx];
		h.next_idx = h_id_to_idx[h.next_idx];
		h.vertex_idx = v_id_to_idx[h.vertex_idx];
		h.edge_idx = e_id_to_idx[h.edge_idx];
		h.face_idx = f_id_to_idx[h.face_idx];
	}
	for (size_t i = 0; i < mesh->vertices.size(); i++) {
		Mesh::Vertex& v = mesh->vertices[i];
		v.halfedge_idx = h_id_to_idx[v.halfedge_idx];
	}
	for (size_t i = 0; i < mesh->edges.size(); i++) {
		Mesh::Edge& e = mesh->edges[i];
		e.halfedge_idx = h_id_to_idx[e.halfedge_idx];
	}
	for (size_t i = 0; i < mesh->faces.size(); i++) {
		Mesh::Face& f = mesh->faces[i];
		f.halfedge_idx = h_id_to_idx[f.halfedge_idx];
	}

	return mesh;
}

int main() {
	// Force line-buffered stdout so progress is visible if a hang occurs.
	setvbuf(stdout, NULL, _IOLBF, 0);
    // CudaRemesher* remesher = new CudaRemesher();
    // Mesh mesh = Mesh::from_indexed_faces({	
	// 	Vec3{-1.0f, 1.0f, 1.0f}, 	Vec3{-1.0f, 1.0f, -1.0f},
	// 	Vec3{-1.0f, -1.0f, -1.0f}, 	Vec3{-1.0f, -1.0f, 1.0f},
	// 	Vec3{1.0f, -1.0f, -1.0f}, 	Vec3{1.0f, -1.0f, 1.0f},
	// 	Vec3{1.0f, 1.0f, -1.0f}, 	Vec3{1.0f, 1.0f, 1.0f}
	// },{
	// 	{3, 0, 1, 2}, 
	// 	{5, 3, 2, 4}, 
	// 	{7, 5, 4, 6}, 
	// 	{0, 7, 6, 1}, 
	// 	{0, 3, 5, 7}, 
	// 	{6, 4, 2, 1} });
	// // std::printf("finished mesh creation\n");
    // remesher->setup(mesh);
	// Isotropic_Remesh_Params params{
	// 	1, 1.5f, 0.5f, 1, 1.0f
	// };
	// remesher->isotropic_remesh(params);
	std::printf("\n=== test_split_edge ===\n");
	test_split_edge();
	std::printf("\n=== test_collapse_edge ===\n");
	test_collapse_edge();
	std::printf("\n=== test_converge ===\n");
	test_converge();

	// Run remesh on test1 dataset (large scale) to validate at full scale
	std::printf("\n=== Loading tests/test1.txt ===\n");
	Mesh* mesh = mesh_from_file("tests/test1.txt");
	if (mesh == nullptr) {
		std::printf("failed to open tests/test1.txt\n");
		return 1;
	}
	std::printf("Loaded mesh: %zu verts, %zu edges, %zu halfedges, %zu faces\n",
		mesh->vertices.size(), mesh->edges.size(), mesh->halfedges.size(), mesh->faces.size());

	CudaRemesher* remesher = new CudaRemesher();
	remesher->setup(*mesh);
	Isotropic_Remesh_Params params{ 3, 1.5f, 0.5f, 1, 0.5f };
	remesher->isotropic_remesh(params);

	delete remesher;
	delete mesh;
	return 0;
}