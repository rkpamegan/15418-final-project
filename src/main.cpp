#include "mesh.h"
#include "cudaRemesh.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include "test.h"
#include "vec3.h"
#include <getopt.h>

// Note that element ids may not necessarily map to vector index after a remesh
Mesh* mesh_from_file(std::string filename) {
	std::ifstream file(filename);
	
	if (!file.is_open()) {
		return nullptr;
	}
	// maps id to position in respective element array
	std::unordered_map< uint32_t, uint32_t> id_to_idx;
	uint32_t h_idx = 0;
	uint32_t v_idx = 0;
	uint32_t e_idx = 0;
	uint32_t f_idx = 0;
	Mesh* mesh = new Mesh();
	
	std::string line;
	// loop through all objects and store the raw ids of the elements they refer to
	while (getline(file, line)) {
		if (line.length() < 2) continue;
		uint32_t id = std::stoul(line.substr(2, line.find("]")-1));
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
				id_to_idx.insert({id, h.id});
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
				id_to_idx.insert({id, v.id});
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
				id_to_idx.insert({id, e.id});
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
				id_to_idx.insert({id, f.id});
				break;
			}
		}
	}
	file.close();
	
	// loop through the elements again, using the map we created to change the raw IDs to vector indices within the mesh object
	for (size_t i = 0; i < mesh->halfedges.size(); i++) {
		mesh->halfedges[i].twin_idx = id_to_idx[mesh->halfedges[i].twin_idx];
		mesh->halfedges[i].next_idx = id_to_idx[mesh->halfedges[i].next_idx];
		mesh->halfedges[i].vertex_idx = id_to_idx[mesh->halfedges[i].vertex_idx];
		mesh->halfedges[i].edge_idx = id_to_idx[mesh->halfedges[i].edge_idx];
		mesh->halfedges[i].face_idx = id_to_idx[mesh->halfedges[i].face_idx];
	}
	for (size_t i = 0; i < mesh->vertices.size(); i++) {
		mesh->vertices[i].halfedge_idx = id_to_idx[mesh->vertices[i].halfedge_idx];
	}
	for (size_t i = 0; i < mesh->edges.size(); i++) {
		mesh->edges[i].halfedge_idx = id_to_idx[mesh->edges[i].halfedge_idx];
	}
	for (size_t i = 0; i < mesh->faces.size(); i++) {
		mesh->faces[i].halfedge_idx = id_to_idx[mesh->faces[i].halfedge_idx];
	}

	return mesh;
}

void help()
{
	printf("Usage: ./remesh [options] meshfile\n");
	printf("Options:\n");
	printf("\t-r seq/par	Remesher type: `seq` for sequential, `par` for parallel (CUDA)\n");
	printf("\t-b t			CUDA block size of `t`, i.e. t threads per block\n");
	printf("\t-o filename	Output mesh description to filename\n");
	printf("")
}

int main(int argc, char* argv[]) {
	int opt;
	std::string remesher_type;
	std::string input_file;
	bool verbose = false;
	while ((opt = getopt(argc, argv, ":r:b:o:vh")) != -1) {
		switch(opt) {
			case 'r':
				remesher_type = optarg;
				printf("%s\n", optarg);
				break;
			case 'h':
				help();
				exit();
				break;
			case 'b':
				break;
			case 'o':
				break;
			case 'v':
				verbose = true;
				break;
		}
	}
    CudaRemesher* remesher = new CudaRemesher();
	Mesh* mesh = mesh_from_file("tests/test2.txt");

    remesher->setup(*mesh);
	Isotropic_Remesh_Params params{
		1, 1.5f, 0.5f, 1, 1.0f
	};
	mesh->describe();
	remesher->isotropic_remesh(params);
	// test_converge();

	free(mesh);
    return 0;
}