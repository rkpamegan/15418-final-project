/**
 * Original code provided by 15-362 Course Staff, adapted for project use
 * TODO: normal() function for vertex, in order to compute tangential smoothing
 */
#include "mesh.h"
#include <stdint.h>

#include "vec3.h"

#include <unordered_map>
#include <set>
#include <map>

#include <iostream>

std::ostream& operator << (std::ostream& outs, const Mesh::Vertex& v);
std::ostream& operator << (std::ostream& outs, const Mesh::Edge& e);
std::ostream& operator << (std::ostream& outs, const Mesh::Halfedge& h);
std::ostream& operator << (std::ostream& outs, const Mesh::Face& f);

namespace std {
template< typename A, typename B >
struct hash< std::pair< A, B > > {
	size_t operator()(std::pair< A, B > const &key) const {
		static const std::hash< A > ha;
		static const std::hash< B > hb;
		size_t hf = ha(key.first);
		size_t hs = hb(key.second);
		//NOTE: if this were C++20 we could use std::rotr or std::rotl
		return hf ^ (hs << (sizeof(size_t)*4)) ^ (hs >> (sizeof(size_t)*4));
	}
};
}

Mesh Mesh::from_indexed_faces(std::vector< Vec3 > const &vertices_, 
		std::vector< std::vector< uint32_t > > const &faces_)
{

	Mesh mesh;

	for (auto const &v : vertices_) {
		uint32_t vi = mesh.emplace_vertex();
		mesh.vertices[vi].position = v;
	}

	std::unordered_map< std::pair< uint32_t, uint32_t >, uint32_t > halfedge_map; //for quick lookup of halfedge index by from/to vertex index

	uint32_t num_faces = static_cast<uint32_t>(faces_.size());
	//helper to add a face (and, later, boundary):
	auto add_loop = [&](std::vector< uint32_t > const &loop, bool boundary,
					    std::vector< uint32_t> const &n_loop = std::vector<uint32_t>{}, 
						std::vector< uint32_t> const &uv_loop = std::vector<uint32_t>{}) {
		assert(loop.size() >= 3);
		
		for (uint32_t j = 0; j < loop.size(); ++j) {
			//omit adding a face with vertices of the same index, otherwise crashes on cylinder/cone
			if (loop[j] == loop[(j + 1) % loop.size()]) { return; }
		}

		uint32_t face_idx = mesh.emplace_face(boundary);
		uint32_t first_he_idx = INVALID_IDX;
		uint32_t prev_he_idx = INVALID_IDX;
		for (uint32_t i = 0; i < loop.size(); ++i) {
			uint32_t a = loop[i];
			uint32_t b = loop[(i + 1) % loop.size()];
			assert(a != b);

			uint32_t he_idx = mesh.emplace_halfedge();
			if (i == 0) {
				mesh.faces[face_idx].halfedge_idx = he_idx;
				first_he_idx = he_idx;
			}
			mesh.halfedges[he_idx].vertex_idx = a;

			//if first to mention vertex, set vertex's halfedge index:
			if (mesh.vertices[a].halfedge_idx == INVALID_IDX) {
				assert(!boundary); //boundary faces should never be mentioning novel vertices, since they are created second
				mesh.vertices[a].halfedge_idx = he_idx;
			}
			mesh.halfedges[he_idx].face_idx = face_idx;

			auto inserted = halfedge_map.emplace(std::make_pair(a,b), he_idx);
			assert(inserted.second); //if edge mentioned more than once in the same direction, not an oriented, manifold mesh

			auto twin = halfedge_map.find(std::make_pair(b,a));
			if (twin == halfedge_map.end()) {
				assert(!boundary); //boundary faces exist only to complete edges so should *always* match
				//not twinned yet -- create an edge just for this halfedge:
				uint32_t edge_idx = mesh.emplace_edge(false);
				mesh.halfedges[he_idx].edge_idx = edge_idx;
				mesh.edges[edge_idx].halfedge_idx = he_idx;
			} else {
				//found a twin -- connect twin indices and reference its edge:
				uint32_t twin_he_idx = twin->second;
				assert(mesh.halfedges[twin_he_idx].twin_idx == INVALID_IDX);
				mesh.halfedges[he_idx].twin_idx = twin_he_idx;
				mesh.halfedges[he_idx].edge_idx = mesh.halfedges[twin_he_idx].edge_idx;
				mesh.halfedges[twin_he_idx].twin_idx = he_idx;
			}

			if (prev_he_idx != INVALID_IDX) mesh.halfedges[prev_he_idx].next_idx = he_idx;
			prev_he_idx = he_idx;
		}

		mesh.halfedges[prev_he_idx].next_idx = first_he_idx;
	};
	
	//add all faces:
	for (uint32_t i = 0; i < num_faces; i++) {
		add_loop(faces_[i], false);
	}
	

	// All halfedges created so far have valid next pointers, but some may be missing twins because they are at a boundary.
	
	std::map< uint32_t, uint32_t > next_on_boundary;


	//first, look for all un-twinned halfedges to figure out the shape of the boundary:
	for (auto const &[ from_to, he_idx ] : halfedge_map) {
		if (mesh.halfedges[he_idx].twin_idx == INVALID_IDX) {
			auto ret = next_on_boundary.emplace(from_to.second, from_to.first); //twin needed on the boundary
			assert(ret.second); //every boundary vertex should have a unique successor because the boundary is "half-disc-like"
		}
	}

	//now pull out boundary loops until all edges are exhausted:
	while (!next_on_boundary.empty()) {
		std::vector< uint32_t > loop;
		loop.emplace_back(next_on_boundary.begin()->first);

		do {
			//look up next hop on the boundary:
			auto next = next_on_boundary.find(loop.back());
			//should never be dead ends on boundary:
			assert(next != next_on_boundary.end());

			//add next hop to loop:
			loop.emplace_back(next->second);
			//...and remove from nexts structure:
			next_on_boundary.erase(next);
		} while (loop[0] != loop.back());

		loop.pop_back(); //remove duplicated first/last element

		assert(loop.size() >= 3); //all faces must be non-degenerate

		//add boundary loop:
		add_loop(loop, true);
	}

	/*
	//with boundary faces created, mesh should be ready to go with all edges nicely twinned.

	//PARANOIA: this should never happen:
	auto error = mesh.validate();
	if (error) {
		std::cerr << "Mesh from_indexed_faces failed validation: " << error->second << std::endl;
		assert(0);
	}
	*/

	return mesh;
}

uint32_t Mesh::emplace_vertex() {
	uint32_t idx = vertices.size();
	vertices.emplace_back(Vertex(next_v_id++));
	return idx;
}

uint32_t Mesh::emplace_edge(bool sharp) {
	uint32_t idx = edges.size();
	edges.emplace_back(Edge(next_e_id++, sharp));
	return idx;
}

uint32_t Mesh::emplace_face(bool boundary) {
	uint32_t idx = faces.size();
	faces.emplace_back(Face(next_f_id++, boundary));
	return idx;
}

uint32_t Mesh::emplace_halfedge() {
	uint32_t idx = halfedges.size();
	halfedges.emplace_back(Halfedge(next_h_id++));
	return idx;
}

std::string Mesh::Vertex::to_string() const {
	std::string s = "v" + std::to_string(id);
	if (halfedge_idx != INVALID_IDX) {
		s += " h" + std::to_string(halfedge_idx);
	}
	s += " @ " + ::to_string(position);
	return s;
}

std::ostream& operator << (std::ostream& outs, const Mesh::Vertex& v) {
	return outs << v.to_string();
}


std::string Mesh::Edge::to_string() const {
	std::string s = "e" + std::to_string(id);
	if (halfedge_idx != INVALID_IDX) s += " h" + std::to_string(halfedge_idx); 
	return s;
}
std::ostream& operator << (std::ostream& outs, const Mesh::Edge& e) {
	return outs << e.to_string();
}
std::string Mesh::Halfedge::to_string() const {
	std::string s = "h" + std::to_string(id);
	if (twin_idx != INVALID_IDX) s += " t" + std::to_string(twin_idx);
	if (next_idx != INVALID_IDX) s += " n" + std::to_string(next_idx); 
	if (vertex_idx != INVALID_IDX) s += " v" + std::to_string(vertex_idx); 
	if (edge_idx != INVALID_IDX) s += " e" + std::to_string(edge_idx);
	if (face_idx != INVALID_IDX) s += " f" + std::to_string(face_idx);
	return s;
}
std::ostream& operator << (std::ostream& outs, const Mesh::Halfedge& h) {
	return outs << h.to_string();
}
std::string Mesh::Face::to_string() const {
	std::string s = "f" + std::to_string(id);
	if (halfedge_idx != INVALID_IDX) s += " h" + std::to_string(halfedge_idx);
	s += " b" + std::to_string(boundary); 
	return s;
}
std::ostream& operator << (std::ostream& outs, const Mesh::Face& f) {
	return outs << f.to_string();
}

void Mesh::describe() const {
	for (size_t i = 0; i < vertices.size(); i++)
	{
		std::cout << vertices[i] << std::endl;
	}

	for (size_t i = 0; i < edges.size(); i++)
	{
		std::cout << edges[i] << std::endl;
	}

	for (size_t i = 0; i < halfedges.size(); i++)
	{
		std::cout << halfedges[i] << std::endl;
	}

	for (size_t i = 0; i < faces.size(); i++)
	{
		std::cout << faces[i] << std::endl;
	}
}