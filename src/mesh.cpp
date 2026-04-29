/**
 * Original code provided by 15-362 Course Staff, adapted for project use
 * TODO: normal() function for vertex, in order to compute tangential smoothing
 */
#include "mesh.h"
#include <stdint.h>

#include "vec3.h"

#include <unordered_map>
#include <unordered_set>
#include <set>
#include <map>

#include <iostream>
#include <optional>

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

std::optional<std::pair<uint32_t, std::string>> Mesh::validate() const {

	//helpers for error messages:
	auto describe_vertex   = [this](uint32_t idx)   { return "Vertex with id " + std::to_string(vertices[idx].id); };
	auto describe_edge     = [this](uint32_t idx)     { return "Edge with id " + std::to_string(edges[idx].id); };
	auto describe_face     = [this](uint32_t idx)     { return "Face with id " + std::to_string(faces[idx].id); };
	auto describe_halfedge = [this](uint32_t idx) { return "Halfedge with id " + std::to_string(halfedges[idx].id); };


	//-----------------------------
	//check element data:
	for (uint32_t i = 0; i < vertices.size(); i++) {
		if (!std::isfinite(vertices[i].position.x)) return {{i, describe_vertex(i) + " has position.x set to a non-finite value: " + std::to_string(vertices[i].position.x) + "."}};
		if (!std::isfinite(vertices[i].position.y)) return {{i, describe_vertex(i) + " has position.y set to a non-finite value: " + std::to_string(vertices[i].position.y) + "."}};
		if (!std::isfinite(vertices[i].position.z)) return {{i, describe_vertex(i) + " has position.z set to a non-finite value: " + std::to_string(vertices[i].position.z) + "."}};
	}

	// NOTE: removed edge, face, halfedge data checks since we don't use that data


	//-----------------------------
	//all references held by elements are to members of the vertices, edges, faces, or halfedges lists

	// //for checking whether a reference is held in a given list:
	// std::unordered_set< Vertex const * > in_vertices = element_addresses(vertices);
	// std::unordered_set< Edge const * > in_edges = element_addresses(edges);
	// std::unordered_set< Face const * > in_faces = element_addresses(faces);
	// std::unordered_set< Halfedge const * > in_halfedges = element_addresses(halfedges);

	// //helpers for describing things that aren't in the element lists:
	// std::unordered_set< Vertex const * > in_free_vertices = element_addresses(free_vertices);
	// std::unordered_set< Edge const * > in_free_edges = element_addresses(free_edges);
	// std::unordered_set< Face const * > in_free_faces = element_addresses(free_faces);
	// std::unordered_set< Halfedge const * > in_free_halfedges = element_addresses(free_halfedges);

	// auto describe_missing_vertex = [this,&in_free_vertices](uint32_t) -> std::string {
	// 	if (v == vertices.end()) return "past-the-end vertex";
	// 	else if (in_free_vertices.count(&*v)) return "erased vertex with old id " + std::to_string(v->id & 0x7fffffffu);
	// 	else return "out-of-mesh vertex with address " + to_string(&*v);
	// };

	// auto describe_missing_edge = [this,&in_free_edges](EdgeCRef e) -> std::string {
	// 	if (e == edges.end()) return "past-the-end edge";
	// 	else if (in_free_edges.count(&*e)) return "erased edge with old id " + std::to_string(e->id & 0x7fffffffu);
	// 	else return "out-of-mesh edge with address " + to_string(&*e);
	// };

	// auto describe_missing_face = [this,&in_free_faces](FaceCRef f) -> std::string {
	// 	if (f == faces.end()) return "past-the-end face";
	// 	else if (in_free_faces.count(&*f)) return "erased face with old id " + std::to_string(f->id & 0x7fffffffu);
	// 	else return "out-of-mesh face with address " + to_string(&*f);
	// };

	// auto describe_missing_halfedge = [this,&in_free_halfedges](HalfedgeCRef h) -> std::string {
	// 	if (h == halfedges.end()) return "past-the-end halfedge";
	// 	else if (in_free_halfedges.count(&*h)) return "erased halfedge with old id " + std::to_string(h->id & 0x7fffffffu);
	// 	else return "out-of-mesh halfedge with address " + to_string(&*h);
	// };

	//check references made by vertices:
	for (uint32_t i = 0; i < vertices.size(); i++) {
		uint32_t h_idx = vertices[i].halfedge_idx;
		if (!(0 <= h_idx && h_idx < halfedges.size())) return {{i, describe_vertex(i) + " references halfedge with index " + std::to_string(h_idx) + "."}};
	}

	//check references made by edges:
	for (uint32_t i = 0; i < edges.size(); i++) {
		uint32_t h_idx = edges[i].halfedge_idx;
		if (!(0 <= h_idx && h_idx < halfedges.size())) return {{i, describe_edge(i) + " references halfedge with index " + std::to_string(h_idx) + "."}};
	}

	//check references made by faces:
	for (uint32_t i = 0; i < faces.size(); i++) {
		uint32_t h_idx = faces[i].halfedge_idx;
		if (!(0 <= h_idx && h_idx < halfedges.size())) return {{i, describe_face(i) + " references halfedge with index " + std::to_string(h_idx) + "."}};
	}

	//check references made by halfedges:
	for (uint32_t i = 0; i < halfedges.size(); i++) {
		uint32_t t_idx = halfedges[i].twin_idx;
		if (!(0 <= t_idx && t_idx < halfedges.size())) return {{i, describe_halfedge(i) + " references halfedge with index " + std::to_string(t_idx) + "."}};

		uint32_t n_idx = halfedges[i].next_idx;
		if (!(0 <= n_idx && n_idx < halfedges.size())) return {{i, describe_halfedge(i) + " references halfedge with index " + std::to_string(n_idx) + "."}};

		uint32_t v_idx = halfedges[i].vertex_idx;
		if (!(0 <= v_idx && v_idx < vertices.size())) return {{i, describe_halfedge(i) + " references vertex with index " + std::to_string(v_idx) + "."}};

		uint32_t e_idx = halfedges[i].edge_idx;
		if (!(0 <= e_idx && e_idx < edges.size())) return {{i, describe_halfedge(i) + " references edge with index " + std::to_string(e_idx) + "."}};

		uint32_t f_idx = halfedges[i].face_idx;
		if (!(0 <= f_idx && f_idx < faces.size())) return {{i, describe_halfedge(i) + " references face with index " + std::to_string(f_idx) + "."}};
	}

	//------------------------------
	// - `edge->halfedge(->twin)^n` is a cycle of two halfedges
	//   - this is also exactly the set of halfedges that reference `edge`
	// - `face->halfedge(->next)^n` is a cycle of at least three halfedges
	//   - this is also exactly the set of halfedges that reference `face`
	// - `vertex->halfedge(->twin->next)^n` is a cycle of at least two halfedges
	//   - this is also exactly the set of halfedges that reference `vertex`

	//first, build list of all halfedges that reference every other feature:
	// maps vertex/edge/face index to set of halfedge indices
	std::unordered_map< uint32_t, std::unordered_set< uint32_t > > vertex_halfedges;
	std::unordered_map< uint32_t, std::unordered_set< uint32_t > > edge_halfedges;
	std::unordered_map< uint32_t, std::unordered_set< uint32_t > > face_halfedges;
	for (uint32_t h = 0; h < halfedges.size(); h++) {
		auto vret = vertex_halfedges[halfedges[h].vertex_idx].emplace(h);
		assert(vret.second); //every halfedge is unique, so emplace can't fail
		auto eret = edge_halfedges[halfedges[h].edge_idx].emplace(h);
		assert(eret.second); //every halfedge is unique, so emplace can't fail
		auto fret = face_halfedges[halfedges[h].face_idx].emplace(h);
		assert(fret.second); //every halfedge is unique, so emplace can't fail
	}

	//check edge->halfedge(->twin)^n:
	for (uint32_t e = 0; e < edges.size(); e++) {
		std::unordered_set< uint32_t > to_visit = edge_halfedges[e];
		std::string path = "halfedge";
		uint32_t h = edges[e].halfedge_idx;
		do {
			if (halfedges[h].edge_idx != e) return {{e, describe_edge(e) + " has " + path + " of " + describe_halfedge(h) + ", which does not reference the edge."}};
			auto found = to_visit.find(h);
			if (found == to_visit.end()) return {{e, describe_edge(e) + " has halfedge(->twin)^n which is not a cycle."}};
			to_visit.erase(found);

			h = halfedges[h].twin_idx;
			path += "->twin";
		} while (h != edges[e].halfedge_idx);
		if (!to_visit.empty()) return {{e, describe_edge(e) + " is referenced by " + describe_halfedge(*to_visit.begin()) + ", which is not in halfedge(->twin)^n."}};
		if (edge_halfedges[e].size() != 2) return {{e, describe_edge(e) + " has " + std::to_string(edge_halfedges[e].size()) + " (!= 2) elements in its halfedge(->twin)^n cycle."}};
	}

	//check face->halfedge(->next)^n:
	for (uint32_t f = 0; f < faces.size(); f++) {
		std::unordered_set< uint32_t > to_visit = face_halfedges[f];
		std::string path = "halfedge";
		uint32_t h = faces[f].halfedge_idx;
		do {
			if (halfedges[h].face_idx != f) return {{f, describe_face(f) + " has " + path + " of " + describe_halfedge(h) + ", which does not reference the face."}};
			auto found = to_visit.find(h);
			if (found == to_visit.end()) return {{f, describe_face(f) + " has halfedge(->next)^n which is not a cycle."}};
			to_visit.erase(found);

			h = halfedges[h].next_idx;
			path += "->next";
		} while (h != faces[f].halfedge_idx);
		if (!to_visit.empty()) return {{f, describe_face(f) + " is referenced by " + describe_halfedge(*to_visit.begin()) + ", which is not in halfedge(->next)^n."}};
		if (face_halfedges[f].size() < 3) return {{f, describe_face(f) + " has " + std::to_string(face_halfedges[f].size()) + " (< 3) elements in its halfedge(->next)^n cycle."}};
	}

	//check vertex->halfedge(->twin->next)^n:
	for (uint32_t v = 0; v < vertices.size(); v++) {
		std::unordered_set< uint32_t > to_visit = vertex_halfedges[v];
		std::string path = "halfedge";
		uint32_t h = vertices[v].halfedge_idx;
		do {
			if (halfedges[h].vertex_idx != v) return {{v, describe_vertex(v) + " has " + path + " of " + describe_halfedge(h) + ", which does not reference the vertex."}};
			auto found = to_visit.find(h);
			if (found == to_visit.end()) return {{v, describe_vertex(v) + " has halfedge(->twin->next)^n which is not a cycle."}};
			to_visit.erase(found);

			h = halfedges[halfedges[h].twin_idx].next_idx;
			path += "->twin->next";
		} while (h != vertices[v].halfedge_idx);
		if (!to_visit.empty()) return {{v, describe_vertex(v) + " is referenced by " + describe_halfedge(*to_visit.begin()) + ", which is not in halfedge(->twin->next)^n."}};
		if (vertex_halfedges[v].size() < 2) return {{v, describe_vertex(v) + " has " + std::to_string(vertex_halfedges[v].size()) + " (< 2) elements in its halfedge(->twin->next)^n cycle."}};
	}

	//------------------------------
	// - vertices are not orphaned (they have at least one non-boundary face adjacent)
	// - vertices are on at most one boundary face

	for (uint32_t v = 0; v < vertices.size(); v++) {
		uint32_t non_boundary = 0;
		uint32_t boundary = 0;
		uint32_t h = vertices[v].halfedge_idx;
		do {
			if (faces[halfedges[h].face_idx].boundary) ++boundary;
			else ++non_boundary;

			h = halfedges[halfedges[h].twin_idx].next_idx;
		} while (h != vertices[v].halfedge_idx);

		if (non_boundary == 0) return {{v, describe_vertex(v) + " is orphaned (has no adjacent non-boundary faces)."}};
		if (boundary > 1) return {{v, describe_vertex(v) + " is on " + std::to_string(boundary) + " (> 1) boundary faces."}};
	}

	//------------------------------
	// - edges are not orphaned (they have at least one non-boundary face adjacent)
	for (uint32_t e = 0; e != edges.size(); e++) {
		if (faces[halfedges[edges[e].halfedge_idx].face_idx].boundary && 
				faces[halfedges[halfedges[edges[e].halfedge_idx].twin_idx].face_idx].boundary)
			return {{e, describe_edge(e) + " is orphaned (has no adjacent non-boundary face)."}};
	}


	//------------------------------
	// - faces are simple (touch each vertex / edge at most once)
	for (uint32_t f = 0; f < faces.size(); f++) {
		std::unordered_set< uint32_t > touched_vertices;
		std::unordered_set< uint32_t > touched_edges;

		uint32_t h = faces[f].halfedge_idx;
		do {
			if (!touched_vertices.emplace(halfedges[h].vertex_idx).second) return {{f, describe_face(f) + " touches " + describe_vertex(halfedges[h].vertex_idx) + " more than once."}};
			if (!touched_edges.emplace(halfedges[h].edge_idx).second) return {{f, describe_face(f) + " touches " + describe_edge(halfedges[h].edge_idx) + " more than once."}};

			h = halfedges[h].next_idx;
		} while (h != faces[f].halfedge_idx);
	}

	return std::nullopt;
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