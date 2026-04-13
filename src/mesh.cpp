#include "mesh.h"
#include <stdint.h>

#include <unordered_map>
#include <set>
#include <map>

#include <iostream>

// float Mesh::Edge::length() const {
// 	Vertex* const v1 = halfedge->vertex;
// 	Vertex* const v2 = halfedge->twin->vertex;

// 	Vec3 d = v1->position - v2->position;
// 	return d.norm();
// }
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

	std::vector< Vertex > vertices; //for quick lookup of vertices by index
	vertices.reserve(vertices_.size());
	for (auto const &v : vertices_) {
		vertices.emplace_back(mesh.emplace_vertex());
		vertices.back().position = v;
	}


	std::unordered_map< std::pair< uint32_t, uint32_t >, Halfedge > halfedges; //for quick lookup of halfedges by from/to vertex index

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

		Face f = mesh.emplace_face(boundary);
		Face* face = &f;
		Halfedge* prev = nullptr; //keep track of previous edge around face to set next pointer
		for (uint32_t i = 0; i < loop.size(); ++i) {
			uint32_t a = loop[i];
			uint32_t b = loop[(i + 1) % loop.size()];
			assert(a != b);

			
			Halfedge hf = mesh.emplace_halfedge();
			Halfedge* halfedge = &hf;
			if (i == 0) face->halfedge = halfedge; //store first edge as face's halfedge pointer
			halfedge->vertex = &vertices[a];

			//if first to mention vertex, set vertex's halfedge pointer:
			if ((&vertices[a])->halfedge == nullptr) {
				assert(!boundary); //boundary faces should never be mentioning novel vertices, since they are created second
				(&vertices[a])->halfedge = halfedge;
			}
			halfedge->face = face;

			auto inserted = halfedges.emplace(std::make_pair(a,b), *halfedge);
			assert(inserted.second); //if edge mentioned more than once in the same direction, not an oriented, manifold mesh

			auto twin = halfedges.find(std::make_pair(b,a));
			if (twin == halfedges.end()) {
				assert(!boundary); //boundary faces exist only to complete edges so should *always* match
				//not twinned yet -- create an edge just for this halfedge:
				Edge e = mesh.emplace_edge(false);
				Edge* edge = &e;
				halfedge->edge = edge;
				edge->halfedge = halfedge;
			} else {
				//found a twin -- connect twin pointers and reference its edge:
				assert((&(twin->second))->twin == nullptr);
				halfedge->twin = &twin->second;
				halfedge->edge = (&(twin->second))->edge;
				(&(twin->second))->twin = halfedge;
			}

			if (i != 0) prev->next = halfedge; //set previous halfedge's next pointer
			prev = halfedge;
		}

		prev->next = face->halfedge; //set next pointer for last halfedge to first edge
	};
	
	//add all faces:
	for (uint32_t i = 0; i < num_faces; i++) {
		add_loop(faces_[i], false);
	}
	std::printf("%d\n", mesh.vertices.size());
	for (size_t i = 0; i < mesh.vertices.size(); i++)
	{
		std::cout << mesh.vertices[i] << std::endl;
	}

	for (size_t i = 0; i < mesh.edges.size(); i++)
	{
		std::cout << mesh.edges[i] << std::endl;
	}

	// for (size_t i = 0; i < mesh.halfedges.size(); i++)
	// {
	// 	std::cout << mesh.halfedges[i] << std::endl;
	// }

	// for (size_t i = 0; i < mesh.faces.size(); i++)
	// {
	// 	std::cout << mesh.faces[i] << std::endl;
	// }
	

	// All halfedges created so far have valid next pointers, but some may be missing twins because they are at a boundary.
	
	std::map< uint32_t, uint32_t > next_on_boundary;


	//first, look for all un-twinned halfedges to figure out the shape of the boundary:
	for (auto const &[ from_to, halfedge ] : halfedges) {
		std::printf("(%d -> %d)\n", from_to.first, from_to.second);
		if (halfedge.twin == nullptr) {
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

Mesh::Vertex Mesh::emplace_vertex() {
	Vertex vertex;
	vertex = vertices.emplace_back(Vertex(next_v_id++));
	//make sure vertex doesn't reference anything:
	vertex.halfedge = NULL;
	std::printf("%p %p\n", &vertices, &vertex);
	return vertex;
}

Mesh::Edge Mesh::emplace_edge(bool sharp) {
	Edge edge;
	//allocate a new edge:
    edge = edges.emplace_back(Edge(next_e_id++, sharp));
	//make sure edge doesn't reference anything:
	edge.halfedge = nullptr;
	return edge;
}

Mesh::Face Mesh::emplace_face(bool boundary) {
	Face face;
    face = faces.emplace_back(Face(next_f_id++, boundary));
	face.halfedge = nullptr;
	return face;
}

Mesh::Halfedge Mesh::emplace_halfedge() {
	Halfedge halfedge;
    //allocate a new halfedge:
    halfedge = halfedges.emplace_back(Halfedge(next_h_id++));
	//set pointers to default values:
	halfedge.twin = nullptr;
	halfedge.next = nullptr;
	halfedge.vertex = nullptr;
	halfedge.edge = nullptr;
	halfedge.face = nullptr;
	return halfedge;
}

std::string Mesh::Vertex::to_string() const {
	std::string s = "v" + std::to_string(id);
	if (halfedge != NULL) {
		s += " h" + std::to_string(halfedge->id);
		std::printf("%s\n", halfedge->to_string());
	}
	return s;
}

std::ostream& operator << (std::ostream& outs, const Mesh::Vertex& v) {
	return outs << v.to_string();
}


std::string Mesh::Edge::to_string() const {
	std::string s = "e" + std::to_string(id);
	if (halfedge != NULL) s += " h" + std::to_string(halfedge->id); 
	return s;
}
std::ostream& operator << (std::ostream& outs, const Mesh::Edge& e) {
	return outs << e.to_string();
}
std::string Mesh::Halfedge::to_string() const {
	std::string s = "h" + std::to_string(id);
	if (twin) s += " t" + std::to_string(twin->id);
	if (next) s += " n" + std::to_string(next->id); 
	if (vertex) s += " v" + std::to_string(vertex->id); 
	if (edge) s += " e" + std::to_string(edge->id);
	if (face) s += " f" + std::to_string(face->id);
	return s;
}
std::ostream& operator << (std::ostream& outs, const Mesh::Halfedge& h) {
	return outs << h.to_string();
}
std::string Mesh::Face::to_string() const {
	std::string s = "f" + std::to_string(id);
	if (halfedge) s += " h" + std::to_string(halfedge->id);
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