#pragma once 

#include <vector>
#include "mathlib.h"

class Mesh {
public:
    class Vertex;
    class Edge;
    class Halfedge;
    class Face;

    class Vertex {
        public:
            
            Halfedge* halfedge; // one halfedge exiting from this vertex
            Vec3 position;
            uint32_t id;
            uint32_t color;

            // TODO: implement vertex functions
            // bool on_boundary() const;
            // uint32_t degree() const;
            // Vec3 neighborhood_center() const;
        private:
            Vertex() = default;
            Vertex(uint32_t _id) : id(_id) {};
        friend class Mesh;
    };

    class Edge {
        public:
            Halfedge* halfedge; // one of the halfedges composing this edge
            uint32_t id;
            uint32_t color;
            bool sharp;
            
            // TODO: implement edge functions
            // bool on_boundary() const;
            // float length() const;
        private:
            Edge() = default;
            Edge(uint32_t _id, bool _sharp) : id(_id), sharp(_sharp) {};
        friend class Mesh;
    };

    class Halfedge {
        public:
            Halfedge* twin; // the halfedge on the other side of the edge
            Halfedge* next; // the next halfedge going counter-clockwise around the face
            Vertex* vertex; // the vertex this halfedge leaves from 
            Edge* edge; // the edge this halfedge is along
            Face* face; // the face this halfedge is along

            uint32_t id;
        private:
            Halfedge() = default;
            Halfedge(uint32_t _id) : id(_id) {};
        friend class Mesh;
    };

    class Face {
        public:
            Halfedge* halfedge; // one halfedge along the face

            uint32_t id;
            bool boundary = false; // is boundary loop

            // TODO: implement face functions
            // float area() const; // area of face;
        private:
            Face() = default;
            Face(uint32_t _id, bool _boundary) : id(_id), boundary(_boundary) {};
        friend class Mesh;
    };

    // next id to be assigned to an element of each type
    uint32_t next_v_id;
    uint32_t next_h_id;
    uint32_t next_e_id;
    uint32_t next_f_id;

    // list of elements of each type composing this mesh
    std::vector<Vertex> vertices;
    std::vector<Edge> edges;
    std::vector<Halfedge> halfedges;
    std::vector<Face> faces;

    // create a mesh from a list of vertices and a list of polygons composed of the vertices
    Mesh from_indexed_faces(std::vector< Vec3 > const &vertices_,  std::vector< std::vector< uint32_t > > const &faces_);

    Vertex emplace_vertex();
    Edge emplace_edge(bool sharp);
    Face emplace_face(bool boundary);
    Halfedge emplace_halfedge();
// bool validate() const;
};