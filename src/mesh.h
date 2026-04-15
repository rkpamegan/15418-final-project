#pragma once 

#include <vector>
#include <stdint.h>
#include "mathlib.h"

#define INVALID_IDX UINT32_MAX

class Mesh {
public:
    class Vertex;
    class Edge;
    class Halfedge;
    class Face;

    class Vertex {
        public:
            
            uint32_t halfedge_idx; // index of one halfedge exiting from this vertex
            Vec3 position;
            uint32_t id;
            uint32_t color;

            // TODO: implement vertex functions
            // bool on_boundary() const;
            // uint32_t degree() const;
            // Vec3 neighborhood_center() const;
            std::string to_string() const;
        private:
            Vertex() : halfedge_idx(INVALID_IDX) {};
            Vertex(uint32_t _id) : id(_id), halfedge_idx(INVALID_IDX) {};
        friend class Mesh;
    };

    class Edge {
        public:
            uint32_t halfedge_idx; // index of one of the halfedges composing this edge
            uint32_t id;
            uint32_t color;
            bool sharp;
            
            // TODO: implement edge functions
            // bool on_boundary() const;
            std::string to_string() const;
        private:
            Edge() : halfedge_idx(INVALID_IDX) {};
            Edge(uint32_t _id, bool _sharp) : id(_id), sharp(_sharp), halfedge_idx(INVALID_IDX) {};
        friend class Mesh;
    };

    class Halfedge {
        public:
            uint32_t twin_idx; // index of the halfedge on the other side of the edge
            uint32_t next_idx; // index of the next halfedge going counter-clockwise around the face
            uint32_t vertex_idx; // index of the vertex this halfedge leaves from 
            uint32_t edge_idx; // index of the edge this halfedge is along
            uint32_t face_idx; // index of the face this halfedge is along

            uint32_t id;
            Halfedge() : twin_idx(INVALID_IDX), next_idx(INVALID_IDX), vertex_idx(INVALID_IDX), edge_idx(INVALID_IDX), face_idx(INVALID_IDX) {};
            std::string to_string() const;
        private:
            Halfedge(uint32_t _id) : id(_id), twin_idx(INVALID_IDX), next_idx(INVALID_IDX), vertex_idx(INVALID_IDX), edge_idx(INVALID_IDX), face_idx(INVALID_IDX) {};
        friend class Mesh;
    };

    class Face {
        public:
            uint32_t halfedge_idx; // index of one halfedge along the face

            uint32_t id;
            bool boundary = false; // is boundary loop

            // TODO: implement face functions
            // float area() const; // area of face;
            std::string to_string() const;
        private:
            Face() : halfedge_idx(INVALID_IDX) {};
            Face(uint32_t _id, bool _boundary) : id(_id), boundary(_boundary), halfedge_idx(INVALID_IDX) {};
        friend class Mesh;
    };

    // list of elements of each type composing this mesh
    std::vector<Vertex> vertices;
    std::vector<Edge> edges;
    std::vector<Halfedge> halfedges;
    std::vector<Face> faces;

    // create a mesh from a list of vertices and a list of polygons composed of the vertices
    static Mesh from_indexed_faces(std::vector< Vec3 > const &vertices_,  
        std::vector< std::vector< uint32_t > > const &faces_);
    // bool validate() const;
    void describe() const;
    Mesh() = default;
    ~Mesh() = default;
    private:
        // next id to be assigned to an element of each type
        uint32_t next_v_id = 0;
        uint32_t next_h_id = 0;
        uint32_t next_e_id = 0;
        uint32_t next_f_id = 0;

        uint32_t emplace_vertex();
        uint32_t emplace_edge(bool sharp);
        uint32_t emplace_face(bool boundary);
        uint32_t emplace_halfedge();
};