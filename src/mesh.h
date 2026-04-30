#pragma once 

#include <vector>
#include <optional>
#include <stdint.h>
#include "mathlib.h"

#define INVALID_IDX UINT32_MAX

//improve mesh quality via isotropic remeshing
struct Isotropic_Remesh_Params {
    uint32_t num_iters; //how many outer loops through the remeshing process to take
    float split_factor; //edges longer than longer_factor * target_length are split
    float collapse_factor; //edges shorter than shorter_factor * target_length are collapsed
    uint32_t smoothing_iters; //how many tangential smoothing iterations to run
    float smoothing_step; //amount to interpolate vertex positions toward their centroid each smoothing step
    uint32_t block_size = 256; //CUDA threads per block
    uint32_t num_blocks = 0;   //CUDA blocks (0 = auto: ceil(N/block_size))
};

void cuda_clear_last_error();

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
            Vertex() : halfedge_idx(INVALID_IDX) {};
        private:
            Vertex(uint32_t _id) : halfedge_idx(INVALID_IDX),  id(_id) {};
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
            Edge() : halfedge_idx(INVALID_IDX) {};
        private:
            Edge(uint32_t _id, bool _sharp) : halfedge_idx(INVALID_IDX), id(_id), sharp(_sharp)  {};
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
            Halfedge(uint32_t _id) : twin_idx(INVALID_IDX), next_idx(INVALID_IDX), vertex_idx(INVALID_IDX), edge_idx(INVALID_IDX), face_idx(INVALID_IDX), id(_id) {};
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
            Face() : halfedge_idx(INVALID_IDX) {};
        private:
            Face(uint32_t _id, bool _boundary) :  halfedge_idx(INVALID_IDX), id(_id), boundary(_boundary) {};
        friend class Mesh;
    };

    // list of elements of each type composing this mesh
    std::vector<Vertex> vertices;
    std::vector<Edge> edges;
    std::vector<Halfedge> halfedges;
    std::vector<Face> faces;

    // remesh functions
    uint32_t vertex_degree(uint32_t v);
    float edge_length(uint32_t e);
    void flip_edge(uint32_t e);
    void flip_edges();
    void split_edge(uint32_t e);
    uint32_t split_edges(float avg_len, float split_factor);
    void collapse_edge(uint32_t e);
    void collapse_edges(float avg_len, float collapse_factor);
    void smooth_vertices(std::vector<Vec3>& vertex_pos, std::vector<Vec3>& vertex_normals, float smoothing_factor);
    void get_vertex_normals(std::vector<Vec3>& vertex_normals);
    void update_vertex_pos(std::vector<Vec3>& vertex_pos);
    void isotropic_remesh(Isotropic_Remesh_Params const &params);

    /**
     * Create a mesh from a list of vertices and a list of polygons composed of the vertices.
     * Vertices for each face MUST be specified in counter-clockwise order
     */ 

    static Mesh from_indexed_faces(std::vector< Vec3 > const &vertices_,  
        std::vector< std::vector< uint32_t > > const &faces_);
    std::optional<std::pair<uint32_t, std::string>> validate() const;
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