#include "mesh.h"
#include <stdint.h>

//improve mesh quality via isotropic remeshing
struct Isotropic_Remesh_Params {
    uint32_t num_iters; //how many outer loops through the remeshing process to take
    float split_factor; //edges longer than longer_factor * target_length are split
    float collapse_factor; //edges shorter than shorter_factor * target_length are collapsed
    uint32_t smoothing_iters; //how many tangential smoothing iterations to run
    float smoothing_step; //amount to interpolate vertex positions toward their centroid each smoothing step
};

class CudaRemesher {
    private:
        Mesh* mesh; // the mesh to be remeshed

        // count of each element in the mesh
        uint32_t numVertices;
        uint32_t numEdges;
        uint32_t numHalfedges;
        uint32_t numFaces;

        // the elements of the mesh on the CUDA device
        Mesh::Vertex* cudaDeviceVertices;
        Mesh::Edge* cudaDeviceEdges;
        Mesh::Halfedge* cudaDeviceHalfedges;
        Mesh::Face* cudaDeviceFaces;

        float* edge_lengths; // the length of each edge
        int* edge_color_mask; // the color of each edge
        int* edge_op_mask; // denotes if an operation will be performed on an edge
        float* vertex_pos; // new vertex positions in the form of [x1, y1, z1, x2, y2, z2, ...]
        int* vertex_color_mask; // the color of each vertex

        void update_mesh();
    public:
        CudaRemesher();
        ~CudaRemesher();

        void setup(Mesh &mesh);
        
        void isotropic_remesh(Isotropic_Remesh_Params const &params);
};