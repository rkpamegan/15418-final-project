#include "mesh.h"
#include <stdint.h>

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
        int* split_offsets; // exclusive prefix sum of op_mask for split indexing
        Vec3* vertex_pos; // new vertex positions in the form of [x1, y1, z1, x2, y2, z2, ...]
        Vec3* vertex_normals; // normal of vertex
        int* vertex_color_mask; // the color of each vertex
        int* vertex_priorities; // random priority for graph coloring
        int* edge_priorities; // random priority for edge graph coloring
        bool* d_coloring_done; // device flag for coloring convergence

        void update_mesh();
    public:
        CudaRemesher();
        ~CudaRemesher();

        void setup(Mesh &mesh);
        
        void isotropic_remesh(Isotropic_Remesh_Params const &params);
};