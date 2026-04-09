#include "mesh.h"

//improve mesh quality via isotropic remeshing
struct Isotropic_Remesh_Params {
    uint32_t num_iters; //how many outer loops through the remeshing process to take
    float longer_factor; //edges longer than longer_factor * target_length are split
    float shorter_factor; //edges shorter than shorter_factor * target_length are collapsed
    uint32_t smoothing_iters; //how many tangential smoothing iterations to run
    float smoothing_step; //amount to interpolate vertex positions toward their centroid each smoothing step
};

class CudaRemesher {
    private:
        Mesh mesh;
    public:
        void setup(Mesh mesh);
        
        void isotropic_remesh(Isotropic_Remesh_Params const &params);
};