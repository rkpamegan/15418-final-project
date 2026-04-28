CudaRemesher::CudaRemesher() {
	vertices = NULL;
	halfedges = NULL;
	faces = NULL;
	edges = NULL;

	numVertices = 0;
	numEdges = 0;
	numHalfedges = 0;
	numFaces = 0;

	float* edge_lengths = NULL;
    int* edge_color_mask = NULL;
	int* edge_op_mask = NULL;
	int* vertex_color_mask = NULL;
	Vec3* vertex_pos = NULL;
	Vec3* vertex_normals = NULL;
}