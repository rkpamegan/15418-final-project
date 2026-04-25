# Project Context: CUDA Parallel Isotropic Remeshing

## Project Overview
CMU 15-418 final project. Two students: Junbo + Ryan. Implementing parallel isotropic remeshing on GPU using CUDA.

## Repo
- Local: `/Users/nicosama/CMU/15618/Final_Project/15418-final-project`
- Remote: `https://github.com/rkpamegan/15418-final-project.git`
- On GHC: `~/private/15618/15418-final-project`

## Tech Stack
- CUDA 11.7, GHC machines (P100, compute_61)
- Halfedge mesh, **index-based** (uint32_t), `INVALID_IDX = UINT32_MAX` marks deleted/invalid
- Thrust library: `exclusive_scan` (prefix sum), `reduce` (sum/avg), `max_element`
- Jones-Plassmann graph coloring for parallel safety

## Code Structure
```
src/
  cudaRemesh.h     - CudaRemesher class, Isotropic_Remesh_Params
  cudaRemesh.cu    - All CUDA kernels + isotropic_remesh() orchestrator
  mesh.h / mesh.cpp - Halfedge mesh data structure (host)
  vec3.h           - Vec3 with __host__ __device__ operators
  main.cpp         - Currently calls test_smooth_vertex()
  test.cpp         - test_split_edge(), test_smooth_vertex()
  Makefile
```

## Pipeline (in `isotropic_remesh()`)
For each iteration:
1. **Edge coloring** (Jones-Plassmann)
2. **Flip** edges that improve degree regularity
3. **Split** edges longer than `avg * split_factor` (uses prefix sum + array realloc)
4. **Collapse** edges shorter than `avg * collapse_factor` (mark-invalid pattern, no shrink)
5. **Vertex coloring** + **smooth** (tangential, double buffered via vertex_pos[])

**Pattern for each op (flip/split/collapse):**
- "get" kernel marks op_mask
- coloring ensures parallel safety
- "do" kernel loops by color: `for (c=0..max_color) kernel_X<<<>>>(... , c)`

## Status
- ✅ All kernels implemented (color_vertices, color_edges, get_vertex_normals, smooth_vertex,
  update_vertex_pos, get_flip_edges, flip_edge, get_edge_lengths, get_split_edges, split_edge,
  get_collapse_edges, collapse_edge)
- ✅ `isotropic_remesh()` fully wired: color → flip → split → collapse → smooth
- ⬜ **Not yet compiled on GHC** — next step: `make` and fix any errors
- ⬜ Correctness testing not started
- ⬜ Benchmark vs serial not started

## Key Implementation Details

### kernel_split_edge
- Uses `split_offsets` (prefix sum of op_mask) so each thread knows where to write
- Creates per split: 1 vertex, 3 edges, 6 halfedges, 2 faces at midpoint M
- Arrays must be reallocated BEFORE calling (host loops `cudaMalloc` larger arrays + memcpy)

### kernel_collapse_edge
- Diamond structure: edge B-C with two triangles (f0=ABC, f1=BDC)
- Moves B to midpoint of B and C
- Walks around C, redirects all halfedges from C → B
- Merges twin pairs on the diamond boundary (hn_twin↔hp_twin, tn_twin↔tp_twin)
- Marks 1V, 3E, 6H, 2F as INVALID_IDX (no array shrink)
- Updates vA, vB, vD halfedge pointers to surviving outer halfedges

### Memory pattern after split
Host code reallocates `cudaDeviceVertices/Edges/Halfedges/Faces` to new sizes.
Edge-sized arrays (`edge_lengths`, `edge_color_mask`, `edge_op_mask`, `edge_priorities`)
are also reallocated. `gridDim` updated for new edge count.

## Test Entry Point
`main.cpp` → `test_smooth_vertex()` (cube mesh, 8 vertices, 6 quad faces split into tris)
or `test_split_edge()` (small triangle mesh).

## Next Steps on GHC
1. `git pull`
2. `cd src && make`
3. Fix compilation errors
4. Run `./remesh` on small test mesh
5. Check for CUDA runtime errors / crashes
6. Visual check output mesh
7. Benchmark
