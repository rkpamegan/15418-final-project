# 15-418 Final Project: Parallel Remeshing

Ryan Kpamegan (rkpamega), Junbo Huang (junboh)

## Proposal

### Summary

We propose a parallel implementation of a remeshing algorithm using CUDA. The program will take modify a mesh to be more uniform.

### Background

In computer graphics, a mesh is the surface of a 3D object, defined by its vertices and edges, along with faces, which are finite planes enclosed by a set of vertices and edges. In general, "good" meshes are regular: they have vertices with similar neighborhood size, edges with similar lengths, and faces with roughly equal angles. There are of course exceptions, but these properties are the target most of the time. <br>
![Averaging vertex positions](https://github.com/rkpamegan/15418-final-project/blob/main/docs/background_avg.png?raw=true)
![Splitting long edges](https://github.com/rkpamegan/15418-final-project/blob/main/docs/background_split.png?raw=true)
![Collapsing short edges](https://github.com/rkpamegan/15418-final-project/blob/main/docs/background_collapse.png?raw=true)

A remeshing algorithm takes in a mesh as input and manipulates its elements to give it the desired properties. The algorithm we will use can add or remove edges, and move vertices, as shown in the above figures. This is very computationally heavy, as each vertex and edge in the entire mesh must be iterated over. As the level of detail on a mesh increases, so does the number of objects, and thus the remeshing becomes more expensive. When the algorithm is isotropic the same steps are applied to every vertex and edge indiscriminately, which lends itself to parallelism and can make remeshing much faster.

### The Challenge

The biggest challenge to implementing this project is the race conditions for modifying the graph underlying the mesh. For example, when adjusting edges, we have the option to split the edge into two and add a vertex in between, thus halving the edge length. This requires updating the surrounding vertices and faces so that they include the new edges. Similarly, we can collapse the edge, pulling its two incident vertices into one, which also requires updates to surrounding elements. If done naively in parallel, this will cause data races, where some elements can refer to others that don't exist, or fail to refer to newly created elements. Locality introduces an even bigger problem, as we would like to keep nearby elements in the cache for later computation while also keeping them far apart enough to not affect each other. 

Divergent execution also presents a problem for the algorithm. As mentioned above, edges can be either split or removed during remeshing. Thus, it is possible to lose lots of performance when many edges undergo different operations. We define length thresholds during the routine for which operation should be performed on an edge (or whether or not it is performed at all); the value of these thresholds can impact scaling due to this divergent execution.

### Resources

We plan to mainly use [Fall 2025 15-362 Computer Graphics](https://15362.courses.cs.cmu.edu/fall2025/home) lecture notes  and assignments as a base for the structure and logic of the algorithm. We may use test cases from the assignments for our own implementation. We will also use Assignment 2 from this course as a reference for writing CUDA code.

### Goals and Deliverables

The goal we plan to achieve is the implement parallel isotropic remeshing, with a speedup of 6-7x on 8 cores. The goal we hope to achieve if all goes well is another remeshing algorithm that uses [Centroidal Voronoi Diagrams](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2004.00769.x), although we have not yet evaluated its potential for parallelism.

### Platform Choice

The project will be implemented using CUDA on the GHC machines. Isotropic remeshing (or at least most of it) is inherently SIMD and thus CUDA is a natural choice.  The GHC machines are easily accessible and relatively inexpensive to use, allowing us to rapidly develop our program.

### Schedule

| Week of | Item                                            |
|:-------:| :---------------------------------------------- |
| 4/6     | Establish base code, generate small test cases  |
| 4/13    | Begin parallel implementation                   |
| 4/20    | Finish parallel implementation                  |
| 4/27    | Optimize performance                            |

## Milestone Report
