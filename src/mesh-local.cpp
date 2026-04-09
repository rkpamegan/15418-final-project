
#include "mesh.h"
#include "mathlib.h"

#include <assert.h>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <iostream>

/******************************************************************
*********************** Local Operations **************************
******************************************************************/

/* Note on local operation return types:

    The local operations all return a std::optional<T> type. This is used so that your
    implementation can signify that it cannot perform an operation (i.e., because
    the resulting mesh does not have a valid representation).

    An optional can have two values: std::nullopt, or a value of the type it is
    parameterized on. In this way, it's similar to a pointer, but has two advantages:
    the value it holds need not be allocated elsewhere, and it provides an API that
    forces the user to check if it is null before using the value.

    In your implementation, if you have successfully performed the operation, you can
    simply return the required reference:

            ... collapse the edge ...
            return collapsed_vertex_ref;

    And if you wish to deny the operation, you can return the null optional:

            return std::nullopt;

    Note that the stubs below all reject their duties by returning the null optional.
*/


/*
 * add_face: add a standalone face to the mesh
 *  sides: number of sides
 *  radius: distance from vertices to origin
 *
 * We provide this method as an example of how to make new halfedge mesh geometry.
 */
std::optional<Mesh::FaceRef> Mesh::add_face(uint32_t sides, float radius) {
	//faces with fewer than three sides are invalid, so abort the operation:
	if (sides < 3) return std::nullopt;


	std::vector< VertexRef > face_vertices;
	//In order to make the first edge point in the +x direction, first vertex should
	// be at -90.0f - 0.5f * 360.0f / float(sides) degrees, so:
	float const start_angle = (-0.25f - 0.5f / float(sides)) * 2.0f * PI_F;
	for (uint32_t s = 0; s < sides; ++s) {
		float angle = float(s) / float(sides) * 2.0f * PI_F + start_angle;
		VertexRef v = emplace_vertex();
		v->position = radius * Vec3(std::cos(angle), std::sin(angle), 0.0f);
		face_vertices.emplace_back(v);
	}

	assert(face_vertices.size() == sides);

	//assemble the rest of the mesh parts:
	FaceRef face = emplace_face(false); //the face to return
	FaceRef boundary = emplace_face(true); //the boundary loop around the face

	std::vector< HalfedgeRef > face_halfedges; //will use later to set ->next pointers

	for (uint32_t s = 0; s < sides; ++s) {
		//will create elements for edge from a->b:
		VertexRef a = face_vertices[s];
		VertexRef b = face_vertices[(s+1)%sides];

		//h is the edge on face:
		HalfedgeRef h = emplace_halfedge();
		//t is the twin, lies on boundary:
		HalfedgeRef t = emplace_halfedge();
		//e is the edge corresponding to h,t:
		EdgeRef e = emplace_edge(false); //false: non-sharp

		//set element data to something reasonable:
		//(most ops will do this with interpolate_data(), but no data to interpolate here)
		h->corner_uv = a->position.xy() / (2.0f * radius) + 0.5f;
		h->corner_normal = Vec3(0.0f, 0.0f, 1.0f);
		t->corner_uv = b->position.xy() / (2.0f * radius) + 0.5f;
		t->corner_normal = Vec3(0.0f, 0.0f,-1.0f);

		//thing -> halfedge pointers:
		e->halfedge = h;
		a->halfedge = h;
		if (s == 0) face->halfedge = h;
		if (s + 1 == sides) boundary->halfedge = t;

		//halfedge -> thing pointers (except 'next' -- will set that later)
		h->twin = t;
		h->vertex = a;
		h->edge = e;
		h->face = face;

		t->twin = h;
		t->vertex = b;
		t->edge = e;
		t->face = boundary;

		face_halfedges.emplace_back(h);
	}

	assert(face_halfedges.size() == sides);

	for (uint32_t s = 0; s < sides; ++s) {
		face_halfedges[s]->next = face_halfedges[(s+1)%sides];
		face_halfedges[(s+1)%sides]->twin->next = face_halfedges[s]->twin;
	}

	return face;
}


/*
 * bisect_edge: split an edge without splitting the adjacent faces
 *  e: edge to split
 *
 * returns: added vertex
 *
 * We provide this as an example for how to implement local operations.
 * (and as a useful subroutine!)
 */
std::optional<Mesh::VertexRef> Mesh::bisect_edge(EdgeRef e) {
	// Phase 0: draw a picture
	//
	// before:
	//    ----h--->
	// v1 ----e--- v2
	//   <----t---
	//
	// after:
	//    --h->    --h2->
	// v1 --e-- vm --e2-- v2
	//    <-t2-    <--t--
	//

	// Phase 1: collect existing elements
	HalfedgeRef h = e->halfedge;
	HalfedgeRef t = h->twin;
	VertexRef v1 = h->vertex;
	VertexRef v2 = t->vertex;

	// Phase 2: Allocate new elements, set data
	VertexRef vm = emplace_vertex();
	vm->position = (v1->position + v2->position) / 2.0f;
	interpolate_data({v1, v2}, vm); //set bone_weights

	EdgeRef e2 = emplace_edge();
	e2->sharp = e->sharp; //copy sharpness flag

	HalfedgeRef h2 = emplace_halfedge();
	interpolate_data({h, h->next}, h2); //set corner_uv, corner_normal

	HalfedgeRef t2 = emplace_halfedge();
	interpolate_data({t, t->next}, t2); //set corner_uv, corner_normal

	// The following elements aren't necessary for the bisect_edge, but they are here to demonstrate phase 4
    FaceRef f_not_used = emplace_face();
    HalfedgeRef h_not_used = emplace_halfedge();

	// Phase 3: Reassign connectivity (careful about ordering so you don't overwrite values you may need later!)

	vm->halfedge = h2;

	e2->halfedge = h2;

	assert(e->halfedge == h); //unchanged

	//n.b. h remains on the same face so even if h->face->halfedge == h, no fixup needed (t, similarly)

	h2->twin = t;
	h2->next = h->next;
	h2->vertex = vm;
	h2->edge = e2;
	h2->face = h->face;

	t2->twin = h;
	t2->next = t->next;
	t2->vertex = vm;
	t2->edge = e;
	t2->face = t->face;
	
	h->twin = t2;
	h->next = h2;
	assert(h->vertex == v1); // unchanged
	assert(h->edge == e); // unchanged
	//h->face unchanged

	t->twin = h2;
	t->next = t2;
	assert(t->vertex == v2); // unchanged
	t->edge = e2;
	//t->face unchanged


	// Phase 4: Delete unused elements
    erase_face(f_not_used);
    erase_halfedge(h_not_used);

	// Phase 5: Return the correct iterator
	return vm;
}


/*
 * split_edge: split an edge and adjacent (non-boundary) faces
 *  e: edge to split
 *
 * returns: added vertex. vertex->halfedge should lie along e
 *
 * Note that when splitting the adjacent faces, the new edge
 * should connect to the vertex ccw from the ccw-most end of e
 * within the face.
 *
 * Do not split adjacent boundary faces.
 */
std::optional<Mesh::VertexRef> Mesh::split_edge(EdgeRef e) {
	// A2L2 (REQUIRED): split_edge

	// Phase 1: collect
	VertexRef v1 = *bisect_edge(e); // bisect can't go wrong, right?
	HalfedgeRef h = e->halfedge->next;
	HalfedgeRef tp = h->twin;
	HalfedgeRef t = tp->next;

	FaceRef f1 = h->face;
	FaceRef f2 = t->face;

	std::vector<HalfedgeCRef> f1_halfedges;
	HalfedgeRef temp_h1 = f1->halfedge;
	do {
		f1_halfedges.emplace_back(temp_h1);
		temp_h1 = temp_h1->next;
	} while (temp_h1 != f1->halfedge);

	std::vector<HalfedgeCRef> f2_halfedges;
	HalfedgeRef temp_h2 = f2->halfedge;
	do {
		f2_halfedges.emplace_back(temp_h2);
		temp_h2 = temp_h2->next;
	} while (temp_h2 != f2->halfedge);
	// To keep simple, we want to always assume f1 is not a boundary face,
	// and f2 may or may not be a boundary face. The edge must have at least
	// one non-boundary face.
	if (f1->boundary)
	{
		std::swap(h, t);
		std::swap(f1, f2);
	}
	
	HalfedgeRef hn = h->next;
	HalfedgeRef tn = t->next;

	HalfedgeRef hp = h->next;
	while (hp->next != h) hp = hp->next;

	tp = t->next; // reset so we are sure we have the correct tp
	while (tp->next != t) tp = tp->next;

	VertexRef v2 = h->next->next->vertex;

	// Phase 2: create
	EdgeRef e2 = emplace_edge(false);
	HalfedgeRef h2 = emplace_halfedge();
	HalfedgeRef t2 = emplace_halfedge();
	FaceRef f3 = emplace_face(false);
	interpolate_data(f1_halfedges, h2);
	interpolate_data(f1_halfedges, t2);
	// Phase 3: connect
	e2->halfedge = h2;

	f3->halfedge = h2;

	h2->twin = t2;
	h2->next = h;
	h2->vertex = v2;
	h2->edge = e2;
	h2->face = f3;

	t2->twin = h2;
	t2->next = hn->next;
	t2->vertex = v1;
	t2->edge = e2;
	t2->face = f1;

	hn->next = h2;
	hn->face = f3;

	hp->next = t2;

	h->face = f3;

	f1->halfedge = t2;
	
	if (!f2->boundary) // do same thing on f2 side if its not a boundary
	{
		// more collect
		VertexRef v3 = t->next->next->vertex;

		// more create
		EdgeRef e3 = emplace_edge(false);
		HalfedgeRef h3 = emplace_halfedge();
		HalfedgeRef t3 = emplace_halfedge();
		FaceRef f4 = emplace_face(false);
		interpolate_data(f2_halfedges, h3);
		interpolate_data(f2_halfedges, t3);
		// more connect
		e3->halfedge = h3;

		f4->halfedge = t;

		h3->twin = t3;
		h3->next = t;
		h3->vertex = v3;
		h3->edge = e3;
		h3->face = f4;

		t3->twin = h3;
		t3->next = tn->next;
		t3->vertex = v1;
		t3->edge = e3;
		t3->face = f2;

		tn->face = f4;
		tn->next = h3;

		t->face = f4;

		tp->next = t3;
		
		f2->halfedge = t3;
		 
	}
	// (void)e; //this line avoids 'unused parameter' warnings. You can delete it as you fill in the function.
    return v1;
}



/*
 * inset_vertex: divide a face into triangles by placing a vertex at f->center()
 *  f: the face to add the vertex to
 *
 * returns:
 *  std::nullopt if insetting a vertex would make mesh invalid
 *  the inset vertex otherwise
 */
std::optional<Mesh::VertexRef> Mesh::inset_vertex(FaceRef f) {
	// A2Lx4 (OPTIONAL): inset vertex
	
	(void)f;
    return std::nullopt;
}


/* [BEVEL NOTE] Note on the beveling process:

	Each of the bevel_vertex, bevel_edge, and extrude_face functions do not represent
	a full bevel/extrude operation. Instead, they should update the _connectivity_ of
	the mesh, _not_ the positions of newly created vertices. In fact, you should set
	the positions of new vertices to be exactly the same as wherever they "started from."

	When you click on a mesh element while in bevel mode, one of those three functions
	is called. But, because you may then adjust the distance/offset of the newly
	beveled face, we need another method of updating the positions of the new vertices.

	This is where bevel_positions and extrude_positions come in: these functions are
	called repeatedly as you move your mouse, the position of which determines the
	amount / shrink parameters. These functions are also passed an array of the original
	vertex positions, stored just after the bevel/extrude call, in order starting at
	face->halfedge->vertex, and the original element normal, computed just *before* the
	bevel/extrude call.

	Finally, note that the amount, extrude, and/or shrink parameters are not relative
	values -- you should compute a particular new position from them, not a delta to
	apply.
*/

/*
 * bevel_vertex: creates a face in place of a vertex
 *  v: the vertex to bevel
 *
 * returns: reference to the new face
 *
 * see also [BEVEL NOTE] above.
 */
std::optional<Mesh::FaceRef> Mesh::bevel_vertex(VertexRef v) {
	//A2Lx5 (OPTIONAL): Bevel Vertex
	// Reminder: This function does not update the vertex positions.
	// Remember to also fill in bevel_vertex_helper (A2Lx5h)

	(void)v;
    return std::nullopt;
}

/*
 * bevel_edge: creates a face in place of an edge
 *  e: the edge to bevel
 *
 * returns: reference to the new face
 *
 * see also [BEVEL NOTE] above.
 */
std::optional<Mesh::FaceRef> Mesh::bevel_edge(EdgeRef e) {
	//A2Lx6 (OPTIONAL): Bevel Edge
	// Reminder: This function does not update the vertex positions.
	// remember to also fill in bevel_edge_helper (A2Lx6h)

	(void)e;
    return std::nullopt;
}

/*
 * extrude_face: creates a face inset into a face
 *  f: the face to inset
 *
 * returns: reference to the inner face
 *
 * see also [BEVEL NOTE] above.
 */
std::optional<Mesh::FaceRef> Mesh::extrude_face(FaceRef f) {
	//A2L4: Extrude Face
	// Reminder: This function does not update the vertex positions.
	// Remember to also fill in extrude_helper (A2L4h)
	if (f->boundary) return std::nullopt;

	// Phase 1: collect
	std::vector<VertexRef> vertices;
	std::vector<HalfedgeCRef> halfedges;
	std::vector<HalfedgeCRef> halfedge_twins;
	HalfedgeRef h_add = f->halfedge;
	do
	{
		halfedges.emplace_back(h_add);
		halfedge_twins.emplace_back(h_add->twin);
		vertices.emplace_back(h_add->vertex);
		h_add = h_add->next;
	} while (h_add != f->halfedge);
	int n = vertices.size();

	// Phase 2: create
	std::vector<FaceRef> new_faces;
	std::vector<VertexRef> new_vertices;
	std::vector<EdgeRef> edges_to_new;
	std::vector<EdgeRef> edges_between_new;

	std::vector<HalfedgeRef> halfedges_to_new;
	std::vector<HalfedgeRef> halfedges_between_new;

	for (int i = 0; i < n; i++)
	{
		FaceRef new_f = emplace_face(false);
		new_faces.emplace_back(new_f);

		VertexRef v = emplace_vertex();
		v->position = vertices[i]->position;
		new_vertices.emplace_back(v);

		EdgeRef e = emplace_edge(false);
		edges_to_new.emplace_back(e);

		EdgeRef e2 = emplace_edge(false);
		edges_between_new.emplace_back(e2);

		HalfedgeRef h_to_new = emplace_halfedge();
		HalfedgeRef t_to_new = emplace_halfedge();
		halfedges_to_new.emplace_back(h_to_new);
		halfedges_to_new.emplace_back(t_to_new);
		HalfedgeRef h_between_new = emplace_halfedge();
		HalfedgeRef t_between_new = emplace_halfedge();
		halfedges_between_new.emplace_back(h_between_new);
		halfedges_between_new.emplace_back(t_between_new);
	}

	// Phase 3: connect
	
	// for each vertex on the face, 
	// connect that vertex to the corresponding 
	// vertex in the extruded face
	HalfedgeRef ref_h = f->halfedge;
	while (ref_h->vertex != vertices[0]) ref_h = ref_h->next;

	for (int i = 0; i < n; i++)
	{
		HalfedgeRef h = halfedges_to_new[2*i];
		HalfedgeRef t = halfedges_to_new[2*i+1];
		EdgeRef e = edges_to_new[i];
		VertexRef v = vertices[i];
		VertexRef new_v = new_vertices[i];
		int prev_v_idx = ((i-1)+n) % n;

		h->twin = t;
		t->twin = h;

		h->next = ref_h;
		t->next = halfedges_between_new[2 * prev_v_idx];

		h->vertex = new_v;
		t->vertex = v;

		h->edge = e;
		t->edge = e;

		h->face = new_faces[i];
		new_faces[i]->halfedge = h;
		t->face = new_faces[prev_v_idx];
		
		new_v->halfedge = h;
		e->halfedge = h;

		HalfedgeRef next_ref = ref_h->next;
		ref_h->next = halfedges_to_new[2*((i+1)%n)+1];
		ref_h->face = new_faces[i];
		ref_h = next_ref;
	}

	for (int i = 0; i < n; i++)
	{
		HalfedgeRef h = halfedges_between_new[2*i];
		HalfedgeRef t = halfedges_between_new[2*i+1];
		EdgeRef e = edges_between_new[i];

		int next_v_idx = ((i+1)%n);
		h->twin = t;
		t->twin = h;

		h->next = halfedges_to_new[2*i];
		t->next = halfedges_between_new[2*next_v_idx+1];

		h->vertex = new_vertices[next_v_idx];
		t->vertex = new_vertices[i];

		h->edge = e;
		t->edge = e;

		h->face = new_faces[i];
		t->face = f;
		f->halfedge = t;

		e->halfedge = h;
	}

    return f;
}

/*
 * flip_edge: rotate non-boundary edge ccw inside its containing faces
 *  e: edge to flip
 *
 * if e is a boundary edge, does nothing and returns std::nullopt
 * if flipping e would create an invalid mesh, does nothing and returns std::nullopt
 *
 * otherwise returns the edge, post-rotation
 *
 * does not create or destroy mesh elements.
 */
std::optional<Mesh::EdgeRef> Mesh::flip_edge(EdgeRef e) {
	//A2L1: Flip Edge
	if (e->on_boundary()) return std::nullopt;

	// Phase 1: collect
	HalfedgeRef h =  e->halfedge;
	HalfedgeRef t = h->twin;

	VertexRef v1 = h->next->vertex;
	VertexRef v2 = t->next->vertex;

	VertexRef v3 = h->next->next->vertex;
	VertexRef v4 = t->next->next->vertex;

	FaceRef f1 = h->face;
	FaceRef f2 = t->face;

	HalfedgeRef h_next = h->next;
	HalfedgeRef t_next = t->next;

	HalfedgeRef h_prev = h->next;
	while (h_prev->next != h) h_prev = h_prev->next;
	HalfedgeRef t_prev = t->next;
	while (t_prev->next != t) t_prev = t_prev->next;

	if (t_prev->vertex == v3 || h_prev->vertex == v4) return std::nullopt;

	// Phase 2: connect
	h_prev->next = t_next;
	t_prev->next = h_next;

	v1->halfedge = h_next;
	v2->halfedge = t_next;

	h->vertex = v4;
	h->next = h_next->next;

	t->vertex = v3;
	t->next = t_next->next;

	h_next->next = t;
	h_next->face = f2;

	t_next->next = h;
	t_next->face = f1;

	f1->halfedge = h;
	f2->halfedge = t;
	
	v3->halfedge = t;
	v4->halfedge = h;

    return e;
}


/*
 * make_boundary: add non-boundary face to boundary
 *  face: the face to make part of the boundary
 *
 * if face ends up adjacent to other boundary faces, merge them into face
 *
 * if resulting mesh would be invalid, does nothing and returns std::nullopt
 * otherwise returns face
 */
std::optional<Mesh::FaceRef> Mesh::make_boundary(FaceRef face) {
	//A2Lx7: (OPTIONAL) make_boundary

	return std::nullopt; //TODO: actually write this code!
}

/*
 * dissolve_vertex: merge non-boundary faces adjacent to vertex, removing vertex
 *  v: vertex to merge around
 *
 * if merging would result in an invalid mesh, does nothing and returns std::nullopt
 * otherwise returns the merged face
 */
std::optional<Mesh::FaceRef> Mesh::dissolve_vertex(VertexRef v) {
	// A2Lx1 (OPTIONAL): Dissolve Vertex

    return std::nullopt;
}

/*
 * dissolve_edge: merge the two faces on either side of an edge
 *  e: the edge to dissolve
 *
 * merging a boundary and non-boundary face produces a boundary face.
 *
 * if the result of the merge would be an invalid mesh, does nothing and returns std::nullopt
 * otherwise returns the merged face.
 */
std::optional<Mesh::FaceRef> Mesh::dissolve_edge(EdgeRef e) {
	// A2Lx2 (OPTIONAL): dissolve_edge

	//Reminder: use interpolate_data() to merge corner_uv / corner_normal data
	
    return std::nullopt;
}

/* collapse_edge: collapse edge to a vertex at its middle
 *  e: the edge to collapse
 *
 * if collapsing the edge would result in an invalid mesh, does nothing and returns std::nullopt
 * otherwise returns the newly collapsed vertex
 */
std::optional<Mesh::VertexRef> Mesh::collapse_edge(EdgeRef e) {
	//A2L3: Collapse Edge

	//Reminder: use interpolate_data() to merge corner_uv / corner_normal data on halfedges
	// (also works for bone_weights data on vertices!)

	// get all halfedges starting at vertex v, except the one going towards exclude
	auto getHalfedgesFrom = [](VertexRef v, VertexRef exclude){
		std::unordered_set<HalfedgeRef> halfedges;
		HalfedgeRef h = v->halfedge;
		do {
			if (h->twin->vertex != exclude)
			{
				halfedges.insert(h);
			}
			h = h->twin->next;
		
		} while (h != v->halfedge);
		return halfedges;
	};

	// Phase 1: collect
	HalfedgeRef he1 = e->halfedge;
	HalfedgeRef he2 = he1->twin;

	VertexRef v1 = he1->vertex;
	VertexRef v2 = he2->vertex;

	// get all halfedges protruding from vertices
	std::unordered_set<HalfedgeRef> v1_halfedges = getHalfedgesFrom(v1, v2);
	std::unordered_set<HalfedgeRef> v2_halfedges = getHalfedgesFrom(v2, v1);

	// want to check if the operation would leave an hourglass-shaped surface.
	// the number of boundary faces should be at most one.
	int num_boundaries = 0;
	for (auto he : v1_halfedges)
	{
		if (he->face->boundary || he->twin->face->boundary)
		{
			num_boundaries++;
			break;
		}
	}
	for (auto he : v2_halfedges)
	{
		if (he->face->boundary || he->twin->face->boundary)
		{
			num_boundaries++;
			break;
		}
	}
	// for (auto face : edge_faces) { std::cout << face->id << std::endl; if (face->boundary) num_boundaries++; }
	if (num_boundaries > 1 && !e->on_boundary()) return std::nullopt;

	// Phase 2: create
	VertexRef v_new = emplace_vertex();
	interpolate_data({v1, v2}, v_new);
	v_new->position = (v1->position + v2->position) / 2.0f;

	std::cout << std::endl;
	FaceRef df; // face to be deleted

	auto collapse_halfedge = [&](HalfedgeRef h) {
		HalfedgeRef hn = h->next;
		HalfedgeRef hp = h->next;
		while (hp->next != h) hp = hp->next;
		if (h->face->boundary)
		{
			if (hn->next == hp)
			{
				/*
					v1 ---- v2
					 \		/
					  \  b /
					   \  /
					    \/
						c
				*/
				HalfedgeRef hnt = hn->twin;
				HalfedgeRef hntp = hnt->next;
				while (hntp->next != hnt) hntp = hntp->next;

				HalfedgeRef hpt = hp->twin;
				HalfedgeRef hptn = hpt->next;
				HalfedgeRef hptp = hpt->next;
				while (hptp->next != hpt) hptp = hptp->next;
				
				df = hn->face;
				hn->face = hpt->face;
				hn->vertex = v_new;
				v_new->halfedge = hn;
				hpt->face->halfedge = hn;
				interpolate_data({hnt, hp}, hnt);
				interpolate_data({hn, hpt}, hn);
				
				hptp->next = hn;
				hn->next = hptn;

				hnt->vertex->halfedge = hnt;
				erase_edge(hp->edge);
				erase_halfedge(hp);
				erase_halfedge(hpt);
				erase_face(df);
			}
			else
			{
				/*		
						h
					v1 ----- v2
				hp /		   \ hn
				  /				\
				 /				 \
				
				*/
				hp->next = hn;
				hn->vertex = v_new;
				h->face->halfedge = hn;	
				v_new->halfedge = hn;
			}
		}
		else
		{
			if (hn->next == hp)
			{
				/*
					v1 --------- v2
					 \	    h    /
					  \hn     hp/
				hn_twin\       /hp_twin
						\     /
						 \   /
						  \ /
						   c
				*/
				HalfedgeRef hnt = hn->twin;
				HalfedgeRef hntp = hnt->next;
				while (hntp->next != hnt) hntp = hntp->next;

				HalfedgeRef hpt = hp->twin;
				HalfedgeRef hptn = hpt->next;
				HalfedgeRef hptp = hpt->next;
				while (hptp->next != hpt) hptp = hptp->next;
				
				df = hn->face;
				hn->face = hpt->face;
				hn->vertex = v_new;
				v_new->halfedge = hn;
				hpt->face->halfedge = hn;
				
				hptp->next = hn;
				hn->next = hptn;
				interpolate_data({hnt, hp}, hnt);
				interpolate_data({hn, hpt}, hn);
				hnt->vertex->halfedge = hnt;

				erase_edge(hp->edge);
				erase_halfedge(hp);
				erase_halfedge(hpt);
				erase_face(df);
				
			}
			else 
			{
				/*
					v1 ----- v2
					/    h     \
				   /      	    \
				  / hn   	  hp \
				*/
				hp->next = hn;
				hn->vertex = v_new;
				h->face->halfedge = hn;
				v_new->halfedge = hn;
			}
		}
	};

	// Phase 3: connect (+ some delete)
	collapse_halfedge(he1);
	collapse_halfedge(he2);
	for (auto he : v1_halfedges) { he->vertex = v_new; }
	for (auto he : v2_halfedges) { he->vertex = v_new; }

	// Phase 4: delete
	erase_vertex(v1);
	erase_vertex(v2);
	erase_edge(e);
	erase_halfedge(he1);
	erase_halfedge(he2);

	return v_new;
}

/*
 * collapse_face: collapse a face to a single vertex at its center
 *  f: the face to collapse
 *
 * if collapsing the face would result in an invalid mesh, does nothing and returns std::nullopt
 * otherwise returns the newly collapsed vertex
 */
std::optional<Mesh::VertexRef> Mesh::collapse_face(FaceRef f) {
	//A2Lx3 (OPTIONAL): Collapse Face

	//Reminder: use interpolate_data() to merge corner_uv / corner_normal data on halfedges
	// (also works for bone_weights data on vertices!)

    return std::nullopt;
}

/*
 * weld_edges: glue two boundary edges together to make one non-boundary edge
 *  e, e2: the edges to weld
 *
 * if welding the edges would result in an invalid mesh, does nothing and returns std::nullopt
 * otherwise returns e, updated to represent the newly-welded edge
 */
std::optional<Mesh::EdgeRef> Mesh::weld_edges(EdgeRef e, EdgeRef e2) {
	//A2Lx8: Weld Edges

	//Reminder: use interpolate_data() to merge bone_weights data on vertices!

    return std::nullopt;
}



/*
 * bevel_positions: compute new positions for the vertices of a beveled vertex/edge
 *  face: the face that was created by the bevel operation
 *  start_positions: the starting positions of the vertices
 *     start_positions[i] is the starting position of face->halfedge(->next)^i
 *  direction: direction to bevel in (unit vector)
 *  distance: how far to bevel
 *
 * push each vertex from its starting position along its outgoing edge until it has
 *  moved distance `distance` in direction `direction`. If it runs out of edge to
 *  move along, you may choose to extrapolate, clamp the distance, or do something
 *  else reasonable.
 *
 * only changes vertex positions (no connectivity changes!)
 *
 * This is called repeatedly as the user interacts, just after bevel_vertex or bevel_edge.
 * (So you can assume the local topology is set up however your bevel_* functions do it.)
 *
 * see also [BEVEL NOTE] above.
 */
void Mesh::bevel_positions(FaceRef face, std::vector<Vec3> const &start_positions, Vec3 direction, float distance) {
	//A2Lx5h / A2Lx6h (OPTIONAL): Bevel Positions Helper
	
	// The basic strategy here is to loop over the list of outgoing halfedges,
	// and use the preceding and next vertex position from the original mesh
	// (in the start_positions array) to compute an new vertex position.
	
}

/*
 * extrude_positions: compute new positions for the vertices of an extruded face
 *  face: the face that was created by the extrude operation
 *  move: how much to translate the face
 *  shrink: amount to linearly interpolate vertices in the face toward the face's centroid
 *    shrink of zero leaves the face where it is
 *    positive shrink makes the face smaller (at shrink of 1, face is a point)
 *    negative shrink makes the face larger
 *
 * only changes vertex positions (no connectivity changes!)
 *
 * This is called repeatedly as the user interacts, just after extrude_face.
 * (So you can assume the local topology is set up however your extrude_face function does it.)
 *
 * Using extrude face in the GUI will assume a shrink of 0 to only extrude the selected face
 * Using bevel face in the GUI will allow you to shrink and increase the size of the selected face
 * 
 * see also [BEVEL NOTE] above.
 */
void Mesh::extrude_positions(FaceRef face, Vec3 move, float shrink) {
	//A2L4h: Extrude Positions Helper

	//General strategy:
	// use mesh navigation to get starting positions from the surrounding faces,
	// compute the centroid from these positions + use to shrink,
	// offset by move
	std::unordered_set<VertexRef> vertices;
	HalfedgeRef h = face->halfedge;
	Vec3 centroid;
	do {
		vertices.emplace(h->vertex);
		centroid += h->vertex->position;
		h = h->next;
	} while(h != face->halfedge);

	centroid /= vertices.size();
	for (VertexRef v : vertices) {
		v->position = (1 - shrink) * v->position + shrink * centroid + move;
	}
}

