#include "tess.h"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h> 

#include <boost/iterator/counting_iterator.hpp>

typedef CGAL::Exact_predicates_inexact_constructions_kernel  K;
typedef CGAL::Triangulation_vertex_base_with_info_3<int, K> Vb; 
typedef CGAL::Triangulation_data_structure_3<Vb>           Tds; 
typedef CGAL::Delaunay_triangulation_3<K, Tds>            Tess;
typedef Tess::Vertex_handle   Vertex_handle;
typedef Tess::Point           Point;
typedef Tess::Edge            Edge;
typedef Tess::Cell            Cell;
typedef Tess::Cell_handle     Cell_handle;
typedef Tess::Cell_circulator Cell_circulator;

typedef CGAL::Spatial_sort_traits_adapter_3<K, Point*> Search_traits_3;


Tess3d::Tess3d(void) {
    ptess = NULL;
    pvt_list = NULL;
}

void Tess3d::reset_tess(void) {
    delete (Tess*) ptess;
    delete (std::vector<Vertex_handle>*) pvt_list;
    ptess = NULL;
    pvt_list = NULL;
}

int Tess3d::build_initial_tess(
        double *x[3],
        double *radius,
        int num_real_particles) {
    /*

    Creates initial tessellation of real particles and calculates
    the radius. If the radius is infinite the values is to -1.

    Parameters
    ----------
    x : double[3]*
       Pointer to the position of the particles.

    radius : double*
        Pointer to store the radius of real particles. The radius is
        the circle than encompass all circumcircles from the voronoi
        for the given particle.

    num_real_particles : int
        Starting index of the first ghost particle in the particle
        data container.

    */
    local_num_particles = num_real_particles;
    total_num_particles = num_real_particles;

    // add all particles
    std::vector<Point> particles;
    for (int i=0; i<num_real_particles; i++)
        particles.push_back(Point(x[0][i], x[1][i], x[2][i]));

    // create tessellation
    ptess = (void*) new Tess;
    pvt_list = (void*) (new std::vector<Vertex_handle>(num_real_particles));

    // pointers to tessellation and vertices
    Tess &tess = *(Tess*) ptess;
    std::vector<Vertex_handle> &vt_list = *(std::vector<Vertex_handle>*) pvt_list;

    // FIX: remove boost and use finite_vertices_iterator to loop over vertices
    // and tess.insert(points.begin(), points.end()) for sorting and building
    // the tessellation.

    // sort particles
    std::vector<std::ptrdiff_t> indices;
    indices.reserve(num_real_particles);
    std::copy(
            boost::counting_iterator<std::ptrdiff_t>(0),
            boost::counting_iterator<std::ptrdiff_t>(num_real_particles),
            std::back_inserter(indices));

    // sort particles by hilbert keys
    CGAL::spatial_sort(indices.begin(), indices.end(), Search_traits_3(&(particles[0])),
            CGAL::Hilbert_sort_median_policy());

    // insert sorted particles into the tessellation
    Vertex_handle vt;
    for (std::vector<std::ptrdiff_t>::iterator it=indices.begin(); it!=indices.end(); it++) {
        vt = tess.insert(particles[*it]);
        vt->info() = *it;
        vt_list[*it] = vt;
    }

    // container for delaunay edges
    std::vector<Edge> edges;
    std::vector<Cell_handle> cells;

    // hold neighbors and flag neighbors accounted for
    std::vector<bool> sites_ngb_used(num_real_particles, false);
    std::vector<int>  site_ngb_list;

    // loop over local particles
    for (int i=0; i<num_real_particles; i++) {

        // exctract particle position and vertex
        const Vertex_handle& vi = vt_list[i];
        const Point& pos = particles[i];
        bool infinite_radius = false;

        edges.clear();
        cells.clear();
        site_ngb_list.clear();

        // collect all tetrahedra that are incident on vertex
        tess.incident_cells(vi, std::back_inserter(cells));

        // loop over each tetrahedra
        const int ncells = cells.size();
        for (int icell=0; icell<ncells; icell++) {
            const Cell_handle& ci = cells[icell];

            // loop over vertices of tetrahedra
            // find incident vertex to ignore
            int idx = -1;
            for (int iv=0; iv<4; iv++) {
                if (ci->vertex(iv) == vi)
                    idx = iv;
            }

            int iadd = 0;
            for (int iv=0; iv<4; iv++) {

                // ignore incident
                if (iv == idx)
                    continue;

                // cgal puts infinite vertex in the tessellation
                const Vertex_handle& v = ci->vertex(iv);
                if (tess.is_infinite(v))
                    continue;

                // check if vertex has been processed 
                const int id = v->info();
                if (sites_ngb_used[id])
                    continue;

                iadd++;
                sites_ngb_used[id] = true;
                site_ngb_list.push_back(id);
                edges.push_back(Edge(ci, idx, iv));
            }
        }

        // reset neighbors processed
        const int nngb = site_ngb_list.size();
        for (int j=0; j<nngb; j++)
            sites_ngb_used[site_ngb_list[j]] = false;

        int cnt0 = 0;
        double radius_max_sq = -1.0;

        // should turn to while loop with check on infinite radius
        for (std::vector<Edge>::iterator edge_it = edges.begin();
                edge_it != edges.end(); edge_it++) {

            // grab vertices of voronoi face associated with edge
            const Cell_circulator cc_end = tess.incident_cells(*edge_it);
            Cell_circulator cc(cc_end);

            do {

                cnt0++;
                // check for infinite voronoi vertices
                if (tess.is_infinite(cc)) {
                    infinite_radius = true;

                // calculate distance between particle
                // and voronoi vertex
                } else {
                    const Point c = tess.dual(cc);
                    radius_max_sq = std::max(radius_max_sq,
                            (c.x()-pos.x())*(c.x()-pos.x()) +
                            (c.y()-pos.y())*(c.y()-pos.y()) +
                            (c.z()-pos.z())*(c.z()-pos.z()) );
                }
            } while (++cc != cc_end);
        }
        if(cnt0 == 0)
            infinite_radius = true;

        if (infinite_radius)
            // flag bad particle
            radius[i] = -1;
        else
            radius[i] = 2.0*std::sqrt(radius_max_sq);
    }
    return 0;
}

int Tess3d::update_initial_tess(
        double *x[3],
        int begin_particles,
        int end_particles) {
    /*

    Update the mesh by adding ghost particles. This function
    is called multiple times untill all real particles have
    finite volume.

    Parameters
    ----------
    x : double[3]*
        Pointer to the position of the particles.

    begin_particles : int
        Starting index of ghost particles to add.

    end_particles : int
        Ending index of ghost particles to add.

    */
    if (begin_particles == end_particles)
        return 0;

    std::vector<Vertex_handle> &vt_list = *(std::vector<Vertex_handle>*) pvt_list;

    total_num_particles = end_particles;
    Tess &tess = *(Tess*) ptess;

    // create points for ghost particles 
    std::vector<Point> particles;
    for (int i=begin_particles; i<end_particles; i++)
        particles.push_back(Point(x[0][i], x[1][i], x[2][i]));

    // sort particles
//    std::vector<std::ptrdiff_t> indices;
//    indices.reserve(particles.size());
//    std::copy(
//            boost::counting_iterator<std::ptrdiff_t>(0),
//            boost::counting_iterator<std::ptrdiff_t>(particles.size()),
//            std::back_inserter(indices));
//
//    // sort particles by hilbert keys
//    CGAL::spatial_sort(indices.begin(), indices.end(), Search_traits_3(&(particles[0])),
//            CGAL::Hilbert_sort_median_policy());
//
//    Vertex_handle vt;
//    for (std::vector<std::ptrdiff_t>::iterator it=indices.begin(); it!=indices.end(); it++) {
//        vt = tess.insert(particles[*it]);
//        vt->info() = *it + begin_particles;
//        vt_list.push_back(vt);
//    }
    Vertex_handle vt;
    for (int i=0, j=begin_particles; i<particles.size(); i++, j++) {
        vt = tess.insert(particles[i]);
        vt->info() = j;
        vt_list.push_back(vt);
    }
    return 0;
}

int Tess3d::count_number_of_faces(void) {
    /*

    Count the number of faces that belong to a real
    particle. This is used to allocate storage for
    containers holding face information.

    */
    Tess &tess = *(Tess*) ptess;
    std::vector<Vertex_handle> &vt_list = *(std::vector<Vertex_handle>*) pvt_list;
    int num_faces = 0;

    std::vector<Cell_handle> cells;
    std::vector<bool> sites_ngb_used(total_num_particles, false);
    std::vector<int>  site_ngb_list;

    for (int i=0; i<local_num_particles; i++) {

        const Vertex_handle &vi = vt_list[i];

        cells.clear();
        site_ngb_list.clear();
        tess.incident_cells(vi, std::back_inserter(cells));

        const int ncells = cells.size();
        for (int icell=0; icell<ncells; icell++) {
            const Cell_handle& ci = cells[icell];

            int idx = -1;
            for (int iv=0; iv<4; iv++) {
                if (ci->vertex(iv) == vi)
                    idx = iv;
            }

            int iadd = 0;
            for (int iv=0; iv<4; iv++) {
                if (iv == idx)
                    continue;

                const Vertex_handle& v = ci->vertex(iv);
                if (tess.is_infinite(v))
                    continue;

                const int id = v->info();
                if (sites_ngb_used[id])
                    continue;

                if (i < id)
                    num_faces++;

                iadd++;
                sites_ngb_used[id] = true;
                site_ngb_list.push_back(id);
            }
        }
        const int nngb = site_ngb_list.size();
        for (int j=0; j<nngb; j++)
            sites_ngb_used[site_ngb_list[j]] = false;
    }
    return num_faces;
}


int Tess3d::extract_geometry(
        double* x[3],
        double* dcenter_of_mass[3],
        double* volume,
        double* face_area,
        double* face_com[3],
        double* face_n[3],
        int* pair_i,
        int* pair_j,
        std::vector< std::vector<int> > &neighbors) {
    /*

    Extract all geometric information pertaining to the mesh,
    i.e. area, normal, volume, ...

    Parameters
    ----------
    x : double[3]*
        Pointer to the position of the particles.

    dcenter_of_mass : double[3]*
        Pointer to the center of mass of each real particle
        realtive to its position.

    volume : double*
        Pointer to the volume of each real particle.

    face_area : double*
        Pointer to face area defined by particle i and j .

    face_com : double[3]*
        Pointer to face center of mass defined by particle i and j .

    pair_i : int*
        Pointer to left most particle defining the face.

    pair_j : int*
        Pointer to right most particle defining the face.

    */
    // face counter
    int fc=0;
    Tess &tess = *(Tess*) ptess;
    std::vector<Vertex_handle> &vt_list = *(std::vector<Vertex_handle>*) pvt_list;

    double tot_volume = 0;
    std::vector<Edge> edges;
    std::vector<Cell_handle> cells;
    std::vector<bool> sites_ngb_used(total_num_particles, false);
    std::vector<int>  site_ngb_list;

    // only process local particle information
    for (int i=0; i<local_num_particles; i++) {

        const Vertex_handle &vi = vt_list[i];
        const vector3 ipos(x[0][i], x[1][i], x[2][i]);

        edges.clear();
        cells.clear();
        site_ngb_list.clear();
        tess.incident_cells(vi, std::back_inserter(cells));

        const int ncells = cells.size();
        for (int icell=0; icell<ncells; icell++) {
            const Cell_handle& ci = cells[icell];

            int idx = -1;
            for (int iv=0; iv<4; iv++) {
                if (ci->vertex(iv) == vi)
                    idx = iv;
            }

            int iadd = 0;
            for (int iv=0; iv<4; iv++) {
                if (iv == idx)
                    continue;

                const Vertex_handle& v = ci->vertex(iv);
                if (tess.is_infinite(v))
                    continue;

                const int id = v->info();
                if (sites_ngb_used[id])
                    continue;

                iadd++;
                sites_ngb_used[id] = true;
                site_ngb_list.push_back(id);
                edges.push_back(Edge(ci, idx, iv));
            }
        }

        const int nngb = site_ngb_list.size();
        for (int j=0; j<nngb; j++)
            sites_ngb_used[site_ngb_list[j]] = false;

        const int NMAXEDGE = 1024;
        static std::vector<vector3> vertex_list[NMAXEDGE];

        int nedge = 0;
        for (std::vector<Edge>::iterator edge_it = edges.begin(); edge_it != edges.end(); edge_it++) {
            
            const Cell_circulator cc_end = tess.incident_cells(*edge_it);
            Cell_circulator cc(cc_end);
            vertex_list[nedge].clear();

            do {
                if (tess.is_infinite(cc)) {
                    return -1;
                } else {
                    const Point c = tess.dual(cc);
                    const vector3 centre(c.x(), c.y(), c.z());
                    vertex_list[nedge].push_back(centre);
                }
            } while (++cc != cc_end);
            nedge++;
        }

        double cell_volume = 0.0;
        vector3 cell_centroid(0.0, 0.0, 0.0);
        for (int edge=0; edge<nedge; edge++) {

            vector3 c(0.0, 0.0, 0.0);
            const int nvtx = vertex_list[edge].size();

            for (int j=0; j<nvtx; j++)
                c += vertex_list[edge][j];
            c *= 1.0/nvtx;

            vector3 face_centroid(0.0, 0.0, 0.0);
            vector3 v1 = vertex_list[edge].back() - c;

            const double third  = 1.0/3.0;
            double area1 = 0.0;

            for (int j=0; j<nvtx; j++) {
                
                const vector3 v2 = vertex_list[edge][j] - c;

                // face area and center of mass
                const vector3 norm3 = v1.cross(v2);
                const double area3 = norm3.abs();
                const vector3 c3 = c + (v1 + v2)*third;

                //normal += norm3;
                face_centroid += c3*area3;
                area1 += area3;

                v1 = v2;
            }

            const Edge e = edges[edge];
            int id1, id2;
            const int i1 = e.get<0>()->vertex(e.get<1>())->info();
            const int i2 = e.get<0>()->vertex(e.get<2>())->info();

            id1 = (i1 == i) ? i1 : i2;
            id2 = (i1 == i) ? i2 : i1;

            if (id1 != i){
                return -1;
            }

            const double SMALLDIFF1 = 1.0e-10;
            const double area0 = area1;
            const double L1 = std::sqrt(area0);
            const double L2 = (face_centroid - ipos).abs();
            const double area = (L1 < SMALLDIFF1*L2) ? 0.0 : area0;

            // ignore face
            if (area == 0.0)
                continue;

            face_centroid *= 1.0/area;

            if (id1 < id2) {

                const vector3 jpos(x[0][id2], x[1][id2], x[2][id2]);
                vector3 normal = jpos - ipos;
                normal *= 1.0/normal.abs();

                face_area[fc] = 0.5*area1;

                // orientation of face
                face_n[0][fc] = normal.x;
                face_n[1][fc] = normal.y;
                face_n[2][fc] = normal.z;

                // center of mass of face
                face_com[0][fc] = face_centroid.x;
                face_com[1][fc] = face_centroid.y;
                face_com[2][fc] = face_centroid.z;

                pair_i[fc] = id1;
                pair_j[fc] = id2;

                // store neighbors - face id is stored
                neighbors[id1].push_back(fc);
                neighbors[id2].push_back(fc);

                fc++;
            }

            const double fourth = 1.0/4.0;
            v1 = vertex_list[edge].back() - face_centroid;
            const vector3 cv = ipos - face_centroid;
            for (int j=0; j<nvtx; j++) {
                
                // particle volume and center of mass
                const vector3 v2 = vertex_list[edge][j] - face_centroid;
                const vector3 c4 = face_centroid + (v1 + v2 + cv)*fourth;
                const double vol4 = std::abs(v1.cross(v2)*cv);
                cell_volume += vol4;
                cell_centroid += c4 * vol4;
                v1 = v2;
            }
        }
        cell_centroid *= 1.0/cell_volume;
        cell_volume *= 1.0/6.0;

        tot_volume += cell_volume;

        volume[i] = cell_volume;
        dcenter_of_mass[0][i] = cell_centroid.x - ipos.x;
        dcenter_of_mass[1][i] = cell_centroid.y - ipos.y;
        dcenter_of_mass[2][i] = cell_centroid.z - ipos.z;
    }
    return fc;
}


int Tess3d::update_radius(
        double *x[3],
        double *radius,
        std::list<FlagParticle> &flagged_particles) {
    /*

    For particle in flagged_particle container, upater their
    radius.

    Parameters
    ----------
    x : double[3]*
       Pointer to the position of the particles.

    radius : double*
        Pointer to store the radius of real particles. The radius is
        the circle than encompass all circumcircles from the voronoi
        for the given particle.

    flagged_particles : list<FlagParticle>
        List of particles that have been flagged by the domain mananger
        to create ghost particles.

    */
    Tess &tess = *(Tess*) ptess;
    std::vector<Vertex_handle> &vt_list = *(std::vector<Vertex_handle>*) pvt_list;

    // edges connecting particles
    std::vector<Edge> edges;
    std::vector<Cell_handle> cells;

    // store neighbors not allow repeats
    std::vector<int>  site_ngb_list;
    std::vector<bool> sites_ngb_used(total_num_particles, false);

    // loop over all flagged particles
    for(std::list<FlagParticle>::iterator it = flagged_particles.begin();
            it != flagged_particles.end(); ++it) {

        // exctract particle position and vertex
        int i = it->index;
        const Vertex_handle& vi = vt_list[i];
        double xp = x[0][i], yp = x[1][i], zp = x[2][i];
        bool infinite_radius = false;
        double radius_max_sq = 1.0;

        edges.clear();
        cells.clear();
        site_ngb_list.clear();

        // collect all tetrahedra that are incident
        tess.incident_cells(vi, std::back_inserter(cells));

        const int ncells = cells.size();
        for (int icell=0; icell<ncells; icell++) {
            const Cell_handle& ci = cells[icell];

            // find incident vertex
            int idx = -1;
            for (int iv=0; iv<4; iv++) {
                if (ci->vertex(iv) == vi)
                    idx = iv;
            }

            int iadd = 0;
            for (int iv=0; iv<4; iv++) {

                // ignore incident
                if (iv == idx)
                    continue;

                // cgal puts infinite vertex in the tessellation
                const Vertex_handle& v = ci->vertex(iv);
                if (tess.is_infinite(v))
                    continue;

                // check if vertex has been processed 
                const int id = v->info();
                if (sites_ngb_used[id])
                    continue;

                iadd++;
                sites_ngb_used[id] = true;
                site_ngb_list.push_back(id);
                edges.push_back(Edge(ci, idx, iv));
            }
        }

        // reset neighbors processed
        const int nngb = site_ngb_list.size();
        for (int j=0; j<nngb; j++)
            sites_ngb_used[site_ngb_list[j]] = false;

        if (edges.empty()) {
            infinite_radius = true;
            continue;
        }

        // should turn to while loop with check on infinite radius
        for (std::vector<Edge>::iterator edge_it = edges.begin(); edge_it != edges.end(); edge_it++) {

            // grab vertices of voronoi face associated with edge
            const Cell_circulator cc_end = tess.incident_cells(*edge_it);
            Cell_circulator cc(cc_end);

            // check for infinite voronoi vertices
            do {
                if (tess.is_infinite(cc)) {
                    infinite_radius = true;

                } else {
                    // calculate distance between particle
                    // and voronoi vertex
                    const Point c = tess.dual(cc);
                    radius_max_sq = std::max(radius_max_sq,
                            (c.x() - xp)*(c.x() - xp) +
                            (c.y() - yp)*(c.y() - yp) +
                            (c.z() - zp)*(c.z() - zp));
                }
            } while (++cc != cc_end);
        }

        if (infinite_radius)
            // flag bad particle
            radius[i] = -1;
        else
            radius[i] = 2.0*std::sqrt(radius_max_sq);
    }
    return 0;
}

int Tess3d::reindex_ghost(std::vector<GhostID> &import_ghost_buffer) {
    /*

    Reorder ghost particles because after the mesh is complete in parallel
    ghost particles are put in processor and export order.

    Parameters
    ----------
    import_ghost_buffer : std:vector<GhostID>
        Vector holding old indices and sorted indices of ghost particles

    */
    // create tessellation
    std::vector<Vertex_handle> &vt_list = *(std::vector<Vertex_handle>*) pvt_list;

    Vertex_handle vt;
    for (int i=local_num_particles, j=0; j<import_ghost_buffer.size(); i++, j++) {
        int index = import_ghost_buffer[j].index;
        const Vertex_handle &vt = vt_list[index];
        vt->info() = i; 
    }
    return 0;
}
