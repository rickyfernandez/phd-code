#include "tess.h"
#include <vector>
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
        //double *x,
        //double *y,
        //double *z,
        double *radius,
        int num_particles,
        double huge) {
    
    local_num_particles = num_particles;

    // gernerating vertices for the tesselation 
    std::vector<Point> particles;
    for (int i=0; i<local_num_particles; i++)
        //particles.push_back(Point(x[i], y[i], z[i]));
        particles.push_back(Point(x[0][i], x[1][i], x[2][i]));

    // create tessellation
    ptess = (void*) new Tess;
    pvt_list = (void*) (new std::vector<Vertex_handle>(local_num_particles));

    Tess &tess = *(Tess*) ptess;
    std::vector<Vertex_handle> &vt_list = *(std::vector<Vertex_handle>*) pvt_list;



    std::vector<std::ptrdiff_t> indices;
    indices.reserve(local_num_particles);
    std::copy(
            boost::counting_iterator<std::ptrdiff_t>(0),
            boost::counting_iterator<std::ptrdiff_t>(local_num_particles),
            std::back_inserter(indices));
    CGAL::spatial_sort(indices.begin(), indices.end(), Search_traits_3(&(particles[0])), CGAL::Hilbert_sort_median_policy());



    Vertex_handle vt;
    //for (int i=0; i<local_num_particles; i++) {
    for (std::vector<std::ptrdiff_t>::iterator it=indices.begin(); it!=indices.end(); it++) {
        //vt = tess.insert(particles[i]);
        //vt->info() = i;
        //vt_list[i] = vt;
        vt = tess.insert(particles[*it]);
        vt->info() = *it;
        vt_list[*it] = vt;
    }

    std::vector<Edge> edges;
    std::vector<Cell_handle> cells;
    std::vector<bool> sites_ngb_used(local_num_particles, false);
    std::vector<int>  site_ngb_list;

    for (int i=0; i<local_num_particles; i++) {

        const Vertex_handle& vi = vt_list[i];
        const Point& pos = particles[i];

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

        double radius_max_sq = (edges.empty()) ? huge : 0.0;

        for (std::vector<Edge>::iterator edge_it = edges.begin(); edge_it != edges.end(); edge_it++) {

            // grab vertices of voronoi face associated with edge
            const Cell_circulator cc_end = tess.incident_cells(*edge_it);
            Cell_circulator cc(cc_end);

            do {
                if (tess.is_infinite(cc)) {
                    radius_max_sq = huge;
                } else {
                    const Point c = tess.dual(cc);
                    radius_max_sq = std::max(radius_max_sq,
                            (c.x()-pos.x())*(c.x()-pos.x()) +
                            (c.y()-pos.y())*(c.y()-pos.y()) +
                            (c.z()-pos.z())*(c.z()-pos.z()) );
                }
            } while (++cc != cc_end);
        }
        radius[i] = 2.01*std::sqrt(radius_max_sq);
    }
    return 0;
}

int Tess3d::count_number_of_faces(void) {

    Tess &tess = *(Tess*) ptess;
    std::vector<Vertex_handle> &vt_list = *(std::vector<Vertex_handle>*) pvt_list;
    int num_faces = 0;

    std::vector<Cell_handle> cells;
    std::vector<bool> sites_ngb_used(tot_num_particles, false);
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
    //std::cout << "number faces cpp:" << num_faces << std::endl;
    return num_faces;
}

int Tess3d::update_initial_tess(
        double *x[3],
        //double *x,
        //double *y,
        //double *z,
        int up_num_particles) {

    int start_num = local_num_particles;
    tot_num_particles = local_num_particles + up_num_particles;

    Tess &tess = *(Tess*) ptess;

    // create points for ghost particles 
    std::vector<Point> particles;
    for (int i=start_num; i<tot_num_particles; i++)
        particles.push_back(Point(x[0][i], x[1][i], x[2][i]));
        //particles.push_back(Point(x[i], y[i], z[i]));

    // add ghost particles to the tess
    Vertex_handle vt;
    for (int i=0, j=local_num_particles; i<up_num_particles; i++, j++) {
        vt = tess.insert(particles[i]);
        vt->info() = j;
    }
    return 0;
}

int Tess3d::extract_geometry(
        double* x[3],
        //double* x,
        //double* y,
        //double* z,
        double* dcenter_of_mass[3],
        //double* center_of_mass_x,
        //double* center_of_mass_y,
        //double* center_of_mass_z,
        double* volume,
        double* face_area,
        double* face_com[3],
        //double* face_comx,
        //double* face_comy,
        //double* face_comz,
        double* face_n[3],
        //double* face_nx,
        //double* face_ny,
        //double* face_nz,
        int* pair_i,
        int* pair_j) {

    // face counter
    int fc=0;
    Tess &tess = *(Tess*) ptess;
    std::vector<Vertex_handle> &vt_list = *(std::vector<Vertex_handle>*) pvt_list;

    double tot_volume = 0;
    std::vector<Edge> edges;
    std::vector<Cell_handle> cells;
    std::vector<bool> sites_ngb_used(tot_num_particles, false);
    std::vector<int>  site_ngb_list;

  // only process local particle information
    for (int i=0; i<local_num_particles; i++) {

        const Vertex_handle &vi = vt_list[i];

        const vector3 ipos(x[0][i], x[1][i], x[2][i]);
        //const vector3 ipos(x[i], y[i], z[i]);

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
                    std::cout << "infinite vertex" << std::endl;
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

            //vector3 normal(0.0, 0.0, 0.0);

            vector3 face_centroid(0.0, 0.0, 0.0);
            vector3 v1 = vertex_list[edge].back() - c;
            //const vector3 cv = ipos - c;

            //const double fourth = 1.0/4.0;
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

                // particle volume and center of mass
                //const vector3 c4 = c + (v1 + v2 + cv)*fourth;
                //const double vol4 = std::abs(v1.cross(v2)*cv);
                //cell_volume += vol4;
                //cell_centroid += c4 * vol4;
                v1 = v2;
            }

            //if (i == 0) {
                //std::cout << "particle i= " << i << " area= " << 0.5*area1 << std::endl;
            //}
            
            const Edge e = edges[edge];
            int id1, id2;
            const int i1 = e.get<0>()->vertex(e.get<1>())->info();
            const int i2 = e.get<0>()->vertex(e.get<2>())->info();

            id1 = (i1 == i) ? i1 : i2;
            id2 = (i1 == i) ? i2 : i1;

            //if (i == 2050) {
            //std::cout << "particle i= " << i << " j=" << id2
            //    << " xi=" << x[i] << " yi=" << y[i] << " zi=" << z[i] 
            //    << " xj=" << x[id2] << " yj=" << y[id2] << " zj=" << z[id2] << std::endl;
            //}

            if (id1 != i){
                std::cout << id1 << " != " << i << " not equal!" << std::endl;
                std::cout << "id1 failed" << std::endl;
                std::cout << "i=" << i << " id1=" << id1 << " i1=" << i1 << " id2=" << id2 << " i2=" << i2 << std::endl;
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
                //const vector3 jpos(x[id2], y[id2], z[id2]);
                //vector3 face_n = jpos - ipos;
                //face_n *= 1.0/face_n.abs();
                vector3 normal = jpos - ipos;
                normal *= 1.0/normal.abs();

                face_area[fc] = 0.5*area1;

                // orientation of face
                face_n[0][fc] = normal.x;
                face_n[1][fc] = normal.y;
                face_n[2][fc] = normal.z;
                //face_nx[fc] = face_n.x;
                //face_ny[fc] = face_n.y;
                //face_nz[fc] = face_n.z;

                //std::cout << "particle i= " << i << " j=" << id2 << " area= " << 0.5*area1 
                //    << " x=" << x[i] << " y=" << y[i] << " z=" << z[i] 
                //    << " x=" << x[id2] << " y=" << y[id2] << " z=" << z[id2]  
                //    << " nx=" << face_nx[fc] << " ny=" << face_ny[fc] << " nz=" << face_nz[fc]
                //    << " nx=" << normal.x/normal.abs() << " ny=" << normal.y/normal.abs() << " nz=" << normal.z/normal.abs() << std::endl;

                // center of mass of face
                //face_centroid *= 1.0/area;
                face_com[0][fc] = face_centroid.x;
                face_com[1][fc] = face_centroid.y;
                face_com[2][fc] = face_centroid.z;
                //face_comx[fc] = face_centroid.x;
                //face_comy[fc] = face_centroid.y;
                //face_comz[fc] = face_centroid.z;

                pair_i[fc] = id1;
                pair_j[fc] = id2;

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
        //center_of_mass_x[i] = cell_centroid.x;
        //center_of_mass_y[i] = cell_centroid.y;
        //center_of_mass_z[i] = cell_centroid.z;
    }

    //std::cout << " total volume: " << tot_volume << std::endl;
    return fc;
}
