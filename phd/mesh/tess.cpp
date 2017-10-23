#include "tess.h"
//#include <vector>
#include <iostream>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
//#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h> 
#include <CGAL/number_utils.h>

#include <CGAL/Memory_sizer.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K; 
//typedef CGAL::Exact_predicates_exact_constructions_kernel K; 
typedef CGAL::Triangulation_vertex_base_with_info_2<int, K> Vb; 
typedef CGAL::Triangulation_data_structure_2<Vb>            Tds; 
typedef CGAL::Delaunay_triangulation_2<K, Tds>              Tess;
typedef CGAL::Object          Object;
typedef Tess::Vertex_handle   Vertex_handle;
typedef Tess::Point           Point;
typedef Tess::Edge            Edge;
typedef Tess::Edge_circulator Edge_circulator;

Tess2d::Tess2d(void) {
    ptess = NULL;
    pvt_list = NULL;
}

void Tess2d::reset_tess(void) {
    delete (Tess*) ptess;
    delete (std::vector<Vertex_handle>*) pvt_list;
    ptess = NULL;
    pvt_list = NULL;
}

int Tess2d::build_initial_tess(
        double *x[3],
        double *radius,
        int start_new_ghost,
        int stop_new_ghost) {
    
    // save local number of particles
    local_num_particles = start_new_ghost;

    // gernerating vertices for the tesselation 
    std::vector<Point> particles;
    for (int i=0; i<stop_new_ghost; i++)
        particles.push_back(Point(x[0][i], x[1][i]));

    // create tessellation
    ptess = (void*) new Tess;
    pvt_list = (void*) (new std::vector<Vertex_handle>(stop_new_ghost));

    Tess &tess = *(Tess*) ptess;
    std::vector<Vertex_handle> &vt_list = *(std::vector<Vertex_handle>*) pvt_list;

    Vertex_handle vt;
    for (int i=0; i<stop_new_ghost; i++) {
        vt = tess.insert(particles[i]);
        vt->info() = i;
        vt_list[i] = vt;
    }

    // only real particles
    for (int i=0; i<start_new_ghost; i++) {

        const Vertex_handle &vi = vt_list[i];
        const Point& pos = particles[i];
        bool infinite_radius = false;
        double radius_max_sq = 0.0;

        // find all edges that are incident with particle vertex
        Edge_circulator ed = tess.incident_edges(vi), done(ed);

        // process each edge
        do {
            // skip edge that contains infinite vertex
            if (tess.is_infinite(ed)) 
                continue;

            const Edge e = *ed;
            if (e.first->vertex( (e.second+2)%3 )->info() != i)
                return -1;

            // extract voronoi face from edge
            CGAL::Object o = tess.dual(*ed);

            // finite faces
            if (const K::Segment_2 *sg = CGAL::object_cast<K::Segment_2>(&o)) {

                // loop over face vertices
                for (int j=0; j<2; j++) {
                    const Point& pj = sg->point(j);
                
                    // calculate max radius from particle
                    radius_max_sq = std::max( radius_max_sq,
                            (pj.x() - pos.x())*(pj.x() - pos.x()) + 
                            (pj.y() - pos.y())*(pj.y() - pos.y()) );
                    //radius_max_sq = std::max( radius_max_sq,
                    //        (CGAL::to_double(pj.x()) - CGAL::to_double(pos.x()))
                    //       *(CGAL::to_double(pj.x()) - CGAL::to_double(pos.x())) + 
                    //        (CGAL::to_double(pj.y()) - CGAL::to_double(pos.y()))
                    //       *(CGAL::to_double(pj.y()) - CGAL::to_double(pos.y())) );
                }
            } else {
                // voronoi not complete
                infinite_radius = true;
                break;
            }

            // infinite face case is considered because a particle can
            // have all faces that are rays
            // else if (CGAL::object_cast<K::Ray_2>(&o)) 
            //
            //   radius_max_sq = huge;
            //

        } while (++ed != done);

        if (infinite_radius)
            radius[i] = -1;
        else
            radius[i] = 2.0*std::sqrt(radius_max_sq);
    }
    //std::size_t memory = CGAL::Memory_sizer().resident_size();
    //std::cout << "Tessellation size: " << (memory >> 20) <<  " Mib" <<std::endl;
    return 0;
}

int Tess2d::update_initial_tess(
        double *x[3],
        int begin_particles,
        int end_particles) {

    if (begin_particles == end_particles)
        return 0;

    Tess &tess = *(Tess*) ptess;

    // create points for ghost particles 
    std::vector<Point> particles;
    for (int i=begin_particles; i<end_particles; i++)
        particles.push_back(Point(x[0][i], x[1][i]));

    // add ghost particles to the tessellation
    Vertex_handle vt;
    for (int i=0, j=begin_particles; i<end_particles; i++, j++) {
        vt = tess.insert(particles[i]);
        vt->info() = j;
    }
    return 0;
}

int Tess2d::count_number_of_faces(void) {

    Tess &tess = *(Tess*) ptess;
    std::vector<Vertex_handle> &vt_list = *(std::vector<Vertex_handle>*) pvt_list;
    int num_faces = 0;

    for (int i=0; i<local_num_particles; i++) {

        const Vertex_handle &vi = vt_list[i];
    
        // find all edges that are incident with particle vertex
        Edge_circulator ed = tess.incident_edges(vi), done(ed);

        // process each edge
        do {
            // edge is infinite 
            if (tess.is_infinite(ed)) 
                return -1;

            // extract voronoi face from edge
            CGAL::Object o = tess.dual(*ed);

            // only consider finite faces
            if (CGAL::object_cast<K::Segment_2>(&o)) {

                const Edge e = *ed;
                int id1 = e.first->vertex( (e.second+2)%3 )->info();
                int id2 = e.first->vertex( (e.second+1)%3 )->info();

                if (id1 != i)
                    return -1;
                if (id1 < id2)
                    num_faces++;

            } else {
                // face is a ray, tess is not complete
                return -1;
            }
        } while (++ed != done);
    }
    return num_faces;
}

int Tess2d::extract_geometry(
        double* x[3],
        double* dcom[3],
        double* volume,
        double* face_area,
        double* face_com[3],
        double* face_n[3],
        int* pair_i,
        int* pair_j,
        std::vector< std::vector<int> > &neighbors) {

    // face counter
    int fc=0;
    Tess &tess = *(Tess*) ptess;
    std::vector<Vertex_handle> &vt_list = *(std::vector<Vertex_handle>*) pvt_list;


    // only process local particle information
    for (int i=0; i<local_num_particles; i++) {

        const Vertex_handle &vi = vt_list[i];
        double xp = x[0][i], yp = x[1][i];
        double cx = 0.0, cy = 0.0, vol = 0.0;

        // find all edges that are incident with particle vertex
        Edge_circulator ed = tess.incident_edges(vi), done(ed);

        /* process each edge, find voronoi face and neighbor
         that make the edge with current particle. If face
         area is not zero store face area, normal, and centroid
        */
        do {
            // edge that contains infinite vertex
            if (tess.is_infinite(ed)) {
                std::cout << "infinite" << "x: " << xp << " y: " << yp << std::endl;
                return -1;
            }

            // extract voronoi face from edge
            CGAL::Object o = tess.dual(*ed);

            // only consider finite faces
            if (const K::Segment_2 *sg = CGAL::object_cast<K::Segment_2>(&o)) {

                const Edge e = *ed;

                const int id1 = e.first->vertex( (e.second+2)%3 )->info();
                const int id2 = e.first->vertex( (e.second+1)%3 )->info();

                const Point& p1 = sg->point(0);
                const Point& p2 = sg->point(1);

                double xn = x[0][id2], x1 = p1.x(), x2 = p2.x();
                double yn = x[1][id2], y1 = p1.y(), y2 = p2.y();
                //double xn = x[0][id2], x1 = CGAL::to_double(p1.x()), x2 = CGAL::to_double(p2.x());
                //double yn = x[1][id2], y1 = CGAL::to_double(p1.y()), y2 = CGAL::to_double(p2.y());

                // difference vector between particles
                double xr = xn - xp;
                double yr = yn - yp;

                // distance between particles 
                double h = std::sqrt(xr*xr + yr*yr);

                // edge vector
                double xe = x2 - x1;
                double ye = y2 - y1;

                // face area in 2d is length between voronoi vertices
                //double area = std::sqrt(xe*xe + ye*ye);
                double area0 = std::sqrt(xe*xe + ye*ye);

                // center of mass of face
                double fx = 0.5*(x1 + x2);
                double fy = 0.5*(y1 + y2);

                // need to work on
                const double SMALLDIFF = 1.0e-10;
                const double L1 = std::sqrt(area0);
                const double L2 = std::sqrt( (fx-xp)*(fx-xp) + (fy-yp)*(fy-yp) );
                const double area = (L1 < SMALLDIFF*L2) ? 0.0 : area0;

                // ignore face
                if (area == 0.0)
                    continue;

                // the volume of the cell is the sum of triangle areas - eq. 27
                vol += 0.25*area*h;

                // center of mass of triangle - eq. 31
                double tx = 2.0*fx/3.0 + xp/3.0;
                double ty = 2.0*fy/3.0 + yp/3.0;

                // center of mass of the celll is the sum weighted center of mass of
                // the triangles - eq. 29
                cx += 0.25*area*h*tx;
                cy += 0.25*area*h*ty;

                // faces are defined by real partilces
                if (id1 < id2) {

                    face_area[fc] = area;

                    // orientation of the face
                    face_n[0][fc] = xr/h;
                    face_n[1][fc] = yr/h;

                    // center of mass of face
                    face_com[0][fc] = fx;
                    face_com[1][fc] = fy;

                    pair_i[fc] = id1;
                    pair_j[fc] = id2;

                    // store neighbors - face id is stored
                    neighbors[id1].push_back(fc);
                    neighbors[id2].push_back(fc);

                    fc++;
                }

            } else {
                std::cout << "infinite face " << "x: " << xp << " y: " << yp << std::endl;
                return -1;
            }
        } while (++ed != done);

        // store volume and delta com
        volume[i] = vol;          
        dcom[0][i] = cx/vol - xp;
        dcom[1][i] = cy/vol - yp;

    }

    //std::size_t memory = CGAL::Memory_sizer().resident_size();
    //std::cout << "Tessellation size: " << (memory >> 20) <<  " Mib" <<std::endl;
    return fc;
}

int Tess2d::update_radius(
        double* x[3],
        double *radius,
        std::list<FlagParticle> flagged_particles) {

    Tess &tess = *(Tess*) ptess;
    std::vector<Vertex_handle> &vt_list = *(std::vector<Vertex_handle>*) pvt_list;

    for(std::list<FlagParticle>::iterator it = flagged_particles.begin();
            it != flagged_particles.end(); ++it) {

        // retrieve particle
        int i = it->index;
        const Vertex_handle &vi = vt_list[i];
        double xp = x[0][i], yp = x[1][i];
        bool infinite_radius = false;
        double radius_max_sq = 0.0;

        // find all edges that are incident with particle vertex
        Edge_circulator ed = tess.incident_edges(vi), done(ed);

        // process each edge
        do {
            // skip edge that contains infinite vertex
            if (tess.is_infinite(ed)) 
                continue;

            const Edge e = *ed;
            if (e.first->vertex( (e.second+2)%3 )->info() != i)
                return -1; // sanity check

            // extract voronoi face from edge
            CGAL::Object o = tess.dual(*ed);

            // finite faces
            if (const K::Segment_2 *sg = CGAL::object_cast<K::Segment_2>(&o)) {

                // loop over face vertices
                for (int j=0; j<2; j++) {
                    const Point& pj = sg->point(j);
                
                    // calculate max radius from particle
                    radius_max_sq = std::max(radius_max_sq,
                            (pj.x() - xp)*(pj.x() - xp) + 
                            (pj.y() - yp)*(pj.y() - yp));
                }
            } else {
                // voronoi not complete
                //radius_max_sq = -1;
                infinite_radius = true;
                break;
            }
        } while (++ed != done);

        if (infinite_radius)
            radius[i] = -1 ;
        else
            radius[i] = 2.0*std::sqrt(radius_max_sq);
    }
    return 0;
}
