#pragma once
template<class T>
using mat3d = std::vector< std::vector< std::vector<T> > >;

template<class T>
using mat2d = std::vector< std::vector<T> >;

// CGAL neighbor search typedefs
typedef CGAL::Simple_cartesian<double>					KCart;
typedef KCart::Point_3									Point_d;
typedef CGAL::Search_traits_3<KCart>					TreeTraits;
typedef CGAL::Orthogonal_k_neighbor_search<TreeTraits>	Neighbor_search;
typedef Neighbor_search::Tree							Tree;

// typedefs for defining the adaptor
typedef CGAL::Exact_predicates_inexact_constructions_kernel                  K;
typedef CGAL::Delaunay_triangulation_2<K>                                    DT;
typedef CGAL::Delaunay_triangulation_adaptation_traits_2<DT>                 AT;
typedef CGAL::Delaunay_triangulation_caching_degeneracy_removal_policy_2<DT> AP;
typedef CGAL::Voronoi_diagram_2<DT, AT, AP>                                  VD;

// typedef for the result type of the point location
typedef AT::Site_2                    Site_2;
typedef AT::Point_2                   Point_2;
typedef VD::Locate_result             Locate_result;
typedef VD::Vertex_handle             Vertex_handle;
typedef VD::Face_handle               Face_handle;
typedef VD::Halfedge_handle           Halfedge_handle;
typedef VD::Ccb_halfedge_circulator   Ccb_halfedge_circulator;

//CGAL Polygon
typedef CGAL::Polygon_2<K> Polygon_2;

// CGAL 3D Voronoi
typedef CGAL::Exact_predicates_inexact_constructions_kernel     Kernel;
typedef CGAL::Delaunay_triangulation_3<K, CGAL::Fast_location>  Triangulation;
typedef CGAL::Periodic_3_triangulation_traits_3<Kernel>         PK;
typedef CGAL::Periodic_3_Delaunay_triangulation_3<PK>           P3DT3;
typedef Triangulation::Point                                    Point;
typedef P3DT3::Locate_type										Locate_type;
typedef P3DT3::Cell_handle										Cell_handle;
