// stdafx.h: Includedatei für Standardsystem-Includedateien
// oder häufig verwendete projektspezifische Includedateien,
// die nur in unregelmäßigen Abständen geändert werden.
//

#pragma once

#define _CRT_SECURE_NO_WARNINGS
#define BOOST_CONFIG_SUPPRESS_OUTDATED_MESSAGE
#include <boost/math/special_functions/bessel.hpp>

#include "targetver.h"
#include <random>
#include <algorithm>
#include <iterator>
#include <stdio.h>
#include <tchar.h>
#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <stdlib.h>
#include <cstdlib>
#include <functional>
#include <chrono>
#include <omp.h>
#include <array>
#include <cstdio>
#include <typeinfo>
#include <type_traits>
#include <iomanip>
#include <new>
#include <cfloat>
#include <windows.h>
#include "C:/dev/FFTW_LIBS/fftw3.h"
#include <random>
#include <thread>
#include <future>

// CGAL neighbor search
#include <boost/iterator/transform_iterator.hpp>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Search_traits_3.h>

// includes for defining the Voronoi diagram adaptor
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Voronoi_diagram_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_traits_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_policies_2.h>

// Polygon
#include <CGAL/Polygon_2.h>

// CGAL 3D Voronoi
//#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
//#include <CGAL/Periodic_3_Delaunay_triangulation_traits_3.h>
//#include <CGAL/Periodic_3_Delaunay_triangulation_3.h>
//#include <CGAL/periodic_3_triangulation_3_io.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h> 
#include <CGAL/Delaunay_triangulation_3.h> 
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h> 
#include <CGAL/Periodic_3_triangulation_traits_3.h> 
#include <CGAL/Periodic_3_Delaunay_triangulation_3.h> 

#include <CGAL/Delaunay_triangulation_cell_base_3.h>
#include <CGAL/Triangulation_vertex_base_3.h>