//
// Created by agalex on 2/22/23.
//

#ifndef STANFORDICP_CGALHEADERS_H
#define STANFORDICP_CGALHEADERS_H

// CGAL HEADERS///////////////////////////////////
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Fuzzy_sphere.h>
#include <CGAL/Fuzzy_iso_box.h>
#include <boost/iterator/zip_iterator.hpp>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/mst_orient_normals.h>
#include <CGAL/property_map.h>

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = Kernel::Point_3;
using Vector_3 = Kernel::Vector_3;
using Point_and_int = boost::tuple<Point_3, int>;
using Traits_base = CGAL::Search_traits_3<Kernel>;
using Traits = CGAL::Search_traits_adapter<Point_and_int,
            CGAL::Nth_of_tuple_property_map<0, Point_and_int>,
            Traits_base>;
using CGALTree = CGAL::Kd_tree<Traits>;
using Fuzzy_sphere = CGAL::Fuzzy_sphere<Traits>;
using Fuzzy_iso_box = CGAL::Fuzzy_iso_box<Traits>;
using K_neighbor_search = CGAL::Orthogonal_k_neighbor_search<Traits>;
using Distance = K_neighbor_search::Distance;
using Concurrency_tag = CGAL::Parallel_if_available_tag;
// Point with normal vector stored in a std::pair.
using CGALPointVectorPair = std::pair<Point_3, Vector_3>;
using CGALPointList = std::vector<CGALPointVectorPair>;

#endif //STANFORDICP_CGALHEADERS_H
