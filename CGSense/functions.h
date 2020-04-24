#pragma once

double mean(const std::vector<float>& t)
{
	double m = 0;
	for (int i = 0; i < t.size(); i++)
	{
		m += t[i];
	}
	return m / double(t.size());
}

// source: http://cgal-discuss.949826.n4.nabble.com/Volume-of-a-periodic-voronoi-cell-td3242042.html, modified
template <class Triangulation>
typename Triangulation::Facet get_first_vertex(
	Triangulation& T,
	const typename Triangulation::Facet& f,
	const typename Triangulation::Edge& e)
{
	int r = 0;
	int c = f.first->index(e.first->vertex(e.second));
	int l = f.first->index(e.first->vertex(e.third));
	while (r == c || r == l || r == f.second)
		++r;
	//if orientation(l,c,r) seen from f.second 
	//is counterclockwise then return f else return f.mirror_facet() 
	//if f.second =         0    2     1     3 
	//then we              213  103   302   201 
	//observe              321  310   023   012 
	//if ccw               132  031   230   120 
	//MAYBE SHOULD USE FUNCTION ccw in Triangulation_utils_3 
	int a[3] = { l,c,r };
	int im = 0;
	if (f.second % 2 == 0) {
		while (a[im] != 3) ++im;
		if (a[(im + 2) % 3] < a[(im + 1) % 3])
			return f;
	}
	else {
		while (a[im] != 0) ++im;
		if (a[(im + 2) % 3] > a[(im + 1) % 3])
			return f;
	}
	return T.mirror_facet(f);
}

// source: http://cgal-discuss.949826.n4.nabble.com/Volume-of-a-periodic-voronoi-cell-td3242042.html, modified
template <class Triangulation>
bool is_infinite_vertex(const Triangulation& T, const typename Triangulation::Vertex_handle& vh)
{
	return vh == T.infinite_vertex();
}
// source: http://cgal-discuss.949826.n4.nabble.com/Volume-of-a-periodic-voronoi-cell-td3242042.html, modified
template <class PT, class TDS, class Vertex_handle>
bool is_infinite_vertex(const CGAL::Periodic_3_Delaunay_triangulation_3<PT, TDS>&, Vertex_handle)
{
	return false;
}

// source: http://cgal-discuss.949826.n4.nabble.com/Volume-of-a-periodic-voronoi-cell-td3242042.html, modified
template <class Triangulation>
typename Triangulation::Point point(const Triangulation&, const typename Triangulation::Vertex_handle& vh)
{
	return vh->point();
}

// source: http://cgal-discuss.949826.n4.nabble.com/Volume-of-a-periodic-voronoi-cell-td3242042.html, modified
template <class PT, class TDS, class Vertex_handle>
typename CGAL::Periodic_3_Delaunay_triangulation_3<PT, TDS>::Points point(const CGAL::Periodic_3_Delaunay_triangulation_3<PT, TDS>& T, Vertex_handle vh)
{
	return T.point(T.periodic_point(vh));
}

// source: http://cgal-discuss.949826.n4.nabble.com/Volume-of-a-periodic-voronoi-cell-td3242042.html, modified
template <class Kernel, class Triangulation, class Circumcenter>
typename Kernel::FT get_area_of_dual_delaunay_edge(const Triangulation& T,
	const typename Triangulation::Triangulation_data_structure::Edge& edge,
	const typename Kernel::Point_3& p1,
	const typename Kernel::Point_3& p2,
	const Circumcenter& circumcenter)
{
	typename Kernel::FT area(0);

	//vertices of the edge 
	typename Triangulation::Vertex_handle vh1 = edge.first->vertex(edge.second);
	typename Triangulation::Vertex_handle vh2 = edge.first->vertex(edge.third);
	typename Triangulation::Facet_circulator f_current = T.incident_facets(edge);
	typename Triangulation::Facet_circulator f_start = f_current;

	std::vector<typename Kernel::Point_3> boundary;

	do {
		typename Triangulation::Facet oriented_facet = get_first_vertex(T, *f_current, edge);
		typename Triangulation::Vertex_handle vh3 = oriented_facet.first->vertex(oriented_facet.second);
		//test if the dual facet is infinite. 
		if (is_infinite_vertex(T, vh3))
			return 0;
		typename Triangulation::Vertex_handle vh4;
		for (int i = 1; i<4; ++i) {
			vh4 = oriented_facet.first->vertex((oriented_facet.second + i) % 4);
			if (vh4 != vh1 && vh4 != vh2) break;
		}

		//test if the dual facet is infinite. 
		if (is_infinite_vertex(T, vh4))
			return 0;

		//assert(vh1 != vh2 && vh1 != vh3 && vh1 != vh4 && vh2 != vh3 && vh2 != vh4 && vh3 != vh4);

		typename Kernel::Point_3 p3 = point(T, vh3);
		typename Kernel::Point_3 p4 = point(T, vh4);

		boundary.push_back(circumcenter(p1, p2, p3, p4));
		--f_current;
	} while (f_start != f_current);

	int n = boundary.size();
	//compute area using a triangulation of convex polygon 
	for (int i = 1; i<n - 1; ++i) {
		typename Kernel::Vector_3 vect1 = boundary[i] - boundary[0];
		typename Kernel::Vector_3 vect2 = boundary[i + 1] - boundary[0];
		area += sqrt(CGAL::cross_product(vect1, vect2).squared_length()) / typename Kernel::FT(2);
	}
	return area;
}

// source: http://cgal-discuss.949826.n4.nabble.com/Volume-of-a-periodic-voronoi-cell-td3242042.html, modified
template <class Triangulation>
double volume_of_finite_dual_cell(const Triangulation& T, typename const Triangulation::Vertex_handle& v)
{
	typedef typename Triangulation::Triangulation_data_structure TDS;
	typedef typename Triangulation::Geom_traits                  Kernel;

	typename Kernel::Construct_circumcenter_3 circumcenter = Kernel().construct_circumcenter_3_object();

	double volume_t = 0;
	typename std::list<typename Triangulation::Edge> edgelist;
	T.incident_edges(v, std::back_inserter(edgelist));

	for (typename std::list<typename Triangulation::Edge>::iterator
		e_current = edgelist.begin();
		e_current != edgelist.end();
		++e_current)
	{
		//vertices of the edge 
		typename Triangulation::Vertex_handle vh1 = e_current->first->vertex(e_current->second);
		typename Triangulation::Vertex_handle vh2 = e_current->first->vertex(e_current->third);
		//if the vertex is infinite or on the convex hull 
		if (is_infinite_vertex(T, vh1) || is_infinite_vertex(T, vh2))
			return 0;
		typename Kernel::Point_3  p1 = point(T, vh1);
		typename Kernel::Point_3  p2 = point(T, vh2);
		//double area_t = get_area_of_dual_delaunay_edge<Kernel>(T, *e_current, p1, p2, circumcenter);
		//double height = CGAL::sqrt(squared_distance(p1, p2)) / double(2);
		//volume_t += area_t * height / double(3);
		volume_t += get_area_of_dual_delaunay_edge<Kernel>(T, *e_current, p1, p2, circumcenter) * CGAL::sqrt(squared_distance(p1, p2));
	}
	return volume_t / double(6);
}

std::vector<float> estimateDensityV3D(const mat2d<double>& traj, const int& sX, const int& sY, const int& sZ, const bool& average, const double& cutoff)
{
	const int nSamples = sX * sY * sZ;
	if (traj.size() != nSamples)
	{
		std::cout << "\nError in function estimateDensity: trajectory size does not match number of samples!\n";
		exit(EXIT_FAILURE);
	}

#pragma omp critical
	{
		std::cout << "Estimating k-space density... \n";
	}
	const int n = traj.size();

	std::vector<Point> trajPoints;
	for (int i = 0; i < traj.size(); i++)
	{
		trajPoints.push_back(Point(traj[i][0], traj[i][1], traj[i][2]));
	}
	auto start = std::chrono::system_clock::now();
	// Construct the locking data-structure, using the bounding-box of the points
	//Triangulation::Lock_data_structure locking_ds( CGAL::Bbox_3(0., 0., 0., 1., 1., 1.), 50);
	Triangulation T(trajPoints.begin(), trajPoints.end());

	//assert(T.number_of_vertices() == traj.size());
	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Done inserting\n";
	std::cout << "Elapsed Time : " << elapsed.count() << " ms\n";

	std::vector<float> cellVolumes(n);

	const int range = traj.size() / 16;
#pragma omp parallel for num_threads(16) schedule(static)
	for (int i = 0; i < traj.size(); i++)
	{
		//cellVolumes[i] = volume_of_finite_dual_cell(T, T.finite_vertices_begin());
		if (omp_get_thread_num() == 0)
		{
			if (i % 16384 == 0)
			{
				std::cout << "\t" << int(100 * double(i) / double(range)) << "%     \r";
			}
		}
		//Triangulation::Vertex_handle vh = T.nearest_vertex(trajPoints[i]);
		cellVolumes[i] = volume_of_finite_dual_cell(T, T.nearest_vertex(trajPoints[i]));
		//std::cout << cellVolumes[i] << "\n";
	}
	std::cout << "Done calculating volumes\n";

	if (average)
	{
		float* dAvg = new float[sX]();
#		pragma omp parallel for num_threads(16) schedule(static)
		for (int i = 0; i < traj.size(); i++)
		{
			dAvg[i%sX] += cellVolumes[i];
		}
		double maxAvg = 0;
		for (int i = 0; i < sX; i++)
		{
			maxAvg += dAvg[i];
		}
		maxAvg = 1.0 / double(maxAvg);
#		pragma omp parallel for num_threads(16) schedule(static)
		for (int i = 0; i < traj.size(); i++)
		{
			cellVolumes[i] = maxAvg * dAvg[(std::min)(i%sX, (int)((sX - 1) * cutoff))];
		}
		delete[] dAvg;
	}
	std::cout << "Done averaging\n";
	double max = 0;

	for (int i = 0; i < traj.size(); i++)
	{
		if (cellVolumes[i]>max) max = cellVolumes[i];
	}
	max = 1.0 / double(max);
#	pragma omp parallel for num_threads(16) schedule(static)
	for (int i = 0; i < traj.size(); i++)
	{
		cellVolumes[i] *= max;
	}

	std::cout << "\n";

	end = std::chrono::system_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);


#pragma omp critical
	{
		std::cout << "Done estimating k-space density. Elapsed Time: " << elapsed.count() << " ms\n";
	}
	return cellVolumes;
}


double fftwAbs(const fftwf_complex& point)
{
	return sqrt(point[0] * point[0] + point[1] * point[1]);
}

std::vector<float> rssq2DVecImg(const std::vector<float*>& img, const int& imgSize)
{
	const int nThreads = 4;
	//const int imgSize = img[0].size() / 2;
	std::vector<float> imgAvg(imgSize);

	for (int ch = 0; ch < img.size(); ch++)
	{
#pragma omp parallel for num_threads(nThreads) schedule(static) default(shared)
		for (int i = 0; i < 2 * imgAvg.size(); i += 2)
		{
			imgAvg[i / 2] += img[ch][i] * img[ch][i] + img[ch][i + 1] * img[ch][i + 1];	// int-division is floor division
		}
	}

#pragma omp parallel for num_threads(nThreads) schedule(static) default(shared)
	for (int i = 0; i < imgAvg.size(); i++)
	{
		imgAvg[i] = sqrt(imgAvg[i]);	// re
	}

	return imgAvg;
}

std::vector<float> rssq2DVecImg(const std::vector<fftwf_complex*>& img, const int& imgSize)
{
	const int nThreads = 1;
	//const int imgSize = img[0].size() / 2;
	std::vector<float> imgAvg(imgSize);
	for (int i = 0; i < imgAvg.size(); i++)
	{
		imgAvg[i] = 0.0;
	}

	for (int ch = 0; ch < img.size(); ch++)
	{
#pragma omp parallel for num_threads(nThreads) schedule(static) default(shared)
		for (int i = 0; i < imgAvg.size(); i++)
		{
			imgAvg[i] += img[ch][i][0] * img[ch][i][0] + img[ch][i][1] * img[ch][i][1];
		}
	}

#pragma omp parallel for num_threads(nThreads) schedule(static) default(shared)
	for (int i = 0; i < imgAvg.size(); i++)
	{
		imgAvg[i] = sqrt(imgAvg[i]);	// re
	}

	return imgAvg;
}


void performDensityCorrection(const Reco& r, fftwf_complex* data, const std::vector<float>& d, const int& nSamplesPerCh, const int& nChannels, const int& nThreads = 1)
{
	if (d.size() == 0) return;

	const int dSize = d.size();

#pragma omp parallel for num_threads(nThreads) default(shared) schedule(static)
	for (int ch = 0; ch < nChannels; ch++)
	{
		for (int i = 0; i < dSize; i++)
		{
			data[ch*dSize + i][0] *= d[i];
			data[ch*dSize + i][1] *= d[i];
		}
	}
	//std::cout << "d perf\n";	
}

void print_endpoint(Halfedge_handle e, bool is_src) {
	std::cout << "\t";
	if (is_src) {
		if (e->has_source())  std::cout << e->source()->point() << std::endl;
		else  std::cout << "point at infinity" << std::endl;
	}
	else {
		if (e->has_target())  std::cout << e->target()->point() << std::endl;
		else  std::cout << "point at infinity" << std::endl;
	}
}

float fftwPhase(const fftwf_complex& a)
{
	if (a[0] > 0) return atan(a[1] / a[0]);
	else if ((a[0]<0) && (a[1] >= 0)) return atan(a[1] / a[0]) + PI;
	else if ((a[0]<0) && (a[1]<0)) return atan(a[1] / a[0]) - PI;
	else if ((a[0] == 0) && (a[1]>0)) return PI / 2.0;
	else if ((a[0] == 0) && (a[1]<0)) return -PI / 2.0;
	else return 0.0;
}

double getUnderSamplingFraction(const Reco& r, fftwf_complex* data)
{
	int counter = 0;
	for (int i = 0; i < r.getSX() * r.getSY() * r.getSZ(); i++)
	{
		if ((data[i][0] == 0) && (data[i][1] == 0)) counter++;
	}

	return double(counter) / double(r.getSX() * r.getSY() * r.getSZ());
}