#include "solver.cpp"
#include "Point.cpp"
#include <ctime>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <vector>


class Tsp : public Solver {
protected:
	std::string basedir;
	std::string method;
	int num_points;
	std::vector<int> path;
	std::vector<Point> points;
	double current_best;
	double tour;
	virtual std::vector<int> generate_path(); 
	void read(std::string fname);
	void load(std::string fname);
	void cache();
	double get_distance(std::vector<int> pp);
	std::vector<int> get_path() { return path; }
	void swap_points(int i, int j, std::vector<int> &vec);

public:
	Tsp() {}
	Tsp(std::string fname) { 
		read(fname); current_best = INF; 
		path = generate_path();
		basedir = "/home/noah/class/discopt/tsp/";
		method = "tsp";
	}
	Tsp(std::string fname, std::string cache) { 
		read(fname); load(cache); 
		basedir = "/home/noah/class/discopt/tsp/";
		method = "tsp";
	}
	std::string output();
	void solve() = 0;
};

void Tsp::swap_points(int i, int j, std::vector<int> &vec) {
	std::swap( vec[i], vec[j] );
}


double Tsp::get_distance(std::vector<int> pp) {
	double pdist = 0.0;
	for (int i = 0; i < num_points; i++) {
		Point start = points[pp[i]];
		Point end;
		if (i + 1 < num_points)
			end = points[pp[i+1]];
		else
			end = points[pp[0]];
		pdist += start.distance_to(end);
	}
	return pdist;
}


std::vector<int> Tsp::generate_path() {
	std::vector<int> p(num_points);
	for (int i = 0; i < num_points; i++)
		p[i] = i;
	std::random_shuffle(p.begin(), p.end());
	return p;
}

void Tsp::read(std::string fname) {
	std::fstream ifp(fname, std::ios::in);
	ifp >> num_points;
	std::cout << "Creating Tsp instance with " << num_points << " cities"
			  << std::endl;
	// Load points into point manager
	for (int i = 0; i < num_points; ++i) {
		double x, y;
		ifp >> x;
		ifp >> y;
		Point p(x, y);
		points.push_back(p);
	}
}

void Tsp::load(std::string fname) {
	std::ifstream is(fname);
	std::istream_iterator<double> start(is), end;
	std::vector<double> p(start, end);
	current_best = p[0];
	std::vector<int> pp(p.begin()+1, p.end());
	path = pp;
	std::cout << "Loaded cached path, current best: " << current_best << std::endl;
	// current = Path(pm.get_points(), path);
}


void Tsp::cache() {
	std::ofstream ofs;
	std::string N = std::to_string(num_points);
	std::string filename = basedir + "testing/solutions/" + method + "_" + N + ".data";
	ofs.open(filename);
	ofs << current_best << ' ';
	for (std::vector<int>::iterator it = path.begin(); it != path.end(); it++) {
		ofs << *it << ' ';
	}
	ofs.close();
}

std::string Tsp::output() {
	std::string out = std::to_string(current_best) + " 0\n";
	for (std::vector<int>::iterator it = path.begin(); it != path.end(); it++) {
		out += std::to_string(*it) + ' ';
	}
	return out;
}


// int main(void) {
// 	std::srand(unsigned ( std::time(0) ) );
// 	std::string fname = "../data/tsp_51_1";
// 	std::string cache = "./solutions/rev_trans_anneal_51.data";
// 	Tsp tsp(fname);
// 	Tsp tsp1(fname, cache);
// 	std::cout << tsp1.output() << std::endl;
// 	std::cout << tsp.get_distance(tsp.get_path()) << std::endl;
// 	tsp1.solve();
// 	return 0;
// }
