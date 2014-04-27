#include <string>
#include <cassert>
#include <climits>
#include "PathManager.cpp"
#include <cmath>
#include <ctime>
#include <iostream>
#include <iterator>
#include <fstream>
#include <vector>
#include <algorithm> // for std::copy

double acceptance_probability(double energy, double new_energy, double temp) {
	if (new_energy < energy)
		return 1.0;
	return std::exp((energy - new_energy) / temp);
}

class Anneal {
	
public:
	
	Anneal(vector<Point> ps, vector<int> p, double t, double cr) {
		points = vector<Point>(ps);
		tour = vector<int>(p);
		temp = (double) t;
		cooling_rate = (double) cr;

		Path best = Path(vector<Point>(points), vector<int>(tour));
		Path current = Path(vector<Point>(points), vector<int>(tour));
		
		// cout << "Starting temp: " << temp << endl;
		while (temp > 1) {
			Path new_path = Path(current.get_points(), current.get_path());
		
			// swap cities at random positions in path
			int c1 = rand() % new_path.path_size();
			int c2 = rand() % new_path.path_size();
			while (c1 == c2) {
				c2 = rand() % new_path.path_size();
			}
			new_path.swap_points(c1, c2);
		
			// energies of solutions
			double current_energy = current.get_distance();
			double nbr_energy = new_path.get_distance();
			// cout << "current energy: " << current_energy << endl;
			// cout << "nbr energy: " << nbr_energy << endl;
		
			// test for acceptance of neighboring solution
			double prob = (double) rand() / (RAND_MAX);
			// cout << acceptance_probability(current_energy, nbr_energy, temp) << endl;
			if (acceptance_probability(current_energy, nbr_energy, temp) > prob) {
				// cout << "Accepted neighbor" << endl;
				current = Path(current.get_points(), new_path.get_path());
			}
			// update best solution
			if (current.get_distance() < best.get_distance())
				best = Path(current.get_points(), new_path.get_path());
				
			// update temp
			temp *= (1 - cooling_rate);
		}
		tour = best.get_path();
		tour_length = best.get_distance();
		// return tour_length;
	}
	
	vector<int> get_tour() {
		return tour;
	}
	
	double get_tour_length() {
		return tour_length;
	}
	
	void cache() {
		ofstream ofs;
		std::string N = std::to_string(tour.size());
		std::string filename = "/home/noah/class/discopt/tsp/testing/solutions/anneal_";
		std::string suffix = ".data";
		filename.append(N);
		filename.append(suffix);
		ofs.open(filename);
		ofs << tour_length << ' ';
		for (vector<int>::iterator it = tour.begin(); it != tour.end(); it++) {
			ofs << *it << ' ';
		}
		ofs.close();
	}
private:
	double temp;
	double cooling_rate;
	vector<Point> points;
	vector<int> tour;
	double tour_length;
};


int main(int argc, char * argv[])
{
	if (argc < 2) {
		cout << "Usage: ./tsp_anneal <filename> &optional<cached_path>" << endl;
		return 1;
	}
	
	char * filename = argv[1];
	ifstream ifp(filename, ios::in);
	int num_points;
	ifp >> num_points;
	cout << "Number of points: " << num_points << endl;
	
	// Load points into point manager
	PM pm;
	for (int i = 0; i < num_points; ++i) {
		double x, y;
		ifp >> x;
		ifp >> y;
		Point p = Point(x, y);
		pm.add_point(p);
	}
	srand((unsigned)time(0));

	// max tour
	double max_tour = 430;
	
	// Set temperature
	double temp = 10;
	
	// cooling rate
	double cooling_rate = 0.0005;

	// Create an initial solution
	Path current = Path(pm.size());
	
	// Load cached path if supplied
	vector<double> p;
	vector<int> path;
	double current_best = (double) INT_MAX;
	if (argc == 3) {
		char * cache = argv[2];
		std::ifstream is(cache);
		std::istream_iterator<double> start(is), end;
		std::vector<double> p(start, end);
		current_best = p[0];
		std::vector<int> path = vector<int> (p.begin()+1, p.end());
		current = Path(pm.get_points(), path);
	}
	else {
		current = Path(pm.size());
		current.generate_path();
	}
		
	// best solution starts with initial
	Path best = Path(pm.get_points(), current.get_path());
	cout << "Initial distance: " << best.get_distance() << endl;
	
	// Print initial solution
	// vector<int> init_path = current.get_path();
	// for (vector<int>::iterator it=init_path.begin(); it!=init_path.end(); it++) {
	// 	cout << *it << ' ';
	// }
	// cout << '\n';
	
	// Anneal
	Anneal tour = Anneal(pm.get_points(), best.get_path(), temp, cooling_rate);
	double tour_len = tour.get_tour_length();
	int restart = 0;
	while (tour_len > max_tour) {
		int iter = 0;
	
		for (int i = 0; i < 2000; i++) {
			tour = Anneal(pm.get_points(), tour.get_tour(), temp, cooling_rate);
			tour_len = tour.get_tour_length();
			if (tour_len < current_best) {
				cout << "Saving..." << endl;
				tour.cache();
				current_best = tour_len;
			}
			iter++;
		}
		cout << "Restart: " << restart << ", Current tour: " << tour_len << endl;
		restart++;
		
		// reset to examine other starting locations
		best.generate_path();
		tour = Anneal(pm.get_points(), best.get_path(), temp, cooling_rate);
		tour_len = tour.get_tour_length();
		
	}
	
	// Print solution
	// vector<int> result = tour.get_tour();
	// cout << "Final distance from anneal: " << tour_len << endl;
	// for (int i = 0, n = result.size(); i < n; ++i) {
	// 	cout << result[i] << " ";
	// }
	// cout << endl;
}


