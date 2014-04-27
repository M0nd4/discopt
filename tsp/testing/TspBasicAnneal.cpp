#include "Tsp.cpp"
#include <ctime>
#include <cmath>
#include <algorithm>


class TspBasicAnneal : public Tsp {
protected:
	double temp, cooling_rate, max_tour;
	
	double acceptance_probability(const double energy, 
								  const double new_energy, 
								  const double temp) {
		if (new_energy < energy)
			return 1.0;
		return std::exp((energy - new_energy) / temp);
	}
	double anneal();
	double tour_len() { return current_best; }

public:
	TspBasicAnneal() {}
	TspBasicAnneal(std::string fname, double mtemp, double mcooling_rate, 
			  double mmax_tour = std::pow(2,32))
		: Tsp(fname), temp(mtemp), cooling_rate(mcooling_rate), max_tour(mmax_tour) {
		method = "basic_anneal";
	}
	TspBasicAnneal(std::string fname, std::string cache, double mtemp, 
			  double mcooling_rate, double mmax_tour = std::pow(2,32))
		: Tsp(fname, cache), temp(mtemp), cooling_rate(mcooling_rate), max_tour(mmax_tour) {
		method = "basic_anneal";
	}

	void solve();
};

void TspBasicAnneal::solve() {
	tour = anneal();
	int restart = 0;
	while (tour > max_tour) {
		int iter = 0;
	
		for (int i = 0; i < 10*num_points; i++) {
			tour = anneal();
			if (tour < current_best) {
				std::cout << "Saving..." << std::endl;
				cache();
				current_best = tour;
			}
			iter++;
		}
		std::cout << "Restart: " << restart << ", Current tour: " 
				  << tour << std::endl;
		restart++;
		
		// reset to examine other starting locations
		path = generate_path();
	}
}


double TspBasicAnneal::anneal() {
	std::vector<int> cur(path);
	tour = get_distance(path);
	double t = temp;
	// std::cout << "Initial distance: " << tour << std::endl;
	while (t > 1) {
		std::vector<int> new_path(cur);
		
		// swap cities at random positions in path
		int c1 = rand() % num_points;
		int c2 = rand() % num_points;
		while (c1 == c2) {
			c2 = rand() % num_points;
		}
		swap_points(c1, c2, new_path);
		
		// energies of solutions
		double cur_energy = get_distance(cur);
		const double nbr_energy = get_distance(new_path);
		// std::cout << "cur energy: " << cur_energy << std::endl;
		// std::cout << "nbr energy: " << nbr_energy << std::endl;
		
		// test for acceptance of neighboring solution
		const double prob = rand() / (double) RAND_MAX;
		// cout << acceptance_probability(cur_energy, nbr_energy, t) << endl;
		if (acceptance_probability(cur_energy, nbr_energy, t) > prob) {
			// std::cout << "Accepted neighbor" << std::endl;
			cur = new_path;
			cur_energy = nbr_energy;
		}
		// update best solution
		if (cur_energy < tour) {
			path = std::vector<int>(cur);
			tour = cur_energy;
		}
		// update temp
		t *= (1 - cooling_rate);
	}
	
	// return tour_length;
	return tour;
}


// int main(void) {
// 	std::srand(unsigned ( std::time(0) ) );
// 	std::string fname = "../data/tsp_51_1";
// 	std::string cache = "./solutions/rev_trans_anneal_51.data";
// 	TspBasicAnneal tsp(fname, 10, .0005, 440);
// 	tsp.solve();
// 	std::cout << tsp.output() <<std::endl;
// 	tsp.cache();
// 	return 0;
// }
