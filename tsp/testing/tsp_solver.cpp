#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include "TspBasicAnneal.cpp"

int main(int argc, char *argv[]) {
	srand(unsigned ( std::time(0) ) );
	if (argc < 2) {
		std::cout << "Usage: ./tsp_solver <config.txt>" << std::endl;
		return 1;
	}
	
	// Parameters
	std::string method;
	std::string data;
	std::string cache;
	double temp, cool, maxtour;

	// Parse config file
	std::ifstream infile(argv[1]);
	std::string line;
	int pos = 0;

	while (std::getline(infile, line))
	{
		if (line.length() == 0 || line[0] == '#')
			continue;
		else if (pos == 0) {
			method = line;
			pos++;
		}
		else if (pos == 1) {
			std::istringstream iss(line);
			if (method.compare("BasicAnneal") == 0) {
				std::cout << "Using BasicAnneal..." << std::endl;
				if (!(iss >> temp >> cool >> maxtour)) {
					std::cout << "BasicAnneal requires: temp, cool, maxtour"
							  << std::endl;
				}
				std::cout << "Temp: " << temp << ", cool: " << cool
						  << ", max tour: " << maxtour << std::endl;
			}
			pos++;
		}
		else if (pos == 2) {
			data = line;
			pos++;
		}
		else if (pos == 3) {
			cache = line;
			pos++;
		}
		else {
			continue;
		}
	}
	infile.close();
	
	if (data.length() == 0) {
		std::cout << "Input data required but none found." << std::endl;
		return 3;
	}
	
	std::cout << "Data: " << data << ", cache: " << cache << std::endl;

	// Run Tsp solver
	Tsp *tsp;
	TspBasicAnneal tspba;
	if (method.compare("BasicAnneal") == 0) {
		if (cache.length() > 0) {
			tspba = TspBasicAnneal(data, cache, temp, cool, maxtour);
		}
		else {
			tspba = TspBasicAnneal(data, temp, cool, maxtour);
		}
		tsp = &tspba;
	}
	
	tsp->solve();
	std::cout << tsp->output() << std::endl;

	return 0;
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
