#include <ctime>
#include <vector>
#include "Point.cpp"
#include <algorithm>

class PM {
public:
	PM(): pm( vector<Point> (0) ) {}
	void add_point(Point p) {
		pm.push_back(p);
	}
	Point get_point(int i) {
		return pm[i]; 
	}
	int size() {
		return pm.size();
	}
	
	vector<Point> get_points() {
		return pm;
	}
		
protected:
	vector<Point> pm;
};


class Path : public PM {
public:
	Path() : path(std::vector<int>(0)), path_length(0.0) {}
	Path(PM npm, vector<int> pp) {
		pm = npm.get_points();
		path = vector<int>(pp);
		path_length = 0.0;
		path_length = get_distance();
	}
	Path(PM npm) {
		pm = npm.get_points();
		path = vector<int>(npm.size());
		generate_path();
	}
	Path(Path &o) {
		pm = o.pm;
		path = std::vector<int> (o.get_path());
		path_length = o.path_length;
	}
	Path(vector<Point> ps, vector<int> pp) {
		pm = vector<Point>(ps);
		path = vector<int> (pp);
		path_length = 0.0;
	}
		
	vector<int> get_path() {
		return path;
	}
	
	Point get_point(int index) {
		int ind = path[index];
		return pm[ind];
	}
	
	void swap_points(int i, int j) {
		path_length = 0.0;
		std::swap( path[i], path[j] );
	}

	void set_point(int index, int p) {
		path[index] = p;
		path_length = 0.0;
	}
		
	void generate_path() {
		for (int i = 0, n = pm.size(); i < n; ++i) {
			set_point(i, i);
		}
		std::random_shuffle( path.begin(), path.end() );
	}
	
	unsigned int size() {
		return path.size();
	}
	
	double get_distance() {
		if (path_length == 0) {
			double path_distance = 0.0;
			for (int i = 0, n = path.size(); i < n; ++i) {
				Point start = get_point(i);
				Point end;
				if (i + 1 < n)
					end = get_point(i + 1);
				else
					end = get_point(0);
				path_distance += start.distance_to(end);
			}
			path_length = path_distance;
		}
		return path_length;
	}
	
protected:
	vector<int> path;
	double path_length;
};


// int main(void)
// {
// 	srand(time(NULL));
// 	PM pm;
// 	for (int i = 0; i < 10; ++i) {
// 		int x = rand() % 100;
// 		int y = rand() % 100;
// 		Point p = Point(x, y);
// 		pm.add_point(p);
// 	}
	
// 	Path p = Path(pm);
// 	vector<int> path = p.get_path();
	
// 	cout << "Initial path distance:" << p.get_distance() << endl;
// 	for (int i = 0, n = path.size(); i < n; ++i) {
// 		// cout << pm.get_point(i) << endl;
// 		cout << path[i] << ' ';
// 	}
// 	cout << endl;
	
// 	// swap cities at random positions in path
// 	int c1 = rand() % p.size();
// 	int c2 = rand() % p.size();

// 	p.swap_points(c1, c2);
// 	path = p.get_path();
// 	cout << "Swapped path distance: " << p.get_distance() << endl;
// 	for (int i = 0, n = path.size(); i < n; ++i) {
// 		// cout << pm.get_point(i) << endl;
// 		cout << path[i] << ' ';
// 	}
// 	cout << endl;

// 	// Path copy
// 	Path a = Path(p);
// 	a.generate_path();
// 	vector<int> apath = a.get_path();
// 	cout << "Copied path: " << endl;
// 	for (int i = 0; i < a.size(); i ++) {
// 		cout << apath[i] << ' ';
// 	}
// 	cout << '\n';
// 	vector<int> ppath = p.get_path();
// 	cout << "Original path: " << endl;
// 	for (int i = 0; i < p.size(); i ++) {
// 		cout << ppath[i] << ' ';
// 	}
// 	cout << '\n';
// }
