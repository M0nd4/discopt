#include <algorithm>
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
		
private:
	vector<Point> pm;
};


class Path {
public:
	Path(vector<Point> ps, vector<int> pp) {
		points = vector<Point>(ps);
		path = vector<int> (pp);
		path_length = 0.0;
	}
	
	Path(int n) {
		path = vector<int> (n);
		points = vector<Point> (n);
		path_length = 0.0;
	}
	
	vector<int> get_path() {
		vector<int> p = vector<int> (path.size());
		std::copy(path.begin(), path.end(), p.begin());
		return p;
	}

	vector<Point> get_points() {
		return points;
	}
	
	Point get_point(int index) {
		int ind = path[index];
		return points[ind];
	}
	
	void swap_points(int i, int j) {
		std::swap( path[i], path[j] );
	}

	void set_point(int index, int p) {
		path[index] = p;
		path_length = 0.0;
	}
		
	void generate_path() {
		for (int i = 0, n = points.size(); i < n; ++i) {
			set_point(i, i);
		}
		std::random_shuffle( path.begin(), path.end() );
	}
	
	unsigned int path_size() {
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
	
private:
	vector<int> path;
	vector<Point> points;
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
	
// 	Path p = Path(pm.size());
// 	p.generate_path(pm);
// 	vector<int> path = p.get_path();
	
// 	cout << "Initial path distance:" << p.get_distance() << endl;
// 	for (int i = 0, n = path.size(); i < n; ++i) {
// 		// cout << pm.get_point(i) << endl;
// 		cout << path[i] << endl;
// 	}

// 	// swap cities at random positions in path
// 	int c1 = rand() % p.path_size();
// 	int c2 = rand() % p.path_size();
// 	// Point cc1 = p.get_point(c1);
// 	// Point cc2 = p.get_point(c2);
// 	// cout << "swapping: " << cc1 << " with " << cc2 << endl;

// 	p.set_point(c2, c1);
// 	p.set_point(c1, c2);
// 	path = p.get_path();
// 	cout << "Next path distance: " << p.get_distance() << endl;
// 	for (int i = 0, n = path.size(); i < n; ++i) {
// 		// cout << pm.get_point(i) << endl;
// 		cout << path[i] << endl;
// 	}

// 	cout << p.get_distance() << endl;
	
// }
