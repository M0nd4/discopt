#include <vector>
#include <utility>
#include <map>
#include <iostream>
#include "PathManager.cpp"

using namespace std;

struct Coord {
    int x;
	int y;
};


class Neighborhood {
public:
	
private:
	int dim;
	int nsize;
	vector<double> x_cuts;
	vector<double> y_cuts;
	
};


int main(void) 
{
	Coord one;
	one.x = 0; 
	one.y = 1;
	cout << "(" << one.x << ", " << one.y << ")" << endl;
	Point a = Point(1.0, 2.0);
	vector<Point> points;
	points.push_back(a);

	// http://stackoverflow.com/questions/649793/howto-create-map-of-vector-from-sorted-data
	map<Coord, std::vector<Point> > hoods;
	// hoods[one].push_back(points);

}
