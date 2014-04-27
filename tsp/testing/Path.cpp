#include "PathManager.cpp"
#include <vector>

using namespace std;

class Path {
public:
	Path(vector<Point> p) {
		path = vector<Point>(p);
	}
	
	// Path(PathManager pm) {
		
	// }
	
	vector<Point> get_path() {
		return path;
	}
	
	Point get_point(int index) {
		return path[index];
	}
	
	
private:
	vector<Point> path;
	double path_length;
};

// ostream& operator<< (ostream& out, Point p)
// {
// 	cout << "(" << p.getX() << ", " << p.getY() << ")" << endl;
// 	return out;
// }


int main(void)
{
	PathManager pm = new PathManager();
}
