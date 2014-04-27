#include <iostream>
#include <cmath>


class Point {
public:
	Point();
	Point(double x, double y): x(x), y(y) {};
	double get_x() const { return x; }
	double get_y() const { return y; }
	double distance_to(Point p) {
		return sqrt(std::pow(p.get_x() - x, 2) + std::pow(p.get_y() - y, 2));
	}
	
private:
	double x, y;
};

Point::Point()
{
	x = rand() % 100 + 1;
	y = rand() % 100 + 1;
	
}

std::ostream& operator<< (std::ostream& out, Point p)
{
	std::cout << "(" << p.get_x() << ", " << p.get_y() << ")";
	return out;
}

// int main(void) {
// 	Point a(1.0, 2.0);
// 	Point b(4.0, 6.0);
// 	std::cout << "a: " << a << ", b: " << b << std::endl;
// 	std::cout << "Distance between: " << a.distance_to(b) << std::endl;
// }
