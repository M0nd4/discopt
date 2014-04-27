/*Abstract Solver class for c++ implementations
  Methods: 
  1) Read input
  2) output solution string
  3) cache solution
  4) load cached solution
  4) solve
*/
#include <string>
#define INF std::pow(2,32);

class Solver {
protected:
	virtual void read(std::string fname) = 0;
	virtual void cache() = 0;
	virtual void load(std::string fname) = 0;
	
public:
	virtual std::string output() = 0;
	virtual void solve() = 0;
};


