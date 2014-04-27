#include <vector>
#include <iostream>

using namespace std;

class test {
public:
	test(): v(vector<int>(10,1)) {}
	int size() {
		return v.size();
	}
	
	void set_v(int index, int num) {
		v[index] = num;
	}
	
	int get_v(int index) {
		return v[index];
	}
	
private:
	vector<int> v;
};

int main()
{
	test t = test();
	for (int i = 0; i < t.size(); ++i) {
		cout << t.get_v(i) << " ";
	}
	cout << endl;
	
	t.set_v(0, 2);
	for (int i = 0; i < t.size(); ++i) {
		cout << t.get_v(i) << " ";
	}
	cout << endl;
	
}
	
	
	
