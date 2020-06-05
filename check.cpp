#include "bits/stdc++.h"
#include "ext/pb_ds/assoc_container.hpp"
#include "ext/pb_ds/tree_policy.hpp"
#include "ext/rope"
using namespace std;
using namespace chrono;
using namespace __gnu_pbds;
using namespace __gnu_cxx;
mt19937 rng(high_resolution_clock::now().time_since_epoch().count());
mt19937_64 rngll(high_resolution_clock::now().time_since_epoch().count());
template<typename T> T ctmax(T &x, const T &y){ return x = max(x, y); }
template<typename T> T ctmin(T &x, const T &y){ return x = min(x, y); }
template<typename T> using Tree = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;



int main(int argc, char *argv[]){
	cin.tie(0)->sync_with_stdio(0);
	string bad_sol = argv[1], checker = argv[2];
	cout << "Print the results? Type (y) or (n): ";
	cout.flush();
	char X;
	cin >> X;
	for(int i = 0; ; ++ i){
		system("./stress/_gen>./in");
		auto p1 = high_resolution_clock::now();
		system(("./" + bad_sol + "<./in>./stress/out_bad").c_str());
		auto p2 = high_resolution_clock::now();
		system(("./" + checker + "<./stress/out_bad>./stress/_res").c_str());
		ifstream _res("./stress/_res"), badin("./stress/out_bad");
		int res;
		_res >> res;
		vector<string> bad;
		string t;
		while(badin >> t) bad.push_back(t);
		cout << "Case #" << i << "\n";
		cout << "Bad: " << duration<double>(p2 - p1).count() << " seconds" << endl;
		if(!res){
			cout << "Failed!\n";
			cout << "Bad = ";
			for(auto s: bad){
				cout << s << " ";
			}
			cout << "\n";
			break;
		}
		cout << "Ok\n";
		if(X == 'y'){
			cout << "Bad = ";
			for(auto s: bad){
				cout << s << " ";
			}
			cout << "\n";
		}
		cout << endl;
	}
	return 0;
}

/*

*/

////////////////////////////////////////////////////////////////////////////////////////
//                                                                                    //
//                                   Coded by Aeren                                   //
//                                                                                    //
////////////////////////////////////////////////////////////////////////////////////////