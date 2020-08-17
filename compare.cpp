#include <bits/stdc++.h>
using namespace std;
using namespace chrono;

void check_status(int status, const string &where){
	if(status < 0) cout << where << " -> error: " << strerror(errno) << "\n";
	else if(status >> 8) cout << where << " -> program exited abnormally\n";
	else return;
	exit(0);
}

int main(int argc, char *argv[]){
	cin.tie(0)->sync_with_stdio(0);
	string bad_sol = argv[1], good_sol = argv[2];
	cout << "Print the results? Type (y) or (n): ";
	cout.flush();
	char X;
	cin >> X;
	for(int i = 0; ; ++ i){
		check_status(system("./stress/_gen>./in"), "Generator");
		auto p1 = high_resolution_clock::now();
		check_status(system(("./" + good_sol + "<./in>./stress/out_good").c_str()), "Good Sol");
		auto p2 = high_resolution_clock::now();
		check_status(system(("./" + bad_sol + "<./in>./stress/out_bad").c_str()), "Bad Sol");
		auto p3 = high_resolution_clock::now();
		ifstream goodin("./stress/out_good"), badin("./stress/out_bad");
		vector<string> good, bad;
		string t;
		while(goodin >> t) good.push_back(t);
		while(badin >> t) bad.push_back(t);
		cout << "Case #" << i << "\n";
		cout << "Good: " << duration<double>(p2 - p1).count() << " seconds\n";
		cout << "Bad: " << duration<double>(p3 - p2).count() << " seconds" << endl;
		if(good != bad){
			cout << "Failed\n";
			cout << "good = ";
			for(auto s: good) cout << s << " ";
			cout << "\n";
			cout << "bad = ";
			for(auto s: bad) cout << s << " ";
			cout << "\n";
			break;
		}
		cout << "Ok\n";
		if(X == 'y'){
			cout << "good = ";
			for(auto s: good) cout << s << " ";
			cout << "\n";
			cout << "bad = ";
			for(auto s: bad) cout << s << " ";
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