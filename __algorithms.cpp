/***************************************************************************************************************

                                   This is Aeren's C++ algorithm template library                     
                                            for competitive programming

****************************************************************************************************************


Category


1. Number Theory
	1.1. Modular Exponentiation, Modular Inverse
		156485479_1_1
	1.2. Extended Euclidean Algorithm / Linear Diophantine Equation
		156485479_1_2
	1.3. Number Theory
		156485479_1_3
	1.4. Combinatorics
		156485479_1_4
	1.5. Millar Rabin Primality Test / Pollard Rho Algorithm
		156485479_1_5
	1.6. Tonelli Shanks Algorithm ( Solution to x^2 = a mod p )
		156485479_1_6
	1.7. Chinese Remainder Theorem
		156485479_1_7
	1.8. Lehman Factorization
		156485479_1_8
	1.9. Polynomial Class
		156485479_1_9
	1.10. Discrete Log
		156485479_1_10
	1.11. Continued Fraction
		156485479_1_11
	1.12. Meissel–Lehmer Algorithm / Fast Computaion of pi(N)
		156485479_1_12
	1.13. Xudyh's Sieve
		156485479_1_13
	1.14. Formulae Regarding Mobius Inversion
		156485479_1_14


2. Numeric
	2.1. Linear Recurrence Relation Solver / Berlekamp Massey Algorithm
		156485479_2_1
	2.2. System of Linear Equations
		2.2.1. Coefficients in R
			156485479_2_2_1
		2.2.2. Coefficients in Z_p
			156485479_2_2_2
		2.2.3. Coefficients in Z_2
			156485479_2_2_3
	2.3. Matrix
		2.3.1. Entries in R
			156485479_2_3_1
		2.3.2. Entries in some semiring
			156485479_2_3_2
		2.3.3. Entries in a finite field of characteristic 2
			156485479_2_3_3
	2.4. Convolution
		2.4.1. Addition Convolution
			2.4.1.1 Fast Fourier Transform
				156485479_2_4_1_1
			2.4.1.2. Number Theoric Transform
				156485479_2_4_1_2
		2.4.2. Bitwise Convolution ( Fast Walsh Hadamard Transform, FWHT )
				156485479_2_4_2
	2.5. Binary Search
		156485479_2_5
	2.6. DP Optimization
		2.6.1. Convex Hull Trick ( Line Containers / Li Chao Tree )
			2.6.1.1. Sorted Line Container
				156485479_2_6_1_1
			2.6.1.2. Line Container
				156485479_2_6_1_2
			2.6.1.3. Li Chao Tree
				156485479_2_6_1_3
		2.6.2. Divide and Conquer
			156485479_2_6_2
		2.6.3. Knuth
			156485479_2_6_3
		2.6.4. Lagrange ( Aliens Trick, Wqs Binary Search )
			156485479_2_6_4
		2.6.5. Monotone Queue
			156485479_2_6_5
	2.7. Kadane
		156485479_2_7
	2.8. BigInteger
		156485479_2_8
	2.9. Modular Arithmetics
		156485479_2_9
	2.10. Matroid
		2.10.1. Matroid Intersection
			156485479_2_10_1
		2.10.2. Matroid Union < INCOMPLETE >
			156485479_2_10_2
	2.11. LIS
		156485479_2_11
	2.12 K Dimensional Array
		156485479_2_12
	2.13 K Dimensional Prefix Sum
		156485479_2_13


3. Data Structure
	3.1. Sparse Table / 2D
		156485479_3_1
	3.2. Segment Tree
		3.2.1. Simple Iterative Segment Tree
			156485479_3_2_1
		3.2.2. Iterative Segment Tree with Reversed Operation
			156485479_3_2_2
		3.2.3. 2D Segment Tree
			156485479_3_2_3
		3.2.4. Recursive Lazy Segment Tree
			156485479_3_2_4
		3.2.5. Lazy Segment Tree
			3.2.5.1. Iterative
				156485479_3_2_5_1
			3.2.5.2. Dynamic
				156485479_3_2_5_2
		3.2.6. Persistent Segment Tree
			156485479_3_2_6
	3.3. Fenwick Tree
		3.3.1. Simple Fenwick Tree
			156485479_3_3_1
		3.3.2. Fenwick Tree Supporting Range Queries of The Same Type
			156485479_3_3_2
		3.3.3. 2D Fenwick Tree
			156485479_3_3_3
	3.4. Wavelet Tree
		156485479_3_4
	3.5. Disjoint Set
		156485479_3_5
	3.6. Monotone Stack
		156485479_3_6
	3.7. Less-than-k Query / Distinct Value Query
		156485479_3_7
	3.8. Mo's Algorithm
		156485479_3_8
	3.9. Treap
		156485479_3_9
	3.10. Splay Tree
		156485479_3_10
	3.11. Link Cut Trees
		156485479_3_11
	3.12. Unital Sorter
		156485479_3_12
	3.13. AAA Tree < INCOMPLETE >
		156485479_3_13
	3.14. Bit Trie
		156485479_3_14
	3.15. Query Tree
		156485479_3_15


4. Graph
	4.1. Strongly Connected Component ( Tarjan's Algorithm ) / Condensation / Strong Augmentation
		156485479_4_1
	4.2. Biconnected Component
		156485479_4_2
	4.3. Articulation Points
		156485479_4_3
	4.4. Flow / Matching / Cut
		4.4.1. Dinic's Maximum Flow Algorithm
			156485479_4_4_1
		4.4.2. Minimum Cost Maximum Flow Algorithm
			156485479_4_4_2
		4.4.3. Simple DFS Matching
			156485479_4_4_3
		4.4.4. Hopcroft Karp Algorithm / Fast Bipartite Matching
			156485479_4_4_4
		4.4.5. Hungarian Algorithm / Minimum Cost Maximum Bipartite Matching ( WARNING: UNTESTED )
			156485479_4_4_5
		4.4.6. Global Min Cut < UNTESTED >
			156485479_4_4_6
		4.4.7. Gomory-Hu Tree < INCOMPLETE >
			156485479_4_4_7
		4.4.8. General Matching < INCOMPLETE >
			156485479_4_4_8
	4.5. Tree Algorithms
		4.5.1. LCA ( Unweighted / Weighted )
			156485479_4_5_1
		4.5.2. Binary Lifting ( Unweighted / Weighted )
			156485479_4_5_2
		4.5.3. Vertex Update Path Query
			156485479_4_5_3
		4.5.4. Heavy Light Decomposition
			156485479_4_5_4
		4.5.5. Centroid / Centroid Decomposition
			156485479_4_5_5
		4.5.6. AHU Algorithm ( Rooted Tree Isomorphism ) / Tree Isomorphism
			156485479_4_5_6
		4.5.7. Minimum Spanning Forest
			156485479_4_5_7
		4.5.8. Minimum Spanning Arborescence
			156485479_4_5_8
		4.5.9. Compressed Tree ( Virtual Tree, Auxiliary Tree )
			156485479_4_5_9
		4.5.10. Pruefer Code / Decode
			156485479_4_5_10
	4.6. Shortest Path Tree
		4.6.1. On Sparse Graph ( Dijkstra, Bellman Ford, SPFA )
			156485479_4_6_1
		4.6.2. On Dense Graph ( Dijkstra, Floyd Warshall )
			156485479_4_6_2
	4.7. Topological Sort
		156485479_4_7
	4.8. Two Satisfiability
		156485479_4_8
	4.9. Euler Walk
		156485479_4_9
	4.10. Dominator Tree
		156485479_4_10


5. String
	5.1. Lexicographically Minimal Rotation
		156485479_5_1
	5.2. Manacher's Algorithm ( Find All Palindromic Substrings )
		156485479_5_2
	5.3. Suffix Array and Kasai's Algorithm
		156485479_5_3
	5.4. Z Function
		156485479_5_4
	5.5. Aho Corasic Automaton
		156485479_5_5
	5.6. Prefix Function / Prefix Automaton
		156485479_5_6
	5.7. Polynomial Hash
		156485479_5_7
	5.8. Suffix Automaton
		156485479_5_8
	5.9. Suffix Tree < UNTESTED >
		156485479_5_9
	5.10. Palindrome Tree / Eertree
		156485479_5_10
	5.11. Levenshtein Automaton < INCOMPLETE >
		156485479_5_11
	5.12. Burrows Wheeler Transform / Inverse
		156485479_5_12
	5.13. Main Lorentz Algorithm ( Find All Tandem ( Square ) Substrings )
		156485479_5_13


6. Geometry
	6.1. 2D Geometry
		156485479_6_1
	6.2. Convex Hull and Minkowski Addition
		156485479_6_2
	6.3. KD Tree < UNTESTED >
		156485479_6_3
	6.4. Line Sweep
		6.4.1. Find a Pair of Intersecting Segments < INCOMPLETE >
			156485479_6_4_1
		6.4.2. Find the Closest Pair of Points
			156485479_6_4_2
	6.5. Circle Class
		156485479_6_5

7. Heuristics Algorithms
	7.1. Maximum Independent Set
		156485479_7_1

8. Miscellaneous
	8.1. Custom Hash Function for unordered_set and unordered map
		156485479_8_1
	8.2. Bump Allocator
		156485479_8_2
	8.3. Debug
		156485479_8_3
	8.4. Random Generator
		156485479_8_4
	8.5. Barrett Reduction
		156485479_8_5


***************************************************************************************************************/

// 156485479_1_1
// Modular Exponentiation, Modular Inverse and Geometric Sum
// O(log e)
long long modexp(long long b, long long e, const long long &mod){
	long long res = 1;
	for(; e; b = b * b % mod, e >>= 1) if(e & 1) res = res * b % mod;
	return res;
}
long long modinv(long long a, const long long &mod){
	return modexp(a, mod - 2, mod);
}
long long modgeo(long long b, long long e, const long long &mod){
	if(e < 2) return e;
	long long res = 1;
	for(long long bit = 1 << 30 - __builtin_clz(e), p = 1; bit; bit >>= 1){
		res = res * (1 + p * b % mod) % mod, p = p * p % mod * b % mod;
		if(bit & e) res = (res + (p = p * b % mod)) % mod;
	}
	return res;
}
template<typename T>
T binexp(T b, long long e){
	T res = 1;
	for(; e; b *= b, e >>= 1) if(e & 1) res *= b;
	return res;
}
template<typename T>
T bingeo(const T &b, long long e){
	if(e < 2) return e;
	T res = 1, p = 1;
	for(long long bit = 1 << 30 - __builtin_clz(e); bit; bit >>= 1){
		res *= 1 + p * b, p *= p * b;
		if(bit & e) res += (p *= b);
	}
	return res;
}
template<typename T, typename BO>
T binexp(T b, long long e, BO bin_op, const T &id){
	T res = id;
	for(; e; b = bin_op(b, b), e >>= 1) if(e & 1) res = bin_op(res, b);
	return res;
}
template<typename T, typename AO, typename MO>
T bingeo(const T &b, long long e, AO add_op, const T &add_id, MO mul_op, const T &mul_id){
	if(e < 2) return e ? mul_id : add_id;
	T res = mul_id, p = mul_id;
	for(long long bit = 1 << 30 - __builtin_clz(e); bit; bit >>= 1){
		res = mul_op(res, add_op(mul_id, mul_op(p, b))), p = mul_op(mul_op(p, p), b);
		if(bit & e) res = add_op(res, p = mul_op(p, b));
	}
	return res;
}

// 156485479_1_2
// Extended Euclidean Algorithm / Linear Diophantine Equation
// O(max(log x, log y))
typedef long long ll;
ll euclid(ll a, ll b, ll &x, ll &y){
	if(b){
		ll d = euclid(b, a % b, y, x);
		return y -= a / b * x, d;
	}
	return x = 1, y = 0, a;
}
// solutions to ax + by = c where x in [xlow, xhigh] and y in [ylow, yhigh]
// cnt, leftsol, rightsol, gcd of a and b
array<ll, 6> solve_linear_diophantine(ll a, ll b, ll c, ll xlow, ll xhigh, ll ylow, ll yhigh){
	ll x, y, g = euclid(abs(a), abs(b), x, y);
	array<ll, 6> no_sol{0, 0, 0, 0, 0, g};
	if(c % g) return no_sol;
	x *= c / g, y *= c / g;
	if(a < 0) x = -x;
	if(b < 0) y = -y;
	a /= g, b /= g, c /= g;
	auto shift = [&](ll &x, ll &y, ll a, ll b, ll cnt){ x += cnt * b, y -= cnt * a; };
	int sign_a = a > 0 ? 1 : -1, sign_b = b > 0 ? 1 : -1;

	shift(x, y, a, b, (xlow - x) / b);
	if(x < xlow) shift(x, y, a, b, sign_b);
	if(x > xhigh) return no_sol;
	ll lx1 = x;
	
	shift(x, y, a, b, (xhigh - x) / b);
	if(x > xhigh) shift(x, y, a, b, -sign_b);
	ll rx1 = x;

	shift(x, y, a, b, -(ylow - y) / a);
	if(y < ylow) shift(x, y, a, b, -sign_a);
	if(y > yhigh) return no_sol;
	ll lx2 = x;

	shift(x, y, a, b, -(yhigh - y) / a);
	if(y > yhigh) shift(x, y, a, b, sign_a);
	ll rx2 = x;

	if(lx2 > rx2) swap(lx2, rx2);
	ll lx = max(lx1, lx2), rx = min(rx1, rx2);
	if(lx > rx) return no_sol;
	return {(rx - lx) / abs(b) + 1, lx, (c - lx * a) / b, rx, (c - rx * a) / b, g};
}

// 156485479_1_3
// Number Theory
template<int SZ>
struct number_theory{
	vector<int> lpf, prime, mu, phi; // least prime factor, primes, mobius function, totient function, number of multiples
	number_theory(): lpf(SZ + 1), mu(SZ + 1, 1), phi(SZ + 1, 1){ // O(SZ)
		for(int i = 2; i <= SZ; ++ i){
			if(!lpf[i]) lpf[i] = i, prime.push_back(i);
			if(i / lpf[i] % lpf[i]) mu[i] = -mu[i / lpf[i]], phi[i] = phi[i / lpf[i]] * (lpf[i] - 1);
			else mu[i] = 0, phi[i] = phi[i / lpf[i]] * lpf[i];
			for(int j = 0; j < int(prime.size()) && prime[j] <= lpf[i] && prime[j] * i <= SZ; ++ j) lpf[prime[j] * i] = prime[j];
		}
	}
	int mu_large(long long x){ // O(sqrt(x))
		int res = 1;
		for(long long i = 2; i * i <= x; ++ i) if(x % i == 0){
			if(x / i % i) return 0;
			x /= i, res = -res;
		}
		if(x > 1) res = -res;
		return res;
	}
	long long phi_large(long long x){ // O(sqrt(x))
		long long res = x;
		for(long long i = 2; i * i <= x; ++ i) if(x % i == 0){
			while(x % i == 0) x /= i;
			res -= res / i;
		}
		if(x > 1) res -= res / x;
		return res;
	}
	template<typename IT> // O(n log n)
	auto convolute(IT begin0, IT end0, IT begin1, IT end1){
		int n = distance(begin0, end0);
		assert(n == distance(begin1, end1));
		vector<typename iterator_traits<IT>::value_type> res(n);
		for(int x = 1; x < n; ++ x) for(int y = 1; x * y < n; ++ y) res[x * y] += *(begin0 + x) * *(begin1 + y);
		return res;
	}
	template<typename IT> // O(n log n log k)
	auto conv_exp(IT begin, IT end, long long e){
		int n = end - begin;
		vector<typename iterator_traits<IT>::value_type> res(n), p(begin, end);
		res[1] = 1;
		for(; e; e >>= 1, p = convolute(p.begin(), p.end(), p.begin(), p.end())) if(e & 1) res = convolute(res.begin(), res.end(), p.begin(), p.end());
		return res;
	}
	template<typename IT> // O(n log n)
	void mobius_transform(IT begin, IT end){
		int n = end - begin;
		vector<typename iterator_traits<IT>::value_type> res(n);
		for(int x = 1; x < n; ++ x) for(int mx = x; mx < n; mx += x) res[mx] += *(begin + x);
		move(res.begin(), res.end(), begin);
	}
	template<typename IT> // O(n log n)
	void inverse_transform(IT begin, IT end){
		int n = end - begin;
		vector<typename iterator_traits<IT>::value_type> res(n);
		for(int x = 1; x < n; ++ x) for(int y = 1; x * y < n; ++ y) res[x * y] += *(begin + x) * mu[y];
		move(res.begin(), res.end(), begin);
	}
	vector<int> mul_cnt;
	bool mul_cnt_ready = false;
	template<typename IT> // O(SZ log SZ)
	void init_mul_cnt(IT begin, IT end){
		mul_cnt_ready = true;
		vector<int> cnt(SZ + 1);
		mul_cnt.assign(SZ + 1, 0);
		for(; begin != end; ++ begin) ++ cnt[*begin];
		for(int x = 1; x <= SZ; ++ x) for(int mx = x; mx <= SZ; mx += x) mul_cnt[x] += cnt[mx];
	}
	template<typename T> // Requires Z_p, O((SZ / g) log k)
	T count_tuples_with_gcd(int k, int g = 1){
		assert(mul_cnt_ready);
		T res = 0;
		for(int x = 1; x <= SZ / g; ++ x) res += mu[x] * (T(mul_cnt[x * g]) ^ k);
		return res;
	}
};

// 156485479_1_4
// Combinatorics
// O(N) preprocessing
// Requires Z_p
template<int SZ, typename T = Zp>
struct combinatorics{
	vector<T> inv, fact, invfact;
	vector<vector<T>> stir1, stir2;
	combinatorics(): inv(SZ << 1 | 1, 1), fact(SZ << 1 | 1, 1), invfact(SZ << 1 | 1, 1){
		for(int i = 1; i <= SZ << 1; ++ i) fact[i] = fact[i - 1] * i;
		invfact[SZ << 1] = 1 / fact[SZ << 1];
		for(int i = (SZ << 1) - 1; i >= 0; -- i){
			invfact[i] = invfact[i + 1] * (i + 1);
			inv[i + 1] = invfact[i + 1] * fact[i];
		}
	}
	T C(int n, int k){ return n < k ? 0 : fact[n] * invfact[k] * invfact[n - k]; }
	T P(int n, int k){ return n < k ? 0 : fact[n] * invfact[n - k]; }
	T H(int n, int k){ return C(n + k - 1, k); }
	vector<T> precalc_power(int base, int n = SZ << 1){
		vector<T> res(n + 1, 1);
		for(int i = 1; i <= n; ++ i) res[i] = res[i - 1] * base;
		return res;
	}
	T naive_C(long long n, long long k){
		if(n < k) return 0;
		T res = 1;
		k = min(k, n - k);
		for(long long i = n; i > n - k; -- i) res *= i;
		return res * invfact[k];
	}
	T naive_P(long long n, int k){
		if(n < k) return 0;
		T res = 1;
		for(long long i = n; i > n - k; -- i) res *= i;
		return res;
	}
	T naive_H(long long n, long long k){ return naive_C(n + k - 1, k); }
	bool parity_C(long long n, long long k){ return n < k ? 0 : k & n - k ^ 1; }
	// Catalan's Trapzoids
	// # of bitstrings of n Xs and k Ys such that in each initial segment, (# of X) + m > (# of Y) 
	T Cat(int n, int k, int m = 1){
		if(m <= 0) return 0;
		else if(k >= 0 && k < m) return C(n + k, k);
		else if(k < n + m) return C(n + k, k) - C(n + k, k - m);
		else return 0;
	}
	// Stirling number
	// First kind (unsigned): # of n-permutations with k disjoint cycles
	//                        Also the coefficient of x^k for x_n = x(x+1)...(x+n-1)
	// Second kind: # of ways to partition a set of size n into r non-empty sets
	//              Satisfies sum{k=0~n}(x_k) = x^n
	array<bool, 2> pre{};
	template<bool FIRST = true>
	void precalc_stir(int n, int k){
		auto &s = FIRST ? stir1 : stir2;
		pre[!FIRST] = true;
		s.resize(n + 1, vector<T>(k + 1, 1));
		for(int i = 1; i <= n; ++ i) for(int j = 1; j <= k; ++ j){
			s[i][j] = (FIRST ? i - 1 : j) * s[i - 1][j] + s[i - 1][j - 1];
		}
	}
	// unsigned
	T Stir1(int n, int k){
		if(n < k) return 0;
		assert(pre[0]);
		return stir1[n][k];
	}
	T Stir2(long long n, int k){
		if(n < k) return 0;
		if(pre[1] && n < int(stir2.size())) return stir2[n][k];
		T res = 0;
		for(int i = 0, sign = 1; i <= k; ++ i, sign *= -1) res += sign * C(k, i) * (Zp(k - i) ^ n);
		return res * invfact[k];
	}
	bool parity_Stir2(long long n, long long k){ return n < k ? 0 : k ? !(n - k & k - 1 >> 1) : 0; }
};

// 156485479_1_5
// Millar Rabin Primality Test / Pollard Rho Algorithm
// 7 times slower than a^b mod m / O(n^{1/4}) gcd calls
typedef unsigned long long ull;
typedef long double ld;
ull mod_mul(ull a, ull b, ull M) {
	long long res = a * b - M * ull(ld(a) * ld(b) / ld(M));
	return res + M * (res < 0) - M * (res >= (long long)M);
}
ull mod_pow(ull b, ull e, ull mod) {
	ull res = 1;
	for (; e; b = mod_mul(b, b, mod), e /= 2) if (e & 1) res = mod_mul(res, b, mod);
	return res;
}
bool isprime(ull n){
	if(n < 2 || n % 6 % 4 != 1) return n - 2 < 2;
	vector<ull> A{2, 325, 9375, 28178, 450775, 9780504, 1795265022};
	ull s = __builtin_ctzll(n - 1), d = n >> s;
	for(auto &a: A){
		ull p = mod_pow(a, d, n), i = s;
		while(p != 1 && p != n - 1 && a % n && i --) p = mod_mul(p, p, n);
		if(p != n - 1 && i != s) return 0;
	}
	return 1;
}
ull pfactor(ull n){
	auto f = [n](ull x){ return (mod_mul(x, x, n) + 1) % n; };
	if(!(n & 1)) return 2;
	for(ull i = 2; ; ++ i){
		ull x = i, y = f(x), p;
		while((p = gcd(n + y - x, n)) == 1) x = f(x), y = f(f(y));
		if(p != n) return p;
	}
}
vector<ull> factorize(ull n){
	if(n == 1) return {};
	if(isprime(n)) return {n};
	ull x = pfactor(n);
	auto l = factorize(x), r = factorize(n / x);
	l.insert(l.end(), r.begin(), r.end());
	return l;
}

// 156485479_1_6
// Tonelli Shanks Algorithm ( Solution to x^2 = a mod p )
// O(log^2 p)
long long modexp(long long b, long long e, const long long &mod){
	long long res = 1;
	for(; e; b = b * b % mod, e >>= 1) if(e & 1) res = res * b % mod;
	return res;
}
long long sqrt(long long a, long long p){
	a %= p;
	if(a < 0) a += p;
	if(a == 0) return 0;
	assert(modexp(a, (p - 1) / 2, p) == 1);
	if(p % 4 == 3) return modexp(a, (p+1)/4, p);
	// a^(n+3)/8 or 2^(n+3)/8 * 2^(n-1)/4 works if p % 8 == 5
	long long s = p - 1, n = 2;
	int r = 0, m;
	while(s % 2 == 0) ++ r, s /= 2;
	/// find a non-square mod p
	while(modexp(n, (p - 1) / 2, p) != p - 1) ++ n;
	long long x = modexp(a, (s + 1) / 2, p);
	long long b = modexp(a, s, p), g = modexp(n, s, p);
	for(;; r = m){
		long long t = b;
		for(m = 0; m < r && t != 1; ++ m) t = t * t % p;
		if(m == 0) return x;
		long long gs = modexp(g, 1LL << (r - m - 1), p);
		g = gs * gs % p;
		x = x * gs % p;
		b = b * g % p;
	}
}

// 156485479_1_7
// Chinese Remainder Theorem (Return a number x which satisfies x = a mod m & x = b mod n)
// All the values has to be less than 2^30
// O(log(m + n))
typedef long long ll;
ll euclid(ll x, ll y, ll &a, ll &b){
	if(y){
		ll d = euclid(y, x % y, b, a);
		return b -= x / y * a, d;
	}
	return a = 1, b = 0, x;
}
ll crt_coprime(ll a, ll m, ll b, ll n){
	ll x, y; euclid(m, n, x, y);
	ll res = a * (y + m) % m * n + b * (x + n) % n * m;
	if(res >= m * n) res -= m * n;
	return res;
}
ll crt(ll a, ll m, ll b, ll n){
	ll d = gcd(m, n);
	if(((b -= a) %= n) < 0) b += n;
	if(b % d) return -1; // No solution
	return d * crt_coprime(0LL, m/d, b/d, n/d) + a;
}

// 156485479_1_8
// Lehman Factorization / return a prime divisor of x
// x has to be equal or less than 10^14
// O(N^1/3)
long long primefactor(long long x){
	assert(x > 1);
	if(x <= 21){
		for(long long p = 2; p <= sqrt(x); ++ p) if(x % p == 0) return p;
		return x;
	}
	for(long long p = 2; p <= cbrt(x); ++ p) if(x % p == 0) return p;
	for(long long k = 1; k <= cbrt(x); ++ k){
		double t = 2 * sqrt(k * x);
		for(long long a = ceil(t); a <= floor(t + cbrt(sqrt(x)) / 4 / sqrt(k)); ++ a){
			long long b = a * a - 4 * k * x, s = sqrt(b);
			if(b == s * s) return gcd(a + s, x);
		}
	}
	return x;
}

// 156485479_1_9
// Polynomial Class
// Credit: cp-algorithms.com
// Requires Z_p and ntt
namespace algebra{
	const int inf = 1e9, magic = 250; // threshold for sizes to run the naive algo
	namespace fft{
		const int maxn = 1 << 18;
		typedef double ftype;
		typedef complex<ftype> point;
		point w[maxn];
		const ftype pi = acos(-1);
		bool initiated = false;
		void init(){
			if(!initiated){
				for(int i = 1; i < maxn; i <<= 1) for(int j = 0; j < i; ++ j) w[i + j] = polar(ftype(1), pi * j / i);
				initiated = true;
			}
		}
		template<typename T>
		void fft(T *in, point *out, int n, int k = 1){
			if(n == 1) *out = *in;
			else{
				n >>= 1;
				fft(in, out, n, 2 * k);
				fft(in + k, out + n, n, 2 * k);
				for(int i = 0; i < n; ++ i){
					auto t = out[i + n] * w[i + n];
					out[i + n] = out[i] - t;
					out[i] += t;
				}
			}
		}
		template<typename T>
		void mul_slow(vector<T> &a, const vector<T> &b){
			vector<T> res(int(a.size() + b.size()) - 1);
			for(size_t i = 0; i < a.size(); ++ i) for(size_t j = 0; j < int(b.size()); ++ j) res[i + j] += a[i] * b[j];
			a = res;
		}
		template<typename T>
		void mul(vector<T> &a, const vector<T> &b){
			if(int(min(a.size(), b.size())) < magic){
				mul_slow(a, b);
				return;
			}
			///* // For Z_p
			a = a * b;
			//*/
			/* // for real coefficients
			init();
			static const int shift = 15, mask = (1 << shift) - 1;
			static point A[maxn], B[maxn];
			static point C[maxn], D[maxn];
			for(size_t i = 0; i < n; ++ i){
				A[i] = point(a[i] & mask, a[i] >> shift);
				if(i < b.size()) B[i] = point(b[i] & mask, b[i] >> shift);
				else B[i] = 0;
			}
			fft(A, C, n); fft(B, D, n);
			for(size_t i = 0; i < n; ++ i){
				point c0 = C[i] + conj(C[(n - i) % n]);
				point c1 = C[i] - conj(C[(n - i) % n]);
				point d0 = D[i] + conj(D[(n - i) % n]);
				point d1 = D[i] - conj(D[(n - i) % n]);
				A[i] = c0 * d0 - point(0, 1) * c1 * d1;
				B[i] = c0 * d1 + d0 * c1;
			}
			fft(A, C, n); fft(B, D, n);
			reverse(C + 1, C + n), reverse(D + 1, D + n);
			int t = 4 * n;
			for(size_t i = 0; i < n; ++ i){
				long long A0 = llround(real(C[i]) / t);
				T A1 = llround(imag(D[i]) / t);
				T A2 = llround(imag(C[i]) / t);
				a[i] = A0 + (A1 << shift) + (A2 << 2 * shift);
			}
			*/
		}
	}
	template<typename T>
	T bpow(T x, size_t n){
		return x ^ n;
		// return n ? n & 1 ? x * bpow(x, n - 1) : bpow(x * x, n >> 1) : T(1);
	}
	template<typename T>
	struct polynomial{
		vector<T> a;
		void normalize(){ // get rid of leading zeroes
			while(!a.empty() && a.back() == T(0)) a.pop_back();
		}
		polynomial(){}
		polynomial(T a0): a{a0}{ normalize(); }
		polynomial(const vector<T> &t) :a(t){ normalize(); }
		polynomial &operator=(const polynomial &t){
			a = t.a;
			return *this;
		}
		polynomial &operator+=(const polynomial &t){
			a.resize(max(a.size(), t.a.size()));
			for(size_t i = 0; i < t.a.size(); ++ i) a[i] += t.a[i];
			normalize();
			return *this;
		}
		polynomial &operator-=(const polynomial &t){
			a.resize(max(a.size(), t.a.size()));
			for(size_t i = 0; i < t.a.size(); ++ i) a[i] -= t.a[i];
			normalize();
			return *this;
		}
		polynomial operator+(const polynomial &t) const{ return polynomial(*this) += t; }
		polynomial operator-(const polynomial &t) const{ return polynomial(*this) -= t; }
		polynomial mod_xk(size_t k) const{ // get same polynomialnomial mod x^k
			return vector<T>(begin(a), begin(a) + min(k, a.size()));
		}
		polynomial mul_xk(size_t k) const{ // multiply by x^k
			polynomial res(*this);
			res.a.insert(begin(res.a), k, 0);
			return res;
		}
		polynomial div_xk(size_t k) const{ // divide by x^k, dropping coefficients
			return vector<T>(begin(a) + min(k, a.size()), end(a));
		}
		polynomial substr(size_t l, size_t r) const{ // return mod_xk(r).div_xk(l)
			return vector<T>(begin(a) + min(l, a.size()), begin(a) + min(r, a.size()));
		}
		polynomial inv(size_t n) const{ // get inverse series mod x^n
			assert(!is_zero());
			polynomial ans = 1 / a[0];
			for(int i = 1; i < n; i <<= 1) ans = (ans * 2 - ans * ans * mod_xk(i << 1)).mod_xk(i << 1);
			return ans.mod_xk(n);
		}
		polynomial operator*=(const polynomial &t){ fft::mul(a, t.a); normalize(); return *this; }
		polynomial operator*(const polynomial &t) const{return polynomial(*this) *= t; }
		polynomial reverse(size_t n, bool rev = 0) const{ // reverses and leaves only n terms
			polynomial res(*this);
			if(rev){ // If rev = 1 then tail goes to head
				res.a.resize(max(n, res.a.size()));
			}
			std::reverse(res.a.begin(), res.a.end());
			return res.mod_xk(n);
		}
		pair<polynomial, polynomial> divmod_slow(const polynomial &b) const{ // when divisor or quotient is small
			vector<T> A(a), res;
			for(T invx = 1 / b.a.back(); A.size() >= b.a.size(); ){
				res.push_back(A.back() * invx);
				if(res.back() != T(0)) for(size_t i = 0; i < b.a.size(); ++ i) A[A.size() - i - 1] -= res.back() * b.a[b.a.size() - i - 1];
				A.pop_back();
			}
			std::reverse(begin(res), end(res));
			return {res, A};
		}
		pair<polynomial, polynomial> divmod(const polynomial &b) const{ // returns quotiend and remainder of a mod b
			if(deg() < b.deg()) return {polynomial{0}, *this};
			int d = deg() - b.deg();
			if(min(d, b.deg()) < magic) return divmod_slow(b);
			polynomial D = (reverse(d + 1) * b.reverse(d + 1).inv(d + 1)).mod_xk(d + 1).reverse(d + 1, 1);
			return {D, *this - D * b};
		}
		polynomial operator/(const polynomial &t) const{ return divmod(t).first; }
		polynomial operator%(const polynomial &t) const{ return divmod(t).second; }
		polynomial &operator/=(const polynomial &t){ return *this = divmod(t).first; }
		polynomial &operator%=(const polynomial &t){ return *this = divmod(t).second; }
		polynomial &operator*=(const T &x){
			for(auto &it: a) it *= x;
			normalize();
			return *this;
		}
		polynomial &operator/=(const T &x){
			T invx = 1 / x;
			for(auto &it: a) it *= invx;
			normalize();
			return *this;
		}
		polynomial operator*(const T &x) const{ return polynomial(*this) *= x; }
		polynomial operator/(const T &x) const{ return polynomial(*this) /= x; }
		void print() const{
			for(auto it: a) cout << it << ' ';
			cout << endl;
		}
		T eval(T x) const{ // evaluates in single point x
			T res(0);
			for(int i = int(a.size()) - 1; i >= 0; -- i) res = res * x + a[i];
			return res;
		}
		T &lead(){ // leading coefficient
			return a.back();
		}
		int deg() const{ // degree
			return a.empty() ? -inf : int(a.size()) - 1;
		}
		bool is_zero() const{ // is polynomialnomial zero
			return a.empty();
		}
		T coef(int idx) const{
			return idx >= int(a.size()) || idx < 0 ? T(0) : a[idx];
		}
		T &operator[](int idx){ // mutable reference at idx
			return a[idx];
		}
		bool operator==(const polynomial &t) const{ return a == t.a; }
		bool operator!=(const polynomial &t) const{ return a != t.a; }
		polynomial derivative() const{ // calculate derivative
			static vector<T> res; res.clear();
			for(int i = 1; i <= deg(); ++ i) res.push_back(i * a[i]);
			return res;
		}
		polynomial antiderivative() const{ // calculate integral with C = 0
			static vector<T> res; res.assign(1, 0);
			for(int i = 0; i <= deg(); ++ i) res.push_back(a[i] / (i + 1));
			return res;
		}
		size_t leading_xk() const{ // Let p(x) = x^k * t(x), return k
			if(is_zero()) return inf;
			int res = 0;
			while(a[res] == T(0)) ++ res;
			return res;
		}
		polynomial log(size_t n){ // calculate log p(x) mod x^n
			assert(a[0] == T(1));
			return (derivative().mod_xk(n) * inv(n)).antiderivative().mod_xk(n);
		}
		polynomial exp(size_t n){ // calculate exp p(x) mod x^n
			if(is_zero()) return T(1);
			assert(a[0] == T(0));
			polynomial ans = 1;
			for(size_t a = 1; a < n; a <<= 1){
				polynomial C = ans.log(a << 1).div_xk(a) - substr(a, a << 1);
				ans -= (ans * C).mod_xk(a).mul_xk(a);
			}
			return ans.mod_xk(n);
		}
		polynomial pow_slow(size_t k, size_t n){ // if k is small
			return k ? k % 2 ? (*this * pow_slow(k - 1, n)).mod_xk(n) : (*this * *this).mod_xk(n).pow_slow(k / 2, n) : T(1);
		}
		polynomial pow(size_t k, size_t n){ // calculate p^k(n) mod x^n
			if(is_zero())return *this;
			if(k < magic) return pow_slow(k, n);
			int i = leading_xk();
			T j = a[i];
			polynomial t = div_xk(i) / j;
			return bpow(j, k) * (t.log(n) * T(k)).exp(n).mul_xk(i * k).mod_xk(n);
		}
		polynomial mulx(T x){ // component-wise multiplication with x^k
			T cur = 1;
			polynomial res(*this);
			for(int i = 0; i <= deg(); ++ i) res.coef(i) *= cur, cur *= x;
			return res;
		}
		polynomial mulx_sq(T x){ // component-wise multiplication with x^{k^2}
			T cur = x, total = 1, xx = x * x;
			polynomial res(*this);
			for(int i = 0; i <= deg(); ++ i) res.coef(i) *= total, total *= cur, cur *= xx;
			return res;
		}
		vector<T> chirpz_even(T z, int n){ // P(1), P(z^2), P(z^4), ..., P(z^2(n-1))
			int m = deg();
			if(is_zero()) return vector<T>(n, 0);
			vector<T> vv(m + n);
			T zi = 1 / z, zz = zi * zi, cur = zi, total = 1;
			for(int i = 0; i <= max(n - 1, m); ++ i){
				if(i <= m){ vv[m - i] = total; }
				if(i < n){ vv[m + i] = total; }
				total *= cur, cur *= zz;
			}
			polynomial w = (mulx_sq(z) * vv).substr(m, m + n).mulx_sq(z);
			vector<T> res(n);
			for(int i = 0; i < n; ++ i) res[i] = w[i];
			return res;
		}
		vector<T> chirpz(T z, int n){ // P(1), P(z), P(z^2), ..., P(z^(n-1))
			auto even = chirpz_even(z, n + 1 >> 1);
			auto odd = mulx(z).chirpz_even(z, n >> 1);
			vector<T> ans(n);
			for(int i = 0; i < n >> 1; ++ i) ans[i << 1] = even[i], ans[i << 1 | 1] = odd[i];
			if(n & 1) ans[n - 1] = even.back();
			return ans;
		}
		template<typename iter>
		vector<T> eval(vector<polynomial> &tree, int v, iter l, iter r){ // auxiliary evaluation function
			if(r - l == 1) return {eval(*l)};
			else{
				auto m = l + (r - l >> 1);
				auto A = (*this % tree[v << 1]).eval(tree, v << 1, l, m);
				auto B = (*this % tree[v << 1 | 1]).eval(tree, v << 1 | 1, m, r);
				A.insert(end(A), begin(B), end(B));
				return A;
			}
		}
		template<typename iter>
		vector<T> eval(iter begin, iter end){ // evaluate polynomialnomial in (x1, ..., xn)
			int n = end - begin;
			if(is_zero()) return vector<T>(n, T(0));
			vector<polynomial> tree(n << 2);
			build(tree, 1, begin, end);
			return eval(tree, 1, begin, end);
		}
		template<typename iter>
		polynomial inter(const vector<polynomial> &tree, int v, iter l, iter r, iter ly, iter ry){ // auxiliary interpolation function
			if(r - l == 1) return {*ly / a[0]};
			else{
				auto m = l + (r - l >> 1);
				auto my = ly + (ry - ly >> 1);
				auto A = (*this % tree[v << 1]).inter(tree, v << 1, l, m, ly, my);
				auto B = (*this % tree[v << 1 | 1]).inter(tree, v << 1 | 1, m, r, my, ry);
				return A * tree[v << 1 | 1] + B * tree[v << 1];
			}
		}
	};
	template<typename T>
	polynomial<T> operator*(const T& a, const polynomial<T>& b){
		auto res(b);
		for(auto &it: res.a) it = a * it;
		res.normalize();
		return res;
	}
	template<typename T>
	polynomial<T> xk(int k){ // return x^k
		return polynomial<T>{1}.mul_xk(k);
	}
	template<typename T>
	T resultant(polynomial<T> a, polynomial<T> b){ // computes resultant of a and b
		if(b.is_zero()) return 0;
		else if(b.deg() == 0) return bpow(b.lead(), a.deg());
		else{
			int pw = a.deg();
			a %= b;
			pw -= a.deg();
			T mul = bpow(b.lead(), pw) * T((b.deg() & a.deg() & 1) ? -1 : 1);
			T ans = resultant(b, a);
			return ans * mul;
		}
	}
	template<typename iter>
	polynomial<typename iter::value_type> generate(iter L, iter R){ // computes (x-a1)(x-a2)...(x-an) without building tree
		if(R - L == 1) return vector<typename iter::value_type>{-*L, 1};
		else{
			iter M = L + (R - L >> 1);
			return generate(L, M) * generate(M, R);
		}
	}
	template<typename T, typename iter>
	polynomial<T> &build(vector<polynomial<T>> &res, int v, iter L, iter R){ // builds evaluation tree for (x-a1)(x-a2)...(x-an)
		if(R - L == 1) return res[v] = vector<T>{-*L, 1};
		else{
			iter M = L + (R - L >> 1);
			return res[v] = build(res, v << 1, L, M) * build(res, v << 1 | 1, M, R);
		}
	}
	template<typename T>
	polynomial<T> inter(const vector<T> &x, const vector<T> &y){ // interpolates minimum polynomialnomial from (xi, yi) pairs
		int n = x.size();
		vector<polynomial<T>> tree(n << 2);
		return build(tree, 1, begin(x), end(x)).derivative().inter(tree, 1, begin(x), end(x), begin(y), end(y));
	}
};
using namespace algebra;
using poly = polynomial<Zp>;

// 156485479_1_10
// Discrete Log
// O(sqrt(mod) log(mod))
// Return the minimum x > 0 with a^x = b mod m, -1 if no such x
// Credit: KACTL
long long discrete_log(long long a, long long b, long long m){
	long long n = (long long) sqrt(m) + 1, e = 1, f = 1, j = 1;
	map<long long, long long> A;
	while(j <= n && (e = f = e * a % m) != b % m) A[e * b % m] = j ++;
	if(e == b % m) return j;
	if(__gcd(m, e) == __gcd(m, b)) for(int i = 2; i < n + 2; ++ i) if (A.count(e = e * f % m)) return n * i - A[e];
	return -1;
}

// 156485479_1_11
// Continued Fraction
typedef array<long long, 2> frac;
struct continued_fraction{
	vector<long long> a;
	// Fraction must either be of form p/q where p is an integer and q is a positive integer
	// or p a non-zero integer and q = 0 ( this represents inf / -inf )
	continued_fraction(frac x){
		while(x[1]){
			a.push_back(x[0] / x[1]);
			x = {x[1], x[0] % x[1]};
			if(x[1] < 0) x[1] += x[0], -- a.back();
		}
		if(a.empty()) a.push_back(x[0] > 0 ? 1e9 : -1e9);
	}
	continued_fraction(vector<long long> a): a(move(a)){ }
	void alter(){ int(a.size()) > 1 && a.back() == 1 ? a.pop_back(), ++ a.back() : (a.push_back(1), -- *(next(a.rbegin()))); }
	frac convergent(int len){
		frac res{1, 0};
		for(int i = min(len, int(a.size())) - 1; i >= 0; -- i) res = {res[1] + res[0] * a[i], res[0]};
		return res;
	}
};
bool frac_cmp(frac x, frac y){ return x[0] * y[1] < x[1] * y[0]; }
// assumes 0 <= x < y
// returns a fraction p/q with minimal p ( or equivalently, q ) within range (x, y)
frac best_rational_within(frac low, frac high){
	continued_fraction clow(low), chigh(high);
	for(int ix = 0; ix < 2; ++ ix, clow.alter()) for(int iy = 0; iy < 2; ++ iy, chigh.alter()){
		vector<long long> t;
		clow.a.push_back(numeric_limits<long long>::max()), chigh.a.push_back(numeric_limits<long long>::max());
		for(int i = 0; ; ++ i){
			if(clow.a[i] == chigh.a[i]) t.push_back(clow.a[i]);
			else{
				t.push_back(min(clow.a[i], chigh.a[i]) + 1);
				break;
			}
		}
		clow.a.pop_back(), chigh.a.pop_back();
		continued_fraction frac_t(t);
		auto c = frac_t.convergent(int(t.size()));
		if(frac_cmp(low, c) && frac_cmp(c, high)) return c;
	}
}

// 156485479_1_12
// Meissel–Lehmer Algorithm
// Fast Calculation of Prime Counting Fucntion, or sum of F(p) where F is a multiplicative function
// O(n^(2/3 + eps)) time complexity and O(n^(1/3 + eps)) space complexity for all eps > 0. (Correct me if I'm wrong)
// Credit: chemthan
// Requires number_theory
template<typename NT, typename T = long long>
struct meissel_lehmer{
	const int maxx = 1e2 + 5, maxy = 1e5 + 5, maxn = 1e7 + 5;
	vector<int> cn;
	vector<T> sum;
	vector<vector<T>> f;
	NT &nt;
	T F(long long x){
		return 1;
	}
	T sum_F(long long x){
		return x;
	}
	meissel_lehmer(NT &nt): nt(nt), sum(maxn), f(maxx, vector<T>(maxy)), cn(maxn){
		for(int i = 2, cnt = 0; i < maxn; ++ i){
			sum[i] = sum[i - 1];
			if(nt.lpf[i] == i) sum[i] += F(i), ++ cnt;
			cn[i] = cnt;
		}
		for(int i = 0; i < maxx; ++ i) for(int j = 0; j < maxy; ++ j){
			f[i][j] = i ? f[i - 1][j] - f[i - 1][j / nt.prime[i - 1]] * F(nt.prime[i - 1]) : sum_F(j);
		}
	}
	T legendre_sum(long long m, int n){
		if(!n) return sum_F(m);
		if(m <= nt.prime[n - 1]) return F(1);
		if(m < maxy && n < maxx) return f[n][m];
		return legendre_sum(m, n - 1) - legendre_sum(m / nt.prime[n - 1], n - 1) * F(nt.prime[n - 1]);
	}
	T pi(long long m){
		if(m <= maxn) return sum[m];
		int x = sqrt(m + 0.9), y = cbrt(m + 0.9), a = cn[y];
		T res = legendre_sum(m, a) - F(1) + sum[y];
		for(int i = a; nt.prime[i] <= x; ++ i) res -= (pi(m / nt.prime[i]) - pi(nt.prime[i] - 1)) * F(nt.prime[i]);
		return res;
	}
};
// Fast Computaion of pi(N)
// Credit: https://judge.yosupo.jp/submission/12916
long long pi(const long long N){
	if(N <= 1) return 0;
	if(N == 2) return 1;
	const int v = sqrtl(N);
	int s = (v + 1) / 2;
	vector<int> smalls(s), roughs(s);
	vector<long long> larges(s);
	for(int i = 0; i < s; ++ i) smalls[i] = i, roughs[i] = 2 * i + 1, larges[i] = (N / (2 * i + 1) - 1) / 2;
	vector<bool> skip(v + 1);
	const auto divide = [](long long n, long long d){ return int(n / d); };
	const auto half = [](int n){ return n - 1 >> 1; };
	int pc = 0;
	for(int p = 3; p <= v; p += 2) if(!skip[p]){
		int q = p * p;
		if((long long)(q) * q > N) break;
		skip[p] = true;
		for(int i = q; i <= v; i += 2 * p) skip[i] = true;
		int ns = 0;
		for(int k = 0; k < s; ++ k){
			int i = roughs[k];
			if(skip[i]) continue;
			long long d = (long long)(i) * p;
			larges[ns] = larges[k] - (d <= v ? larges[smalls[d >> 1] - pc] : smalls[half(divide(N, d))]) + pc;
			roughs[ns ++] = i;
		}
		s = ns;
		for(int i = half(v), j = v / p - 1 | 1; j >= p; j -= 2){
			int c = smalls[j >> 1] - pc;
			for(int e = j * p >> 1; i >= e; --i) smalls[i] -= c;
		}
		++ pc;
	}
	larges[0] += (long long)(s + 2 * (pc - 1)) * (s - 1) / 2;
	for(int k = 1; k < s; ++ k) larges[0] -= larges[k];
	for(int l = 1; l < s; ++ l){
		int q = roughs[l];
		long long M = N / q, t = 0;
		int e = smalls[half(M / q)] - pc;
		if(e < l + 1) break;
		for(int k = l + 1; k <= e; ++ k) t += smalls[half(divide(M, roughs[k]))];
		larges[0] += t - (long long)(e - l) * (pc + l - 1);
	}
	return larges[0] + 1;
}

// 156485479_1_13
// Xudyh's Sieve
// Calculate the prefix sum of a multiplicative function fast
// Assuming there exists an arithemetic function g such that pref_g and pref_f*g is easy to evaluate
// O(n^2/3)
template<typename F1, typename F2, typename F3, typename T = Zp>
struct prefix_sum{
	long long th; // threshold, ideally about (2(single query) ~ 5(lots of queries)) * MAXN^2/3
	F1 pf;
	F2 pg;
	F3 pfg;
	unordered_map<long long, T> mp;
	prefix_sum(long long th, F1 pf, F2 pg, F3 pfg): th(th), pf(pf), pg(pg), pfg(pfg){ }
	T query(long long n){
		if(n <= th) return pf(n);
		if(mp.count(n)) return mp[n];
		T res = pfg(n);
		for(long long low = 2, high = 2; low <= n; low = high + 1){
			high = n / (n / low);
			res -= (pg(high) - pg(low - 1)) * query(n / low);
		}
		return mp[n] = res / pg(1);
	}
};

// 156485479_1_14
// Formulae Regarding Mobius Inversion
// Let S_k(n) := Sum{1<=i<=n}( i^k ), id_k(n) := n^k
// Sum{1<=x_i<=n_i for each 1<=i<=k}( Product{1<=i<=k}( x_i^e_i ) gcd{1<=i<=k}( x_i ) )
// = Sum{1<=l<=inf}( Product{1<=i<=k}( l^e_i S_{e_i}( Floor( n_i / l ) ) ) (Mu * id_k)( l ) )

// 156485479_2_1
// Linear Recurrence Relation Solver / Berlekamp - Massey Algorithm
// O(N^2 log N) / O(N^2)
long long modexp(long long b, long long e, const long long &mod){
	long long res = 1;
	for(; e; b = b * b % mod, e >>= 1) if(e & 1) res = res * b % mod;
	return res;
}
long long modinv(long long a, const long long &mod){
	return modexp(a, mod - 2, mod);
}
struct recurrence{
	int N;
	vector<long long> init, coef;
	const long long mod;
	recurrence(const vector<long long> &init, const vector<long long> &coef, long long mod): N(coef.size()), init(init), coef(coef), mod(mod){ }
	// Berlekamp Massey Algorithm
	recurrence(const vector<long long> &s, long long mod): mod(mod){
		int n = int(s.size());
		N = 0;
		vector<long long> B(n), T;
		coef.resize(n);
		coef[0] = B[0] = 1;
		long long b = 1;
		for(int i = 0, m = 0; i < n; ++ i){
			++ m;
			long long d = s[i] % mod;
			for(int j = 1; j <= N; ++ j) d = (d + coef[j] * s[i - j]) % mod;
			if(!d) continue;
			T = coef;
			long long c = d * modinv(b, mod) % mod;
			for(int j = m; j < n; ++ j) coef[j] = (coef[j] - c * B[j - m]) % mod;
			if(2 * N > i) continue;
			N = i + 1 - N, B = T, b = d, m = 0;
		}
		coef.resize(N + 1), coef.erase(coef.begin());
		for(auto &x: coef) x = (mod - x) % mod;
		reverse(coef.begin(), coef.end());
		init.resize(N);
		for(int i = 0; i < N; ++ i) init[i] = s[i] % mod;
	}
	long long operator[](long long n) const{
		auto combine = [&](vector<long long> a, vector<long long> b){
			vector<long long> res(2 * N + 1);
			for(int i = 0; i <= N; ++ i) for(int j = 0; j <= N; ++ j) res[i + j] = (res[i + j] + a[i] * b[j]) % mod;
			for(int i = N << 1; i > N; -- i) for(int j = 0; j < N; ++ j) res[i - 1 - j] = (res[i - 1 - j] + res[i] * coef[N - 1 - j]) % mod;
			res.resize(N + 1);
			return res;
		};
		vector<long long> pol(N + 1), e(pol);
		pol[0] = e[1] = 1;
		for(++ n; n; n >>= 1, e = combine(e, e)) if(n & 1) pol = combine(pol, e);
		long long res = 0;
		for(int i = 0; i < N; ++ i) res = (res + pol[i + 1] * init[i]) % mod;
		return res;
	}
};

// 156485479_2_2_1
// Find a solution of the system of linear equations. Return -1 if no sol, rank otherwise.
// O(n^2 m)
const double eps = 1e-12;
int solve_linear_equations(const vector<vector<double>> &AA, vector<double> &x, const vector<double> &bb){
	auto A = AA;
	auto b = bb;
	int n = int(A.size()), m = int(A[0].size()), rank = 0, br, bc;
	vector<int> col(m);
	iota(col.begin(), col.end(), 0);
	for(int i = 0; i < n; ++ i){
		double v, bv = 0;
		for(int r = i; r < n; ++ r) for(int c = i; c < m; ++ c) if((v = fabs(A[r][c])) > bv) br = r, bc = c, bv = v;
		if(bv <= eps){
			for(int j = i; j < n; ++ j) if(fabs(b[j]) > eps) return -1;
			break;
		}
		swap(A[i], A[br]), swap(b[i], b[br]), swap(col[i], col[bc]);
		for(int j = 0; j < n; ++ j) swap(A[j][i], A[j][bc]);
		bv = 1 / A[i][i];
		for(int j = i + 1; j < n; ++ j){
			double fac = A[j][i] * bv;
			b[j] -= fac * b[i];
			for(int k = i + 1; k < m; ++ k) A[j][k] -= fac * A[i][k];
		}
		++ rank;
	}
	x.resize(m);
	for(int i = rank; i --; ){
		b[i] /= A[i][i];
		x[col[i]] = b[i];
		for(int j = 0; j < i; ++ j) b[j] -= A[j][i] * b[i];
	}
	return rank;
}

// 156485479_2_2_3
// Find a solution of the system of linear equations in Z2. Return -1 if no sol, rank otherwise.
// O(n^2 m)
typedef bitset<1000> bs;
int solve_linear_equations(vector<bs> A, bs &x, vector<int> b, int m){
	int n = int(A.size()), rank = 0, br;
	vector<int> col(m);
	iota(col.begin(), col.end(), 0);
	for(int i = 0; i < n; ++ i){
		for(br = i; br < n; ++ br) if(A[br].any()) break;
		if(br == n){
			for(int j = i; j < n; ++ j) if(b[j]) return -1;
			break;
		}
		int bc = (int)A[br]._Find_next(i-1);
		swap(A[i], A[br]);
		swap(b[i], b[br]);
		swap(col[i], col[bc]);
		for(int j = 0; j < n; ++ j) if(A[j][i] != A[j][bc]) A[j].flip(i), A[j].flip(bc);
		for(int j = i + 1; j < n; ++ j) if(A[j][i]) b[j] ^= b[i], A[j] ^= A[i];
		++ rank;
	}
	x = bs();
	for(int i = rank; i --; ){
		if(!b[i]) continue;
		x[col[i]] = 1;
		for(int j = 0; j < i; ++ j) b[j] ^= A[j][i];
	}
	return rank;
}
// Dynamic size
int solve_linear_equations(vector<vector<int>> A, vector<int>& x, vector<int> b){
	int n = int(A.size()), m = int(A[0].size()), rank = 0, br;
	vector<int> col(m);
	iota(col.begin(), col.end(), 0);
	for(int i = 0; i < n; ++ i){
		for(br = i; br < n; ++ br) if(any_of(A[br].begin(), A[br].end(), [&](int x){ return x; })) break;
		if(br == n){
			for(int j = i; j < n; ++ j) if(b[j]) return -1;
			break;
		}
		int bc = i;
		for(; !A[br][bc]; ++ bc);
		swap(A[i], A[br]);
		swap(b[i], b[br]);
		swap(col[i], col[bc]);
		for(int j = 0; j < n; ++ j) if(A[j][i] != A[j][bc]){
			A[j][i] = !A[j][i];
			A[j][bc] = !A[j][bc];
		}
		for(int j = i + 1; j < n; ++ j) if(A[j][i]){
			b[j] ^= b[i];
			for(int k = 0; k < m; ++ k){
				A[j][k] ^= A[i][k];
			}
		}
		++ rank;
	}
	x = vector<int>(m);
	for(int i = rank; i --; ){
		if (!b[i]) continue;
		x[col[i]] = 1;
		for(int j = 0; j < i; ++ j) b[j] ^= A[j][i];
	}
	return rank;
}

// 156485479_2_3_1
// Matrix for Z_p
struct matrix: vector<vector<long long>>{
	int n, m;
	const long long mod;
	matrix(int n, int m, long long mod, bool is_id = false): n(n), m(m), mod(mod){
		resize(n, vector<long long>(m));
		if(is_id) for(int i = 0; i < min(n, m); ++ i) (*this)[i][i] = 1;
	}
	matrix(const vector<vector<long long>> &arr, long long mod): n(arr.size()), m(arr[0].size()), mod(mod){
		resize(n);
		for(int i = 0; i < n; ++ i) (*this)[i] = arr[i];
	}
	bool operator==(const matrix &otr) const{
		if(n != otr.n || m != otr.m) return false;
		for(int i = 0; i < n; ++ i) for(int j = 0; j < m; ++ j) if((*this)[i][j] != otr[i][j]) return false;
		return true;
	}
	matrix &operator=(const matrix &otr){
		n = otr.n, m = otr.m;
		resize(n);
		for(int i = 0; i < n; ++ i) (*this)[i] = otr[i];
		return *this;
	}
	matrix operator+(const matrix &otr) const{
		matrix res(n, m, mod);
		for(int i = 0; i < n; ++ i) for(int j = 0; j < m; ++ j) res[i][j] = ((*this)[i][j] + otr[i][j]) % mod;
		return res;
	}
	matrix &operator+=(const matrix &otr){
		return *this = *this + otr;
	}
	matrix operator*(const matrix &otr) const{
		assert(m == otr.n);
		int L = otr.m;
		matrix res(n, L, mod);
		for(int i = 0; i < n; ++ i) for(int j = 0; j < L; ++ j) for(int k = 0; k < m; ++ k) (res[i][j] += (*this)[i][k] * otr[k][j]) %= mod;
		return res;
	}
	matrix &operator*=(const matrix &otr){
		return *this = *this * otr;
	}
	matrix operator^(long long e) const{
		assert(n == m);
		matrix res(n, n, mod, 1), b(*this);
		for(; e; b *= b, e >>= 1) if(e & 1) res *= b;
		return res;
	}
	matrix &operator^=(const long long e){
		return *this = *this ^ e;
	}
	long long det() const{
		assert(n == m);
		vector<vector<long long>> temp = *this;
		long long res = 1;
		for(int i = 0; i < n; ++ i){
			for(int j = i + 1; j < n; ++ j){
				while(temp[j][i]){
					long long t = temp[i][i] / temp[j][i];
					if(t) for(int k = i; i < n; ++ k) temp[i][k] = (temp[i][k] - temp[j][k] * t) % mod;
					std::swap(temp[i], temp[j]);
					res *= -1;
				}
			}
			res = (res + mod) * temp[i][i] % mod;
			if(!res) return 0;
		}
		return res;
	}
};

// 156485479_2_3_2
// Matrix for general semiring
// T must support +, *, !=, <<, >>
template<typename T>
struct matrix: vector<vector<T>>{
	int n, m;
	const T add_id, mul_id;
	matrix(int n, int m, const T &add_id, const T &mul_id, bool is_id = false): n(n), m(m), add_id(add_id), mul_id(mul_id){
		this->resize(n, vector<T>(m, add_id));
		if(is_id) for(int i = 0; i < min(n, m); ++ i) (*this)[i][i] = mul_id;
	}
	matrix(const vector<vector<T>> &arr, const T &add_id, const T &mul_id): n(arr.size()), m(arr[0].size()), add_id(add_id), mul_id(mul_id){
		this->resize(n, vector<T>(m, add_id));
		for(int i = 0; i < n; ++ i) for(int j = 0; j < m; ++ j) (*this)[i][j] = arr[i][j];
	}
	bool operator==(const matrix &otr) const{
		if(n != otr.n || m != otr.m) return false;
		for(int i = 0; i < n; ++ i) for(int j = 0; j < m; ++ j) if((*this)[i][j] != otr[i][j]) return false;
		return true;
	}
	matrix &operator=(const matrix &otr){
		n = otr.n, m = otr.m;
		this->resize(n);
		for(int i = 0; i < n; ++ i) (*this)[i] = otr[i];
		return *this;
	}
	matrix operator+(const matrix &otr) const{
		matrix res(n, m, add_id, mul_id);
		for(int i = 0; i < n; ++ i) for(int j = 0; j < m; ++ j) res[i][j] = (*this)[i][j] + otr[i][j];
		return res;
	}
	matrix &operator+=(const matrix &otr){
		return *this = *this + otr;
	}
	matrix operator*(const matrix &otr) const{
		assert(m == otr.n);
		int L = otr.m;
		matrix res(n, L, add_id, mul_id);
		for(int i = 0; i < n; ++ i) for(int j = 0; j < L; ++ j) for(int k = 0; k < m; ++ k) res[i][j] = res[i][j] + (*this)[i][k] * otr[k][j];
		return res;
	}
	matrix &operator*=(const matrix &otr){
		return *this = *this * otr;
	}
	matrix operator^(long long e) const{
		assert(n == m);
		matrix res(n, n, add_id, mul_id, true), b(*this);
		for(; e; b *= b, e >>= 1) if(e & 1) res *= b;
		return res;
	}
	matrix &operator^=(const long long e){
		return *this = *this ^ e;
	}
};

// 156485479_2_3_3
// Matrix for a finite field of characteristic 2
template<int SZ>
struct matrix: vector<bitset<SZ>>{
	int n, m;
	matrix(int n, int m, bool is_id = false): n(n), m(m){
		this->resize(n);
		if(is_id) for(int i = 0; i < min(n, m); ++ i) (*this)[i].set(i);
	}
	template<typename mat>
	matrix(int n, int m, const mat &arr): n(n), m(m){
		this->resize(n);
		for(int i = 0; i < n; ++ i) for(int j = 0; j < m; ++ j) if(arr[i][j]) (*this)[i].set(j);
	}
	bool operator==(const matrix &otr) const{
		if(n != otr.n || m != otr.m) return false;
		for(int i = 0; i < n; ++ i) for(int j = 0; j < m; ++ j) if((*this)[i][j] != otr[i][j]) return false;
		return true;
	}
	matrix &operator=(const matrix &otr){
		n = otr.n, m = otr.m;
		this->resize(n);
		for(int i = 0; i < n; ++ i) (*this)[i] = otr[i];
		return *this;
	}
	matrix operator+(const matrix &otr) const{
		matrix res(n, m);
		for(int i = 0; i < n; ++ i) res[i] = (*this)[i] ^ otr[i];
		return res;
	}
	matrix &operator+=(const matrix &otr){
		return *this = *this + otr;
	}
	matrix operator*(const matrix &otr) const{
		assert(m == otr.n);
		int L = otr.m;
		matrix res(n, L);
		vector<bitset<SZ>> temp(L);
		for(int i = 0; i < L; ++ i) for(int j = 0; j < m; ++ j) temp[i][j] = otr[j][i];
		for(int i = 0; i < n; ++ i) for(int j = 0; j < L; ++ j) if(((*this)[i] & temp[j]).count() & 1) res[i].set(j);
		return res;
	}
	matrix &operator*=(const matrix &otr){
		return *this = *this * otr;
	}
	matrix operator^(long long e) const{
		assert(n == m);
		matrix res(n, n, true), b(*this);
		for(; e; b *= b, e >>= 1) if(e & 1) res *= b;
		return res;
	}
	matrix &operator^=(const long long e){
		return *this = *this ^ e;
	}
};

// 156485479_2_4_1_1
// Fast Fourier Transformation.
// Size must be a power of two.
// O(n log n)
typedef complex<double> cd;
const double PI = acos(-1);
template<typename IT>
void fft(IT begin, IT end, const bool invert = false){
	int n = end - begin;
	for(int i = 1, j = 0; i < n; ++ i){
		int bit = n >> 1;
		for(; j & bit; bit >>= 1) j ^= bit;
		j ^= bit;
		if(i < j) swap(*(begin + i), *(begin + j));
	}
	for(int len = 1; len < n; len <<= 1){
		double theta = PI / len * (invert ? -1 : 1);
		cd w(cos(theta), sin(theta));
		for(int i = 0; i < n; i += len << 1){
			cd wj(1);
			for(int j = 0; j < len; ++ j, wj *= w){
				cd u = *(begin + i + j), v = wj * *(begin + i + j + len);
				*(begin + i + j) = u + v, *(begin + i + j + len) = u - v;
			}
		}
	}
	if(invert) for(auto it = begin; it != end; ++ it) *it /= n;
}
template<typename Poly>
Poly operator*(const Poly &a, const Poly &b){
	vector<cd> f(a.begin(), a.end()), g(b.begin(), b.end());
	int n = max(int(a.size()) + int(b.size()) - 1, 1);
	if(__builtin_popcount(n) != 1) n = 1 << __lg(n) + 1;
	f.resize(n), g.resize(n);
	fft(f.begin(), f.end()), fft(g.begin(), g.end());
	for(int i = 0; i < n; ++ i) f[i] *= g[i];
	fft(f.begin(), f.end(), true);
	Poly res(n);
	for(int i = 0; i < n; ++ i) res[i] = round(f[i].real());
	while(!res.empty() && !res.back()) res.pop_back();
	return res;
}
template<typename Poly>
Poly operator+(const Poly &a, const Poly &b){
	Poly res(max(int(a.size()), int(b.size())));
	for(int i = 0; i < min(int(a.size()), int(b.size())); ++ i) res[i] = a[i] + b[i];
	for(int i = min(int(a.size()), int(b.size())); i < int(a.size()); ++ i) res[i] = a[i];
	for(int i = min(int(a.size()), int(b.size())); i < int(b.size()); ++ i) res[i] = b[i];	
	return res;
}

// 156485479_2_4_1_2
// Number Theoric Transformation
// Use (998244353: 15311432, 1 << 23, 469870224) or (7340033: 5, 1 << 20, 4404020)
// Size must be a power of two
// O(n log n)
template<int root = 15311432, int root_pw = 1 << 23, int inv_root = 469870224, typename IT = vector<Zp>::iterator>
void ntt(IT begin, IT end, const bool invert = false){
	int n = end - begin;
	for(int i = 1, j = 0; i < n; ++ i){
		int bit = n >> 1;
		for(; j & bit; bit >>= 1) j ^= bit;
		j ^= bit;
		if(i < j) swap(*(begin + i), *(begin + j));
	}
	for(int len = 1; len < n; len <<= 1){
		typename iterator_traits<IT>::value_type wlen = invert ? inv_root : root;
		for(int i = len << 1; i < root_pw; i <<= 1) wlen *= wlen;
		for(int i = 0; i < n; i += len << 1){
			typename iterator_traits<IT>::value_type w = 1;
			for(int j = 0; j < len; ++ j){
				auto u = *(begin + i + j), v = *(begin + i + j + len) * w;
				*(begin + i + j) = u + v;
				*(begin + i + j + len) = u - v;
				w *= wlen;
			}
		}
	}
	if(invert){
		auto inv_n = typename iterator_traits<IT>::value_type(1) / n;
		for(auto it = begin; it != end; ++ it) *it *= inv_n;
	}
}
template<typename Poly>
Poly operator*(Poly a, Poly b){
	int n = max(int(a.size()) + int(b.size()) - 1, 1);
	if(__builtin_popcount(n) != 1) n = 1 << __lg(n) + 1;
	a.resize(n), b.resize(n);
	ntt(a.begin(), a.end()), ntt(b.begin(), b.end());
	for(int i = 0; i < n; ++ i) a[i] *= b[i];
	ntt(a.begin(), a.end(), 1);
	while(!a.empty() && !a.back()) a.pop_back();
	return a;
}

// 156485479_2_4_2
// Bitwise Transformation ( Fast Walsh Hadamard Transformation, FWHT ).
// Size must be a power of two.
// O(n log n)
// Credit: TFG
template<char Conv = '^', typename IT = vector<Zp>::iterator>
void fwht(IT begin, IT end, const bool invert = false){
	int n = end - begin;
	for(int len = 1; len < n; len <<= 1){
		for(int i = 0; i < n; i += len << 1){
			for(int j = 0; j < len; ++ j){
				auto u = *(begin + i + j), v = *(begin + i + j + len);
				if(Conv == '^') *(begin + i + j) = u + v, *(begin + i + j + len) = u - v;
				if(Conv == '|') *(begin + i + j + len) += invert ? -u : u;
				if(Conv == '&') *(begin + i + j) += invert ? -v : v;
			}
		}
	}
	if(Conv == '^' && invert){
		auto inv_n = typename iterator_traits<IT>::value_type(1) / n;
		for(auto it = begin; it != end; ++ it) *it *= inv_n;
	}
}

// 156485479_2_5
// Binary Search / Ternary Search
// O(log(high - low)) applications of p
template<typename T, typename Pred>
T binary_search(T low, T high, Pred p, const bool &is_left = true){
	assert(low < high);
	while(high - low >= 2){
		auto mid = low + (high - low >> 1);
		(p(mid) == is_left ? low : high) = mid;
	}
	return is_left ? low : high;
}
// Binary search for numbers with the same remainder mod step
template<typename T, typename Pred>
T binary_search_with_step(T low0, T high0, const T &step, Pred p, const bool &is_left = true){
	assert(low0 < high0);
	auto low = low0 / step - ((low0 ^ step) < 0 && low0 % step), high = high0 / step + ((high0 ^ step) > 0 && high0 % step);
	const auto rem = low0 - low * step;
	while(high - low >= 2){
		auto mid = low + (high - low >> 1);
		(p(mid * step + rem) == is_left ? low : high) = mid;
	}
	return (is_left ? low : high) * step + rem;
}
// Ternary Search
template<typename T, typename Eval>
T ternary_search(T low, T high, Eval e, const bool &is_max = true){
	assert(low < high);
	while(high - low >= 3){
		T mid1 = low + (high - low) / 3, mid2 = low + (high - low) / 3 * 2;
		(is_max ? e(mid1) > e(mid2) : e(mid1) < e(mid2)) ? high = mid2 : low = mid1;
	}
	auto optval = e(low);
	auto res = low;
	for(auto i = low + 1; i < high; ++ i) if(is_max ? optval < e(i) : optval > e(i)) optval = e(res = i);
	return res;
}

// 156485479_2_6_1_1
// Sorted Line Container
// O(log N) per query, amortized O(1) for everything else
struct line{
	long long d, k, p;
	long long eval(long long x){ return d * x + k; }
};
template<bool GET_MAX = true>
struct sorted_line_container: deque<line>{
	static constexpr long long inf = numeric_limits<long long>::max();
	// (for doubles, use inf = 1/.0, div(a,b) = a/b)
	long long div(long long a, long long b){ return a / b - ((a ^ b) < 0 && a % b); }
	bool isect_front(iterator x, iterator y){
		if(y == end()){ x->p = inf; return false; }
		else{ x->p = div(y->k - x->k, x->d - y->d); return x->p >= y->p; }
	}
	bool isect_back(reverse_iterator x, reverse_iterator y){
		if(x == rend()) return false;
		else{ x->p = div(y->k - x->k, x->d - y->d); return x->p >= y->p; }
	}
	void push(line L){
		if(!GET_MAX) L.d = -L.d, L.k = -L.k;
		if(empty() || L.d < front().d){
			L.p = 0, push_front(L), isect_front(begin(), ++ begin());
			while(int(size()) >= 2 && isect_front(begin(), ++ begin())) erase(++ begin());
		}
		else if(L.d > back().d){
			L.p = inf, push_back(L); isect_back(++ rbegin(), rbegin());
			while(int(size()) >= 2 && isect_back(++ ++ rbegin(), ++ rbegin())) erase(-- -- end()), isect_back(++ rbegin(), rbegin());
		}
		else assert(false);
	}
	long long dec_query(long long x){
		while(int(size()) >= 2 && rbegin()->eval(x) <= (++ rbegin())->eval(x)) pop_back(); rbegin()->p = inf;
		return rbegin()->eval(x) * (GET_MAX ? 1 : -1);
	}
	long long inc_query(long long x){
		while(int(size()) >= 2 && begin()->eval(x) <= (++ begin())->eval(x)) pop_front();
		return begin()->eval(x) * (GET_MAX ? 1 : -1);
	}
	long long query(long long x){
		if(int(size()) == 1) return begin()->eval(x) * (GET_MAX ? 1 : -1);
		int low = 0, high = int(size()) - 1;
		if(begin()->eval(x) >= (++ begin())->eval(x)) return begin()->eval(x) * (GET_MAX ? 1 : -1);
		while(high - low > 1){
			int mid = low + high >> 1;
			(*this)[mid].eval(x) < (*this)[mid + 1].eval(x) ? low = mid : high = mid;
		}
		return (*this)[low + 1].eval(x) * (GET_MAX ? 1 : -1);
	}
};

// 156485479_2_6_1_2
// Line Container / Add lines of form d*x + k and query max at pos x
// O(log N) per query
// Credit: KACTL
struct line{
	mutable long long d, k, p;
	bool operator<(const line &otr) const{ return d < otr.d; }
	bool operator<(long long x) const{ return p < x;}
};
template<bool GET_MAX = true>
struct line_container: multiset<line, less<>>{
	// (for doubles, use inf = 1/.0, div(a,b) = a/b)
	static constexpr long long inf = numeric_limits<long long>::max();
	long long div(long long a, long long b){ return a / b - ((a ^ b) < 0 && a % b); }
	bool isect(iterator x, iterator y){
		if(y == end()){ x->p = inf; return false; }
		if(x->d == y->d) x->p = x->k > y->k ? inf : -inf;
		else x->p = div(y->k - x->k, x->d - y->d);
		return x->p >= y->p;
	}
	void push(line L){
		if(!GET_MAX) L.d = -L.d, L.k = -L.k;
		L.p = 0;
		auto z = insert(L), y = z ++, x = y;
		while(isect(y, z)) z = erase(z);
		if(x != begin() && isect(-- x, y)) isect(x, y = erase(y));
		while((y = x) != begin() && (-- x)->p >= y->p) isect(x, erase(y));
	}
	long long query(long long x){
		assert(!empty());
		auto l = *lower_bound(x);
		return (l.d * x + l.k) * (GET_MAX ? 1 : -1);
	}
};

// 156485479_2_6_1_3
// Li Chao Tree
// O(log N) per update and query
// Credit: Benq
struct line{
	long long d, k;
	line(long long d = 0, long long k = numeric_limits<long long>::min() / 2): d(d), k(k){ }
	long long eval(long long x){ return d * x + k; }
	bool majorize(line X, long long L, long long R){ return eval(L) >= X.eval(L) && eval(R) >= X.eval(R); }
};
template<bool GET_MAX = true>
struct lichao{
	lichao *l = 0, *r = 0;
	line S;
	lichao(): S(line()){ }
	~lichao(){
		delete l;
		delete r;
	}
	void mc(int i){
		if(i){ if(!r) r = new lichao(); }
		else{ if(!l) l = new lichao(); }
	}
	long long pq(long long X, long long L, long long R){
		long long ans = S.eval(X), M = L + R >> 1;
		if(X < M) return max(ans, l ? l->pq(X, L, M) : -(long long)9e18);
		else return max(ans, r ? r->pq(X, M, R) : -(long long)9e18);
	}
	long long query(long long X, long long L, long long R){
		return pq(X, L, R) * (GET_MAX ? 1 : -1);
	}
	void pp(line X, long long L, long long R){
		if(X.majorize(S, L, R)) swap(X, S);
		if(S.majorize(X, L, R)) return;
		if(S.eval(L) < X.eval(L)) swap(X, S);
		long long M = L + R >> 1;
		if(X.eval(M) > S.eval(M)) swap(X, S), mc(0), l->pp(X, L, M);
		else mc(1), r->pp(X, M, R);
	}
	void push(line X, long long L, long long R){
		if(!GET_MAX) X.d = -X.d, X.k = -X.k;
		pp(X, L, R);
	}
};

// 156485479_2_6_2
// Divide and Conquer DP Optimization
// Recurrence relation of form dp_next[i] = min/max{j in [0, i)} (dp[j] + cost(j, i))
// Must satisfy opt[j] <= opt[j + 1]
// Special case: cost(j, i) must be a Monge array ( if one interval contains the other, it's better to resolve them )
// Credit: cp-algorithms.com
// O(N log N). Let (i0, j0), (i1, j1), ... be the access seq for cost. Then abs(i0-i1) + abs(j0-j1) is amortized constant.
template<bool GET_MAX = true, typename T = long long>
void DCDP(auto &dp, auto &dp_next, auto &cost, int low, int high, int opt_low, int opt_high){
	if(low >= high) return;
	int mid = low + (high - low >> 1);
	pair<T, int> res{GET_MAX ? numeric_limits<T>::min() : numeric_limits<T>::max(), -1};
	for(int i = min(mid, opt_high) - 1; i >= opt_low; -- i) res = GET_MAX ? max(res, {dp[i] + cost(i, mid), i}) : min(res, {dp[i] + cost(i, mid), i});
	dp_next[mid] = res.first;
	DCDP<GET_MAX, T>(dp, dp_next, cost, low, mid, opt_low, res.second + 1), DCDP<GET_MAX, T>(dp, dp_next, cost, mid + 1, high, res.second, opt_high);
}

// 156485479_2_6_3
// Knuth DP Optimization
// Recurrence relation of form dp[i][j] = min/max{k in [i, j)} (dp[i][k] + dp[k][j] + C[i][j])
// Must satisfy C[a][c] + C[b][d] <= C[a][d] + C[b][d] (C is a monge array) and C[a][d] >= C[b][c] for all a<=b<=c<=d
// It can be proved that opt[i][j - 1] <= opt[i][j] <= opt[i + 1][j]
// Fill the dp table in increasing order of j - i.
// O(N^2)

// 156485479_2_6_4
// Lagrange DP Optimization ( Aliens Trick, Wqs Binary Search )
// Recurrence relation of form dp[i][j] = min{k in [0, j)} (dp[i - 1][k] + C[k + 1][j])
// dp[x][N] must be convex / concave
// Special case: for all a<=b<=c<=d, C[a][c] + C[b][d] <= C[a][d] + C[b][d] ( C is a Monge array )
// f(const ll &lambda, vi &previous, vi &count) returns the reduced DP value
// WARNING: the cost function for f() should be doubled
// O(log(high - low)) applications of f()
// Requires binary_search
template<typename DP, bool GET_MAX = true>
pair<long long, vector<int>> LagrangeDP(int n, DP f, long long k, long long low, long long high){
	long long resp, resq;
	vector<int> prevp(n + 1), cntp(n + 1), prevq(n + 1), cntq(n + 1);
	auto pred = [&](long long lambda){
		swap(resp, resq), swap(prevp, prevq), swap(cntp, cntq);
		resp = f(lambda, prevp, cntp);
		return GET_MAX ? cntp.back() <= k : cntp.back() >= k;
	};
	long long lambda = binary_search(2 * low - 1, 2 * high + 1, 2, pred);
	pred(lambda + 2), pred(lambda);
	if(cntp.back() == k){
		vector<int> path{n};
		for(int u = n; u; ) path.push_back(u = prevp[u]);
		return {resp - lambda * k >> 1, path};
	}
	else{
		resp = resp - lambda * cntp.back() >> 1, resq = resq - (lambda + 2) * cntq.back() >> 1;
		long long res = resp + (resq - resp) / (cntq.back() - cntp.back()) * (k - cntp.back());
		if(!GET_MAX) swap(prevp, prevq), swap(cntp, cntq);
		int i = n, j = n, d = k - cntp.back();
		while(1){
			if(prevp[i] <= prevq[j]){
				while(prevp[i] <= prevq[j] && cntq[j] - cntp[i] > d) j = prevq[j];
				if(prevp[i] <= prevq[j] && cntq[j] - cntp[i] == d) break;
			}
			else i = prevp[i], j = prevq[j];
		}
		vector<int> path{n};
		for(int u = n; u != i; ) path.push_back(u = prevp[u]);
		path.push_back(prevq[j]);
		for(int u = prevq[j]; u; ) path.push_back(u = prevq[u]);
		return {res, path};
	}
}

// 156485479_2_6_5
// Monotone Queue DP Optimization
// Recurrence relation of form dp[i] = min/max{j in [0, i)} (dp[j] + cost(j, i))
// dp[j][i] must be a Monge array ( if one interval contains the other, it's better to resolve them )
// O(n log n)
template<bool GET_MAX = true>
vector<long long> monotone_queue_dp(int n, long long init, auto cost){
	vector<long long> dp(n, init);
	auto cross = [&](int i, int j){
		int l = j, r = n;
		while(r - l > 1){
			int mid = l + r >> 1;
			(GET_MAX ? dp[i] + cost(i, mid) >= dp[j] + cost(j, mid) : dp[i] + cost(i, mid) <= dp[j] + cost(j, mid)) ? l = mid : r = mid;
		}
		return l;
	};
	deque<int> q{0};
	for(auto i = 1; i < n; ++ i){
		while(int(q.size()) > 1 && cross(*q.begin(), *next(q.begin())) < i) q.pop_front();
		dp[i] = dp[q.front()] + cost(q.front(), i);
		while(int(q.size()) > 1 && cross(*next(q.rbegin()), *q.rbegin()) >= cross(*q.rbegin(), i)) q.pop_back();
		q.push_back(i);
	}
	return dp;
}

// 156485479_2_7
// Kadane
// O(N)
template<typename IT, typename T = int>
auto kadane(IT begin, IT end, T init = 0){
	typename iterator_traits<IT>::value_type lm = init, gm = init;
	for(; begin != end; ++ begin) lm = max(*begin, *begin + lm), gm = max(gm, lm);
	return gm;
}

// 156485479_2_8
// Big Integer
// Credit: https://gist.github.com/ar-pa/957297fb3f88996ead11
const int base = 1e9;
const int base_digits = 9; 
struct bigint{
	vector<int> a;
	int sign;
	int size(){
		if(a.empty()) return 0;
		int res = (int(a.size()) - 1) * base_digits, ca = a.back();
		while(ca) ++ res, ca /= 10;
		return res;
	}
	bigint operator^(const bigint &v){
		bigint res = 1, a = *this, b = v;
		for(; !b.isZero(); a *= a, b /= 2) if(b % 2) res *= a;
		return res;
	}
	string to_string(){
		stringstream ss;
		ss << *this;
		string s;
		ss >> s;
		return s;
	}
	int sumof(){
		string s = to_string();
		int res = 0;
		for(auto c: s) res += c - '0';
		return res;
	}
	bigint(): sign(1){ }
	bigint(long long v){ *this = v; }
	bigint(const string &s){ read(s); }
	void operator=(const bigint &v){
		sign = v.sign;
		a = v.a;
	}
	void operator=(long long v){
		sign = 1;
		a.clear();
		if(v < 0) sign = -1, v = -v;
		for (; v > 0; v = v / base) a.push_back(v % base);
	}
	bigint operator+(const bigint &v) const{
		if(sign == v.sign){
			bigint res = v;
			for(int i = 0, carry = 0; i < max(a.size(), v.a.size()) || carry; ++ i){
				if(i == res.a.size()) res.a.push_back(0);
				res.a[i] += carry + (i < a.size() ? a[i] : 0);
				carry = res.a[i] >= base;
				if(carry) res.a[i] -= base;
			}
			return res;
		}
		return *this - (-v);
	}
	bigint operator-(const bigint &v) const{
		if(sign == v.sign){
			if(abs() >= v.abs()){
				bigint res = *this;
				for(int i = 0, carry = 0; i < v.a.size() || carry; ++ i){
					res.a[i] -= carry + (i < v.a.size() ? v.a[i] : 0);
					carry = res.a[i] < 0;
					if(carry) res.a[i] += base;
				}
				res.trim();
				return res;
			}
			return -(v - *this);
		}
		return *this + (-v);
	}
	void operator*=(int v){
		if(v < 0) sign = -sign, v = -v;
		for(int i = 0, carry = 0; i < a.size() || carry; ++ i){
			if(i == a.size()) a.push_back(0);
			long long cur = a[i] * (long long)v + carry;
			carry = (int)(cur / base);
			a[i] = (int)(cur % base);
			//asm("divl %%ecx" : "=a"(carry), "=d"(a[i]) : "A"(cur), "c"(base));
		}
		trim();
	}
	bigint operator*(int v) const{
		bigint res = *this;
		res *= v;
		return res;
	}
	void operator*=(long long v){
		if(v < 0) sign = -sign, v = -v;
		if(v > base){
			*this = *this * (v / base) * base + *this * (v % base);
			return ;
		}
		for(int i = 0, carry = 0; i < a.size() || carry; ++ i){
			if (i == a.size()) a.push_back(0);
			long long cur = a[i] * (long long)v + carry;
			carry = (int)(cur / base);
			a[i] = (int)(cur % base);
			//asm("divl %%ecx" : "=a"(carry), "=d"(a[i]) : "A"(cur), "c"(base));
		}
		trim();
	}
	bigint operator*(long long v) const{
		bigint res = *this;
		res *= v;
		return res;
	}
	friend pair<bigint, bigint> divmod(const bigint &a1, const bigint &b1) {
		int norm = base / (b1.a.back() + 1);
		bigint a = a1.abs() * norm, b = b1.abs() * norm, q, r;
		q.a.resize(a.a.size());
		for(int i = int(a.a.size()) - 1; i >= 0; -- i){
			r *= base;
			r += a.a[i];
			int s1 = r.a.size() <= b.a.size() ? 0 : r.a[b.a.size()];
			int s2 = r.a.size() <= b.a.size() - 1 ? 0 : r.a[int(b.a.size()) - 1];
			int d = ((long long)base * s1 + s2) / b.a.back();
			r -= b * d;
			while(r < 0) r += b, -- d;
			q.a[i] = d;
		}
		q.sign = a1.sign * b1.sign, r.sign = a1.sign;
		q.trim(), r.trim();
		return {q, r / norm};
	}
	bigint operator/(const bigint &v) const{ return divmod(*this, v).first; }
	bigint operator%(const bigint &v) const{ return divmod(*this, v).second; }
	void operator/=(int v){
		if(v < 0) sign = -sign, v = -v;
		for(int i = int(a.size()) - 1, rem = 0; i >= 0; -- i){
			long long cur = a[i] + rem * (long long)base;
			a[i] = (int)(cur / v);
			rem = (int)(cur % v);
		}
		trim();
	}
	bigint operator/(int v) const{
		bigint res = *this;
		res /= v;
		return res;
	}
	int operator%(int v) const{
		if(v < 0) v = -v;
		int m = 0;
		for(int i = int(a.size()) - 1; i >= 0; -- i) m = (a[i] + m * (long long)base) % v;
		return m * sign;
	}
	void operator+=(const bigint &v){ *this = *this + v; }
	void operator-=(const bigint &v){ *this = *this - v; }
	void operator*=(const bigint &v){ *this = *this * v; }
	void operator/=(const bigint &v){ *this = *this / v; }
	bool operator<(const bigint &v) const{
		if(sign != v.sign) return sign < v.sign;
		if(a.size() != v.a.size()) return a.size() * sign < int(v.a.size()) * v.sign;
		for(int i = int(a.size()) - 1; i >= 0; -- i) if(a[i] != v.a[i]) return a[i] * sign < v.a[i] * sign;
		return false;
	}
	bool operator>(const bigint &v) const{ return v < *this; }
	bool operator<=(const bigint &v) const{ return !(v < *this); }
	bool operator>=(const bigint &v) const{ return !(*this < v); }
	bool operator==(const bigint &v) const{ return !(*this < v) && !(v < *this); }
	bool operator!=(const bigint &v) const{ return *this < v || v < *this; }
	void trim(){
		while (!a.empty() && !a.back()) a.pop_back();
		if(a.empty()) sign = 1;
	}
	bool isZero() const{ return a.empty() || (a.size() == 1 && !a[0]); }
	bigint operator-() const{
		bigint res = *this;
		res.sign = -sign;
		return res;
	}
	bigint abs() const{
		bigint res = *this;
		res.sign *= res.sign;
		return res;
	}
	long long longValue() const{
		long long res = 0;
		for(int i = int(a.size()) - 1; i >= 0; -- i) res = res * base + a[i];
		return res * sign;
	}
	friend bigint gcd(const bigint &a, const bigint &b){ return b.isZero() ? a : gcd(b, a % b); }
	friend bigint lcm(const bigint &a, const bigint &b){ return a / gcd(a, b) * b; }
	void read(const string &s){
		sign = 1;
		a.clear();
		int pos = 0;
		while(pos < s.size() && (s[pos] == '-' || s[pos] == '+')){
			if(s[pos] == '-') sign = -sign;
			++ pos;
		}
		for(int i = int(s.size()) - 1; i >= pos; i -= base_digits){
			int x = 0;
			for(int j = max(pos, i - base_digits + 1); j <= i; ++ j) x = x * 10 + s[j] - '0';
			a.push_back(x);
		}
		trim();
	}
	friend istream& operator>>(istream &stream, bigint &v){
		string s;
		stream >> s;
		v.read(s);
		return stream;
	}
	friend ostream& operator<<(ostream &stream, const bigint &v){
		if(v.sign == -1) stream << '-';
		stream << (v.a.empty() ? 0 : v.a.back());
		for(int i = int(v.a.size()) - 2; i >= 0; -- i) stream << setw(base_digits) << setfill('0') << v.a[i];
		return stream;
	}
	static vector<int> convert_base(const vector<int> &a, int old_digits, int new_digits){
		vector<long long> p(max(old_digits, new_digits) + 1);
		p[0] = 1;
		for(int i = 1; i < p.size(); ++ i) p[i] = p[i - 1] * 10;
		vector<int> res;
		long long cur = 0;
		int cur_digits = 0;
		for(int i = 0; i < a.size(); i++){
			cur += a[i] * p[cur_digits];
			cur_digits += old_digits;
			while(cur_digits >= new_digits){
				res.push_back(int(cur % p[new_digits]));
				cur /= p[new_digits];
				cur_digits -= new_digits;
			}
		}
		res.push_back((int)cur);
		while(!res.empty() && !res.back()) res.pop_back();
		return res;
	}
	static vector<long long> karatsubaMultiply(const vector<long long> &a, const vector<long long> &b){
		int n = a.size();
		vector<long long> res(n << 1);
		if(n <= 32){
			for(int i = 0; i < n; ++ i) for(int j = 0; j < n; ++ j) res[i + j] += a[i] * b[j];
			return res;
		}
		int k = n >> 1;
		vector<long long> a1(a.begin(), a.begin() + k), a2(a.begin() + k, a.end());
		vector<long long> b1(b.begin(), b.begin() + k), b2(b.begin() + k, b.end());
		vector<long long> a1b1 = karatsubaMultiply(a1, b1), a2b2 = karatsubaMultiply(a2, b2);
		for(int i = 0; i < k; ++ i) a2[i] += a1[i];
		for(int i = 0; i < k; ++ i) b2[i] += b1[i];
		vector<long long> r = karatsubaMultiply(a2, b2);
		for(int i = 0; i < a1b1.size(); ++ i) r[i] -= a1b1[i];
		for(int i = 0; i < a2b2.size(); ++ i) r[i] -= a2b2[i];
		for(int i = 0; i < r.size(); ++ i) res[i + k] += r[i];
		for(int i = 0; i < a1b1.size(); ++ i) res[i] += a1b1[i];
		for (int i = 0; i < a2b2.size(); ++ i) res[i + n] += a2b2[i];
		return res;
	}
	bigint operator*(const bigint &v) const{
		vector<int> a6 = convert_base(this->a, base_digits, 6), b6 = convert_base(v.a, base_digits, 6);
		vector<long long> a(a6.begin(), a6.end()), b(b6.begin(), b6.end());
		while(a.size() < b.size()) a.push_back(0);
		while(b.size() < a.size()) b.push_back(0);
		while(a.size() & (int(a.size()) - 1)) a.push_back(0), b.push_back(0);
		vector<long long> c = karatsubaMultiply(a, b);
		bigint res;
		res.sign = sign * v.sign;
		for(int i = 0, carry = 0; i < c.size(); ++ i){
			long long cur = c[i] + carry;
			res.a.push_back((int)(cur % 1000000));
			carry = (int)(cur / 1000000);
		}
		res.a = convert_base(res.a, 6, base_digits);
		res.trim();
		return res;
	}
};

// 156485479_2_9
// Modular Arithmetics
// Credit: Tourist
template<typename T>
struct Z_p{
	using Type = typename decay<decltype(T::value)>::type;
	static vector<Type> mod_inv; 
	constexpr Z_p(): value(){ }
	template<typename U> Z_p(const U &x){ value = normalize(x); }
	template<typename U> static Type normalize(const U &x){
		Type v;
		if(-mod() <= x && x < mod()) v = static_cast<Type>(x);
		else v = static_cast<Type>(x % mod());
		if(v < 0) v += mod();
		return v;
	}
	const Type& operator()() const{ return value; }
	template<typename U> explicit operator U() const{ return static_cast<U>(value); }
	constexpr static Type mod(){ return T::value; }
	Z_p &operator+=(const Z_p &otr){ if((value += otr.value) >= mod()) value -= mod(); return *this; }
	Z_p &operator-=(const Z_p &otr){ if((value -= otr.value) < 0) value += mod(); return *this; }
	template<typename U> Z_p &operator+=(const U &otr){ return *this += Z_p(otr); }
	template<typename U> Z_p &operator-=(const U &otr){ return *this -= Z_p(otr); }
	Z_p &operator++(){ return *this += 1; }
	Z_p &operator--(){ return *this -= 1; }
	Z_p operator++(int){ Z_p result(*this); *this += 1; return result; }
	Z_p operator--(int){ Z_p result(*this); *this -= 1; return result; }
	Z_p operator-() const{ return Z_p(-value); }
	template<typename U = T>
	typename enable_if<is_same<typename Z_p<U>::Type, int>::value, Z_p>::type &operator*=(const Z_p& rhs){
		#ifdef _WIN32
		uint64_t x = static_cast<int64_t>(value) * static_cast<int64_t>(rhs.value);
		uint32_t xh = static_cast<uint32_t>(x >> 32), xl = static_cast<uint32_t>(x), d, m;
		asm(
			"divl %4; \n\t"
			: "=a" (d), "=d" (m)
			: "d" (xh), "a" (xl), "r" (mod())
		);
		value = m;
		#else
		value = normalize(static_cast<int64_t>(value) * static_cast<int64_t>(rhs.value));
		#endif
		return *this;
	}
	template<typename U = T>
	typename enable_if<is_same<typename Z_p<U>::Type, int64_t>::value, Z_p>::type &operator*=(const Z_p &rhs){
		int64_t q = static_cast<int64_t>(static_cast<long double>(value) * rhs.value / mod());
		value = normalize(value * rhs.value - q * mod());
		return *this;
	}
	template<typename U = T>
	typename enable_if<!is_integral<typename Z_p<U>::Type>::value, Z_p>::type &operator*=(const Z_p &rhs){
		value = normalize(value * rhs.value);
		return *this;
	}
	Z_p operator^(long long e) const{
		Z_p b = *this, res = 1;
		if(e < 0) b = 1 / b, e = -e;
		for(; e; b *= b, e >>= 1) if(e & 1) res *= b;
		return res;
	}
	Z_p &operator^=(const long long &e){ return *this = *this ^ e; }
	Z_p &operator/=(const Z_p &otr){
		Type a = otr.value, m = mod(), u = 0, v = 1;
		if(a < int(mod_inv.size())) return *this *= mod_inv[a];
		while(a){
			Type t = m / a;
			m -= t * a; swap(a, m);
			u -= t * v; swap(u, v);
		}
		assert(m == 1);
		return *this *= u;
	}
	template<typename U> friend const Z_p<U> &abs(const Z_p<U> &v){ return v; }
	template<typename U> friend bool operator==(const Z_p<U> &lhs, const Z_p<U> &rhs);
	template<typename U> friend bool operator<(const Z_p<U> &lhs, const Z_p<U> &rhs);
	template<typename U> friend istream &operator>>(istream &in, Z_p<U> &number);
	Type value;
};
template<typename T> bool operator==(const Z_p<T> &lhs, const Z_p<T> &rhs){ return lhs.value == rhs.value; }
template<typename T, typename U, enable_if_t<is_integral_v<U>>* = nullptr> bool operator==(const Z_p<T>& lhs, U rhs){ return lhs == Z_p<T>(rhs); }
template<typename T, typename U, enable_if_t<is_integral_v<U>>* = nullptr> bool operator==(U lhs, const Z_p<T> &rhs){ return Z_p<T>(lhs) == rhs; }
template<typename T> bool operator!=(const Z_p<T> &lhs, const Z_p<T> &rhs){ return !(lhs == rhs); }
template<typename T, typename U, enable_if_t<is_integral_v<U>>* = nullptr> bool operator!=(const Z_p<T> &lhs, U rhs){ return !(lhs == rhs); }
template<typename T, typename U, enable_if_t<is_integral_v<U>>* = nullptr> bool operator!=(U lhs, const Z_p<T> &rhs){ return !(lhs == rhs); }
template<typename T> bool operator<(const Z_p<T> &lhs, const Z_p<T> &rhs){ return lhs.value < rhs.value; }
template<typename T> Z_p<T> operator+(const Z_p<T> &lhs, const Z_p<T> &rhs){ return Z_p<T>(lhs) += rhs; }
template<typename T, typename U, enable_if_t<is_integral_v<U>>* = nullptr> Z_p<T> operator+(const Z_p<T> &lhs, U rhs){ return Z_p<T>(lhs) += rhs; }
template<typename T, typename U, enable_if_t<is_integral_v<U>>* = nullptr> Z_p<T> operator+(U lhs, const Z_p<T> &rhs){ return Z_p<T>(lhs) += rhs; }
template<typename T> Z_p<T> operator-(const Z_p<T> &lhs, const Z_p<T> &rhs){ return Z_p<T>(lhs) -= rhs; }
template<typename T, typename U, enable_if_t<is_integral_v<U>>* = nullptr> Z_p<T> operator-(const Z_p<T>& lhs, U rhs){ return Z_p<T>(lhs) -= rhs; }
template<typename T, typename U, enable_if_t<is_integral_v<U>>* = nullptr> Z_p<T> operator-(U lhs, const Z_p<T> &rhs){ return Z_p<T>(lhs) -= rhs; }
template<typename T> Z_p<T> operator*(const Z_p<T> &lhs, const Z_p<T> &rhs){ return Z_p<T>(lhs) *= rhs; }
template<typename T, typename U, enable_if_t<is_integral_v<U>>* = nullptr> Z_p<T> operator*(const Z_p<T>& lhs, U rhs){ return Z_p<T>(lhs) *= rhs; }
template<typename T, typename U, enable_if_t<is_integral_v<U>>* = nullptr> Z_p<T> operator*(U lhs, const Z_p<T> &rhs){ return Z_p<T>(lhs) *= rhs; }
template<typename T> Z_p<T> operator/(const Z_p<T> &lhs, const Z_p<T> &rhs) { return Z_p<T>(lhs) /= rhs; }
template<typename T, typename U, enable_if_t<is_integral_v<U>>* = nullptr> Z_p<T> operator/(const Z_p<T>& lhs, U rhs) { return Z_p<T>(lhs) /= rhs; }
template<typename T, typename U, enable_if_t<is_integral_v<U>>* = nullptr> Z_p<T> operator/(U lhs, const Z_p<T> &rhs) { return Z_p<T>(lhs) /= rhs; }
template<typename T> istream &operator>>(istream &in, Z_p<T> &number){
	typename common_type<typename Z_p<T>::Type, int64_t>::type x;
	in >> x;
	number.value = Z_p<T>::normalize(x);
	return in;
}
template<typename T> ostream &operator<<(ostream &out, const Z_p<T> &number){ return out << number(); }

/*
using ModType = int;
struct VarMod{ static ModType value; };
ModType VarMod::value;
ModType &mod = VarMod::value;
using Zp = Z_p<VarMod>;
*/

constexpr int mod = 1e9 + 7;
//constexpr int mod = 998244353;
using Zp = Z_p<integral_constant<decay<decltype(mod)>::type, mod>>;

template<typename T> vector<typename Z_p<T>::Type> Z_p<T>::mod_inv;
template<typename T = integral_constant<decay<decltype(mod)>::type, mod>>
void precalc_inverse(size_t SZ){
	auto &inv = Z_p<T>::mod_inv;
	if(inv.empty()) inv.assign(2, 1);
	for(; inv.size() <= SZ; ) inv.push_back((mod - 1LL * mod / int(inv.size()) * inv[mod % int(inv.size())] % mod) % mod);
}

// 156485479_2_10_1
// Matroid Intersection
// Credit: tfg / chilli / pajenegod ( https://github.com/tfg50/Competitive-Programming/blob/master/Biblioteca/Math/MatroidIntersection.cpp )
// Examples of Matroids
struct ColorMat{
	vector<int> cnt, clr;
	ColorMat(int n, vector<int> clr): cnt(n), clr(clr){ }
	bool check(int x){ return !cnt[clr[x]]; }
	void add(int x){ ++ cnt[clr[x]]; }
	void clear(){ fill(cnt.begin(), cnt.end(), 0); }
};
struct GraphMat{
	disjoint_set dsu;
	vector<array<int, 2>> e;
	GraphMat(int n, vector<array<int, 2>> e): dsu(n), e(e){ }
	bool check(int x){ return !dsu.share(e[x][0], e[x][1]); }
	void add(int x){ dsu.merge(e[x][0], e[x][1]); }
	void clear(){ dsu = disjoint_set(int(dsu.p.size())); }
};

// R^2N(M2.add + M1.check + M2.check) + R^3 M1.add + R^2 M1.clear + RN M2.clear
template<typename M1, typename M2>
struct Matroid_Intersection{
	int n;
	vector<char> iset;
	M1 m1; M2 m2;
	Matroid_Intersection(M1 m1, M2 m2, int n): n(n), iset(n + 1), m1(m1), m2(m2) {}
	vector<int> solve(){
		for(int i = 0; i < n; ++ i) if(m1.check(i) && m2.check(i)) iset[i] = true, m1.add(i), m2.add(i);
		while(augment());
		vector<int> res;
		for(int i = 0; i < n; ++ i) if(iset[i]) res.push_back(i);
		return res;
	}
	bool augment(){
		vector<int> frm(n, -1);
		queue<int> q({n}); // starts at dummy node
		auto fwdE = [&](int a){
			vector<int> res;
			m1.clear();
			for(int v = 0; v < n; ++ v) if(iset[v] && v != a) m1.add(v);
			for(int b = 0; b < n; ++ b) if(!iset[b] && frm[b] == -1 && m1.check(b)) res.push_back(b), frm[b] = a;
			return res;
		};
		auto backE = [&](int b){
			m2.clear();
			for(int cas = 0; cas < 2; ++ cas) for(int v = 0; v < n; ++ v)
				if((v == b || iset[v]) && (frm[v] == -1) == cas){
					if(!m2.check(v)) return cas ? q.push(v), frm[v] = b, v : -1;
					m2.add(v);
				}
			return n;
		};
		while(!q.empty()){
			int a = q.front(), c; q.pop();
			for(int b: fwdE(a)) while((c = backE(b)) >= 0) if(c == n){
				while(b != n) iset[b] ^= 1, b = frm[b];
				return true;
			}
		}
		return false;
	}
};

// 156485479_2_10_2
// Matroid Union

// 156485479_2_11
// LIS / Returns the indices of the longest increasing sequence
// O(n log n)
// Credit: KACTL
template<typename T>
vector<int> LIS(const vector<T> &a){
	int n = int(a.size());
	if(!n) return {};
	vector<int> prev(n);
	typedef pair<T, int> p;
	vector<p> active;
	for(int i = 0; i < n; ++ i){
		// change 0 -> i for longest non-decreasing subsequence
		auto it = lower_bound(active.begin(), active.end(), p{a[i], 0});
		if(it == active.end()) active.emplace_back(), it = prev(active.end());
		*it = {a[i], i};
		prev[i] = it == active.begin() ? 0 : prev(it)->second;
	}
	int L = int(active.size()), cur = active.back().second;
	vector<int> res(L);
	while(L --) res[L] = cur, cur = prev[cur];
	return res;
}

// 156485479_2_12
// K Dimensional Array
template<typename T>
struct kdarray{
	int K;
	vector<int> n, p;
	vector<T> val;
	T &operator[](const vector<int> &x){
		int pos = 0;
		for(int i = 0; i < int(x.size()); ++ i) pos += p[i] * x[i];
		return val[pos];
	}
	kdarray(){}
	kdarray(const vector<int> &n, T id = T()): K(int(n.size())), n(n), val(accumulate(n.begin(), n.end(), 1, multiplies<>()), id), p(K + 1, 1){
		partial_sum(n.begin(), n.end(), p.begin() + 1, multiplies<>());
	}
	template<typename U>
	kdarray(const kdarray<U> &arr): K(arr.K), n(arr.n), p(arr.p), val(arr.val.begin(), arr.val.end()){ }
};
template<typename T>
istream &operator>>(istream &in, kdarray<T> &arr){
	for(vector<int> i(arr.K); in >> arr[i]; ){
		for(int d = arr.K - 1; d >= 0; -- d){
			if(++ i[d] < arr.n[d]) break;
			if(!d) goto ESCAPE;
			i[d] = 0;
		}
	}
	ESCAPE:;
	return in;
}
template<typename T>
ostream &operator<<(ostream &out, kdarray<T> &arr){
	for(vector<int> i(arr.K); true; ){
		out << "\n[{";
		for(int d = 0; d < arr.K; ++ d){
			out << i[d] << ", ";
		}
		if(arr.K) out << "\b\b";
		out << "}] -> ";
		// out << arr[i]; // format this part
		for(int d = arr.K - 1; d >= 0; -- d){
			if(++ i[d] < arr.n[d]) break;
			if(!d) goto ESCAPE;
			i[d] = 0;
		}
	}
	ESCAPE:;
	return out;
}

// 156485479_2_13
// K Dimensional Prefix Sum
// O(K * Product(n)) processing, O(2^K) per query
template<typename T, typename BO = plus<>, typename IO = minus<>>
struct kdsum{
	BO bin_op;
	IO inv_op;
	T id;
	kdarray<T> val;
	template<typename U>
	kdsum(const kdarray<U> &arr, BO bin_op = plus<>{}, IO inv_op = minus<>{}, T id = 0LL): val(arr), bin_op(bin_op), inv_op(inv_op), id(id){
		vector<int> cur, from;
		for(int t = 0, ncnt; t < val.K; ++ t){
			cur.assign(val.K, 0), from.assign(val.K, 0), -- from[t], ncnt = 1;
			while(1){
				T &c = val[cur];
				c = bin_op(c, ncnt ? id : val[from]);
				for(int i = val.K - 1; i >= 0; -- i){
					if(from[i] < 0) -- ncnt;
					if(++ from[i], ++ cur[i] < val.n[i]) break;
					if(!i) goto ESCAPE;
					cur[i] = 0, ncnt += (i == t) - (from[i] < 0), from[i] = (i != t) - 1;
				}
			}
			ESCAPE:;
		}
	}
	T query(const vector<int> &low, const vector<int> &high){
		T res = id;
		static vector<int> cur; cur.assign(val.K, 0);
		for(int mask = 0, ncnt = 0; mask < 1 << val.K; ++ mask){
			for(int bit = 0; bit < val.K; ++ bit){
				if(mask & 1 << bit){
					ncnt += !low[bit] - !~cur[bit], cur[bit] = low[bit] - 1;
					break;
				}
				else ncnt += !high[bit] - !~cur[bit], cur[bit] = high[bit] - 1;
			}
			res = __builtin_popcount(mask) & 1 ? inv_op(res, ncnt ? id : val[cur]) : bin_op(res, ncnt ? id : val[cur]);
		}
		return res;
	}
	T query(vector<int> high){
		for(int d = 0; d < val.K; ++ d){
			if(high[d]) -- high[d];
			else return id;
		}
		return val[high];
	}
};

// 156485479_3_1
// Sparse Table
// The binary operator must be idempotent and associative
// O(n log n) preprocessing, O(1) per query
// Credit: Kactl
template<typename T, typename BO>
struct sparse_table{
	int n;
	BO bin_op;
	T id;
	vector<vector<T>> val;
	template<typename IT>
	sparse_table(IT begin, IT end, BO bin_op, T id): n(end - begin), bin_op(bin_op), id(id), val(1, {begin, end}){
		for(int p = 1, i = 1; p << 1 <= n; p <<= 1, ++ i){
			val.emplace_back(n - (p << 1) + 1);
			for(int j = 0; j < int(val[i].size()); ++ j) val[i][j] = bin_op(val[i - 1][j], val[i - 1][j + p]);
		}
	}
	sparse_table(){ }
	T query(int l, int r){
		if(l >= r) return id;
		int d = __lg(r - l);
		return bin_op(val[d][l], val[d][r - (1 << d)]);
	}
};
// 2D Sparse Table
// The binary operator must be idempotent and associative
// O(nm log nm) processing, O(1) per query
template<typename T, typename BO>
struct sparse_table_2d{
	int n, m;
	BO bin_op;
	T id;
	vector<vector<vector<vector<T>>>> val;
	sparse_table_2d(const vector<vector<T>> &arr, BO bin_op, T id): n(arr.size()), m(arr[0].size()), bin_op(bin_op), id(id), val(__lg(n) + 1, vector<vector<vector<T>>>(__lg(m) + 1, arr)){
		for(int ii = 0; ii < n; ++ ii) for(int jj = 0; jj < m; ++ jj){
			for(int i = 0, j = 0; j < __lg(m); ++ j) val[i][j + 1][ii][jj] = bin_op(val[i][j][ii][jj], val[i][j][ii][min(m - 1, jj + (1 << j))]);
		}
		for(int i = 0; i < __lg(n); ++ i) for(int ii = 0; ii < n; ++ ii){
			for(int j = 0; j <= __lg(m); ++ j) for(int jj = 0; jj < m; ++ jj){
				val[i + 1][j][ii][jj] = bin_op(val[i][j][ii][jj], val[i][j][min(n - 1, ii + (1 << i))][jj]);
			}
		}
	}
	T query(int pl, int ql, int pr, int qr){
		if(pl >= pr || ql >= qr) return id;
		int pd = __lg(pr - pl), qd = __lg(qr - ql);
		return bin_op(bin_op(val[pd][qd][pl][ql], val[pd][qd][pl][qr - (1 << qd)]), bin_op(val[pd][qd][pr - (1 << pd)][ql], val[pd][qd][pr - (1 << pd)][qr - (1 << qd)]));
	}
};

// 156485479_3_2_1
// Iterative Segment Tree
// O(n) processing, O(log n) per query
// Credit: https://codeforces.com/blog/entry/18051
struct segment_tree{
	using Q = long long; // Query Type
	Q merge(const Q &lval, const Q &rval, int l, int m, int r){
		return lval + rval;
	} // merge two nodes representing the intervals [l, m) and [m, r)
	Q id{};
	Q init(int p){
		return id;
	}

	int n;
	vector<int> roots;
	vector<array<int, 2>> range; // Node u represents the range range[u] = [l, r)
	vector<Q> val;
	void init_range(){
		for(int i = n; i < n << 1; ++ i) range[i] = {i - n, i - n + 1};
		for(int i = n - 1; i > 0; -- i) range[i] = {range[i << 1][0], range[i << 1 | 1][1]};
	}
	void build(int l, int r){
		for(l += n, r += n - 1; l > 1; ){
			l >>= 1, r >>= 1;
			for(int i = r; i >= l; -- i) val[i] = merge(val[i << 1], val[i << 1 | 1], range[i << 1][0], range[i << 1][1], range[i << 1 | 1][1]);
		}
	}
	Q operator[](int p) const{
		return val[p + n];
	}
	template<typename IT>
	segment_tree(IT begin, IT end): n(end - begin), range(n << 1), val(n << 1, id){
		init_range();
		for(int i = 0; i < n; ++ i) val[i + n] = *(begin ++);
		build(0, n);
	}
	segment_tree(int n): n(n), range(n << 1), val(n << 1, id){
		init_range();
		for(int i = 0; i < n; ++ i) val[i + n] = init(i);
		build(0, n);
	}
	void init_roots(){
		vector<int> roots_r;
		for(auto l = n, r = n << 1; l < r; l >>= 1, r >>= 1){
			if(l & 1) roots.push_back(l ++);
			if(r & 1) roots_r.push_back(-- r);
		}
		roots.insert(roots.end(), roots_r.rbegin(), roots_r.rend());
	}
	void update(int p, Q x){
		for(val[p += n] = x; p >>= 1; ) val[p] = merge(val[p << 1], val[p << 1 | 1], range[p << 1][0], range[p << 1][1], range[p << 1 | 1][1]);
	}
	Q query(int ql, int qr){
		if(ql >= qr) return id;
		int mid;
		Q res_l = id, res_r = id;
		for(int l = ql + n, r = qr + n; l < r; l >>= 1, r >>= 1){
			if(l & 1) res_l = merge(res_l, val[l], ql, range[l][0], range[l][1]), mid = range[l][1], ++ l;
			if(r & 1) -- r, res_r = merge(val[r], res_r, range[r][0], range[r][1], qr), mid = range[r][0];
		}
		return merge(res_l, res_r, ql, mid, qr);
	}
};

// 156485479_3_2_2
// Iterative Segment Tree with Reversed Operation ( Commutative Operation Only )
// O(n) Preprocessing, O(1) per query
// Credit: https://codeforces.com/blog/entry/18051
struct reverse_segment_tree{
	using L = array<long long, 2>;
	L apply(L val, L x, array<int, 2> r, array<int, 2> rr){ // r is a subset of rr
		return {val[0] + x[0] + (r[0] - rr[0]) * x[1], val[1] + x[1]};
	}
	L id{};

	int n;
	vector<int> roots;
	vector<array<int, 2>> range;
	vector<L> val;
	void init_range(){
		for(int i = n; i < n << 1; ++ i) range[i] = {i - n, i - n + 1};
		for(int i = n - 1; i > 0; -- i) range[i] = {range[i << 1][0], range[i << 1 | 1][1]};
	}
	template<typename IT>
	reverse_segment_tree(IT begin, IT end): n(end - begin), range(n << 1), val(n << 1, id){
		init_range();
		copy(begin, end, val.begin() + n);
	}
	reverse_segment_tree(int n): n(n), range(n << 1), val(n << 1, id){
		init_range();
	}
	void init_roots(){
		vector<int> roots_r;
		for(auto l = n, r = n << 1; l < r; l >>= 1, r >>= 1){
			if(l & 1) roots.push_back(l ++);
			if(r & 1) roots_r.push_back(-- r);
		}
		roots.insert(roots.end(), roots_r.rbegin(), roots_r.rend());
	}
	void update(int ql, int qr, L x){
		if(ql >= qr) return;
		for(int l = ql + n, r = qr + n; l < r; l >>= 1, r >>= 1){
			if(l & 1) val[l] = apply(val[l], x, range[l], {ql, qr}), ++ l;
			if(r & 1) -- r, val[r] = apply(val[r], x, range[r], {ql, qr});
		}
	}
	L query(int qp){
		L res = id;
		for(int p = qp + n; p > 0; p >>= 1) res = apply(res, val[p], {qp, qp + 1}, range[p]);
		return res;
	}
	void push(){
		for(int i = 1; i < n; ++ i){
			val[i << 1] = apply(val[i << 1], val[i], range[i << 1], range[i]);
			val[i << 1 | 1] = apply(val[i << 1 | 1], val[i], range[i << 1], range[i]);
			val[i] = id;
		}
	}
};

// 156485479_3_2_3
// Iterative 2D Segment Tree ( Only for commutative group )
// O(nm) processing, O(log nm) per query
template<typename T, typename BO>
struct segment_tree_2d{
	int n, m;
	BO bin_op;
	T id;
	vector<vector<T>> val;
	segment_tree_2d(const vector<vector<T>> &arr, BO bin_op, T id): n(arr.size()), m(arr[0].size()), bin_op(bin_op), id(id), val(n << 1, vector<T>(m << 1, id)){
		for(int i = 0; i < n; ++ i) for(int j = 0; j < m; ++ j) val[i + n][j + m] = arr[i][j];
		for(int i = n - 1; i > 0; -- i) for(int j = 0; j < m; ++ j) val[i][j + m] = bin_op(val[i << 1][j + m], val[i << 1 | 1][j + m]);
		for(int i = 1; i < n << 1; ++ i) for(int j = m - 1; j > 0; -- j) val[i][j] = bin_op(val[i][j << 1], val[i][j << 1 | 1]);
	}
	template<bool increment = true>
	void update(int p, int q, T x){
		p += n, q += m, val[p][q] = increment ? bin_op(val[p][q], x) : x;
		for(int j = q; j >>= 1; ) val[p][j] = bin_op(val[p][j << 1], val[p][j << 1 | 1]);
		for(int i = p; i >>= 1; ){
			val[i][q] = bin_op(val[i << 1][q], val[i << 1 | 1][q]);
			for(int j = q; j >>= 1; ) val[i][j] = bin_op(val[i][j << 1], val[i][j << 1 | 1]);
		}
	}
	T query(int pl, int ql, int pr, int qr){
		if(pl >= pr || ql >= qr) return id;
		T res = id;
		for(int il = pl + n, ir = pr + n; il < ir; il >>= 1, ir >>= 1){
			if(il & 1){
				for(int jl = ql + m, jr = qr + m; jl < jr; jl >>= 1, jr >>= 1){
					if(jl & 1) res = bin_op(res, val[il][jl ++]);
					if(jr & 1) res = bin_op(res, val[il][-- jr]);
				}
				++ il;
			}
			if(ir & 1){
				-- ir;
				for(int jl = ql + m, jr = qr + m; jl < jr; jl >>= 1, jr >>= 1){
					if(jl & 1) res = bin_op(res, val[ir][jl ++]);
					if(jr & 1) res = bin_op(res, val[ir][-- jr]);
				}
			}
		}
		return res;
	}
};

// 156485479_3_2_4
// Recursive Lazy Segment Tree
// Credit: https://codeforces.com/blog/entry/65278
// O(n) preprocessing, O(log n) per query
struct recursive_segment_tree{
	using L = long long; // Lazy type
	using Q = long long; // Query type
	L apply_lazy(const L &lazy, const L &x, array<int, 2> r, array<int, 2> rr){ // r is a subset of rr
		return lazy + x;
	} // update lazy node representing r with rr
	Q merge(const Q &lval, const Q &rval, int l, int m, int r){
		return lval + rval;
	} // merge two nodes representing the intervals [l, m) and [m, r)
	Q apply(const Q &val, const L &x, array<int, 2> r, array<int, 2> rr){ // r is a subset of rr
		return val + x * (r[1] - r[0]);
	} // apply to node representing r lazy node representing rr
	pair<L, Q> id{0, 0};
	Q init(int p){
		return id.second;
	}

	int n;
	vector<L> lazy;
	vector<Q> val;
	void push(int u, int l, int r){ // push the internal node u
		if(lazy[u] != id.first && u + 1 < n << 1){
			int m = l + (r - l >> 1), v = u + 1, w = u + (m - l << 1);
			val[v] = apply(val[v], lazy[u], {l, m}, {l, r});
			lazy[v] = apply_lazy(lazy[v], lazy[u], {l, m}, {l, r});
			val[w] = apply(val[w], lazy[u], {m, r}, {l, r});
			lazy[w] = apply_lazy(lazy[w], lazy[u], {m, r}, {l, r});
			lazy[u] = id.first;
		}
	}
	void refresh(int u, int l, int r){
		if(u + 1 < n << 1){
			int m = l + (r - l >> 1), v = u + 1, w = u + (m - l << 1);
			val[u] = merge(val[v], val[w], l, m, r);
		}
		if(lazy[u] != id.first) val[u] = apply(val[u], lazy[u], {l, r}, {l, r});
	}
	template<typename IT>
	recursive_segment_tree(IT begin, IT end): n(end - begin), lazy(n << 1, id.first), val(n << 1, id.second){
		build(begin, end, 0, 0, n);
	}
	recursive_segment_tree(int n): n(n), lazy(n << 1, id.first), val(n << 1, id.second){
		build(0, 0, n);
	}
	template<typename IT>
	void build(IT begin, IT end, int u, int l, int r){
		if(l + 1 == r) val[u] = *begin;
		else{
			int m = l + (r - l >> 1), v = u + 1, w = u + (m - l << 1);
			IT inter = begin + (end - begin >> 1);
			build(begin, inter, v, l, m), build(inter, end, w, m, r);
			val[u] = merge(val[v], val[w], l, m, r);
		}
	}
	void build(int u, int l, int r){
		if(l + 1 == r) val[u] = init(l);
		else{
			int m = l + (r - l >> 1), v = u + 1, w = u + (m - l << 1);
			build(v, l, m), build(w, m, r);
			val[u] = merge(val[v], val[w], l, m, r);
		}
	}
	template<bool First = true>
	void update(int ql, int qr, L x, int u = 0, int l = 0, int r = numeric_limits<int>::max()){
		if(First) r = n;
		if(qr <= l || r <= ql) return;
		if(ql <= l && r <= qr) val[u] = apply(val[u], x, {l, r}, {ql, qr}), lazy[u] = apply_lazy(lazy[u], x, {l, r}, {ql, qr});
		else{
			push(u, l, r);
			int m = l + (r - l >> 1), v = u + 1, w = u + (m - l << 1);
			update<false>(ql, qr, x, v, l, m), update<false>(ql, qr, x, w, m, r);
			refresh(u, l, r);
		}
	} // Apply x to values at [ql, qr)
	template<bool First = true>
	Q query(int ql, int qr, int u = 0, int l = 0, int r = numeric_limits<int>::max()){
		if(First) r = n;
		if(qr <= l || r <= ql) return id.second;
		if(ql <= l && r <= qr) return val[u];
		push(u, l, r);
		int m = l + (r - l >> 1), v = u + 1, w = u + (m - l << 1);
		return merge(query<false>(ql, qr, v, l, m), query<false>(ql, qr, w, m, r), max(ql, l), clamp(m, ql, qr), min(qr, r));
	} // Get the query result for [ql, qr)
};


// 156485479_3_2_5_1
// Iterative Lazy Segment Tree
// O(n) processing, O(log n) per query
// Credit: https://codeforces.com/blog/entry/18051
struct lazy_segment_tree{
	using L = long long; // Lazy type
	using Q = long long; // Query type
	L apply_lazy(const L &lazy, const L &x, array<int, 2> r, array<int, 2> rr){ // r is a subset of rr
		return lazy + x;
	} // update lazy node representing r with rr
	Q merge(const Q &lval, const Q &rval, int l, int m, int r){
		return lval + rval;
	} // merge two nodes representing the intervals [l, m) and [m, r)
	Q apply(const Q &val, const L &x, array<int, 2> r, array<int, 2> rr){ // r is a subset of r
		return val + x * (r[1] - r[0]);
	} // apply to node representing r lazy node representing rr
	pair<L, Q> id{0, 0};
	Q init(int p){
		return id.second;
	}

	int n, h;
	vector<int> roots;
	vector<array<int, 2>> range;
	vector<L> lazy;
	vector<Q> val;
	void init_range(){
		for(int i = n; i < n << 1; ++ i) range[i] = {i - n, i - n + 1};
		for(int i = n - 1; i > 0; -- i) range[i] = {range[i << 1][0], range[i << 1 | 1][1]};
	}
	void push(int u){ // push the internal node u
		if(lazy[u] != id.first){
			val[u << 1] = apply(val[u << 1], lazy[u], range[u << 1], range[u]);
			lazy[u << 1] = apply_lazy(lazy[u << 1], lazy[u], range[u << 1], range[u]);
			val[u << 1 | 1] = apply(val[u << 1 | 1], lazy[u], range[u << 1 | 1], range[u]);
			lazy[u << 1 | 1] = apply_lazy(lazy[u << 1 | 1], lazy[u], range[u << 1 | 1], range[u]);
			lazy[u] = id.first;
		}
	}
	void push(int l, int r){ // push the range [l, r)
		int s = h;
		for(l += n, r += n - 1; s > 0; -- s) for(int i = l >> s; i <= r >> s; ++ i) push(i);
	}
	void refresh(int u){
		if(u < n) val[u] = merge(val[u << 1], val[u << 1 | 1], range[u << 1][0], range[u << 1][1], range[u << 1 | 1][1]);
		if(lazy[u] != id.first) val[u] = apply(val[u], lazy[u], range[u], range[u]);
	}
	void build(int l, int r){
		for(l += n, r += n - 1; l > 1; ){
			l >>= 1, r >>= 1;
			for(int i = r; i >= l; -- i) refresh(i);
		}
	}
	template<typename IT>
	lazy_segment_tree(IT begin, IT end): n(end - begin), h(__lg(n) + 1), range(n << 1), lazy(n << 1, id.first), val(n << 1, id.second){
		init_range();
		copy(begin, end, val.begin() + n);
		build(0, n);
	}
	lazy_segment_tree(int n): n(n), h(__lg(n) + 1), range(n << 1), lazy(n << 1, id.first), val(n << 1, id.second){
		init_range();
		for(int i = n; i < n << 1; ++ i) val[i] = init(i - n);
		build(0, n);
	}
	void init_roots(){
		vector<int> roots_r;
		for(int l = n, r = n << 1; l < r; l >>= 1, r >>= 1){
			if(l & 1) roots.push_back(l ++);
			if(r & 1) roots_r.push_back(-- r);
		}
		roots.insert(roots.end(), roots_r.rbegin(), roots_r.rend());
	}
	void update(int ql, int qr, L x){
		if(ql >= qr || x == id.first) return;
		push(ql, ql + 1), push(qr - 1, qr);
		bool cl = false, cr = false;
		int l = ql + n, r = qr + n;
		for(; l < r; l >>= 1, r >>= 1){
			if(cl) refresh(l - 1);
			if(cr) refresh(r);
			if(l & 1){
				val[l] = apply(val[l], x, range[l], {ql, qr});
				if(l < n) lazy[l] = apply_lazy(lazy[l], x, range[l], {ql, qr});
				++ l;
				cl = true;
			}
			if(r & 1){
				-- r;
				val[r] = apply(val[r], x, range[r], {ql, qr});
				if(r < n) lazy[r] = apply_lazy(lazy[r], x, range[r], {ql, qr});
				cr = true;
			}
		}
		for(-- l; r > 0; l >>= 1, r >>= 1){
			if(cl) refresh(l);
			if(cr && (!cl || l != r)) refresh(r);
		}
	}
	Q query(int ql, int qr){
		if(ql >= qr) return id.second;
		push(ql, ql + 1), push(qr - 1, qr);
		int mid;
		Q res_l = id.second, res_r = id.second;
		for(int l = ql + n, r = qr + n; l < r; l >>= 1, r >>= 1){
			if(l & 1) res_l = merge(res_l, val[l], ql, range[l][0], range[l][1]), mid = range[l][1], ++ l;
			if(r & 1) -- r, res_r = merge(val[r], res_r, range[r][0], range[r][1], qr), mid = range[r][0];
		}
		return merge(res_l, res_r, ql, mid, qr);
	}
	void print(){
		//auto format_val = [&](auto x)->string{ return x; };
		//auto format_lazy = [&](auto x)->string{ return x; };
		for(int u = 0; u < 2 * n; ++ u){
			//cout << "Node " << u << " represent [" << range[u][0] << ", " << range[u][1] << "), val = " << format_val(val[u]) << ", lazy = " << format_lazy(lazy[u]) << "\n";
		}
	}
};

// 156485479_3_2_5_2
// Dynamic Lazy Segment Tree
// O(1) or O(n) processing, O(log L) or O(log n) per query
struct dynamic_lazy_segment_tree{
	using B = int; // Base coordinate type
	using L = long long; // Lazy type
	using Q = long long; // Query type
	L apply_lazy(const L &lazy, const L &x, array<B, 2> r, array<B, 2> rr){ // r is a subset of rr
		return lazy + x;
	} // update lazy node representing r with rr
	Q merge(const Q &lval, const Q &rval, B l, B m, B r){
		return lval + rval;
	} // merge two nodes representing the intervals [l, m) and [m, r)
	Q apply(const Q &val, const L &x, array<B, 2> r, array<B, 2> rr){ // r is a subset of rr
		return val + x * (r[1] - r[0]);
	} // apply to node representing r lazy node representing rr
	pair<L, Q> id{0, 0};
	Q init(B l, B r){
		return id.second;
	}

	dynamic_lazy_segment_tree *l = 0, *r = 0;
	B low, high;
	L lazy = id.first;
	Q val;
	dynamic_lazy_segment_tree(B low, B high): low(low), high(high), val(init(low, high)){ }
	template<typename IT>
	dynamic_lazy_segment_tree(IT begin, IT end, B low, B high): low(low), high(high){
		assert(end - begin == high - low);
		if(high - low > 1){
			IT inter = begin + (end - begin >> 1);
			B mid = low + (high - low >> 1);
			l = new dynamic_lazy_segment_tree(begin, inter, low, mid);
			r = new dynamic_lazy_segment_tree(inter, end, mid, high);
			val = merge(l->val, r->val, low, mid, high);
		}
		else val = *begin;
	}
	~dynamic_lazy_segment_tree(){
		delete l;
		delete r;
	}
	void push(){
		if(!l){
			B mid = low + (high - low >> 1);
			l = new dynamic_lazy_segment_tree(low, mid);
			r = new dynamic_lazy_segment_tree(mid, high);
		}
		if(lazy != id.first){
			l->update(low, high, lazy);
			r->update(low, high, lazy);
			lazy = id.first;
		}
	}
	void update(B ql, B qr, L x){
		if(qr <= low || high <= ql || x == id.first) return;
		if(ql <= low && high <= qr){
			lazy = apply_lazy(lazy, x, {low, high}, {ql, qr});
			val = apply(val, x, {low, high}, {ql, qr});
		}
		else{
			push();
			l->update(ql, qr, x);
			r->update(ql, qr, x);
			B mid = low + (high - low >> 1);
			val = merge(l->val, r->val, low, mid, high);
		}
	}
	Q query(B ql, B qr){
		if(qr <= low || high <= ql) return id.second;
		if(ql <= low && high <= qr) return val;
		push();
		B mid = clamp(low + (high - low >> 1), ql, qr);
		return merge(l->query(ql, qr), r->query(ql, qr), max(low, ql), clamp(mid, ql, qr), min(high, qr));
	}
};

// 156485479_3_2_6
// Persistent Segment Tree
// O(n) preprocessing, O(log n) per query
struct persistent_segment_tree{
	using B = int; // Base Coordinate Type
	using Q = int; // Query Type
	Q merge(Q lval, Q rval, B l, B m, B r){
		return lval + rval;
	}
	Q id = {};
	Q init(B l, B r){
		return id;
	}

	B n;
	struct node{
		node *l, *r;
		Q val;
		~node(){
			delete l;
			delete r;
		}
	};
	vector<node *> state;
	template<typename IT>
	persistent_segment_tree(IT begin, IT end): n(end - begin){
		function<node *(IT, IT, B, B)> bulid = [&](IT begin, IT end, B low, B high){
			if(high - low == 1) return new node{0, 0, *begin};
			IT inter = begin + (end - begin >> 1);
			B mid = low + (high - low >> 1);
			node *l = build(begin, inter, low, mid), *r = build(inter, end, mid, high);
			return new node{l, r, merge(l->val, r->val, low, mid, high)};
		};
		state.push_back(build(begin, end, 0, n));
	}
	persistent_segment_tree(B n): n(n), state({new node{0, 0, init(0, n)}}){ }
	void extend(node *u, B l, B r){
		if(!u->l){
			B m = l + (r - l >> 1);
			u->l = new node{0, 0, init(l, m)}, u->r = new node{0, 0, init(m, r)};
		}
	}
	template<bool First = true>
	node *update(B p, Q x, node *u, B l = 0, B r = numeric_limits<B>::max()){
		if(First) r = n;
		if(r - l == 1) u = new node{0, 0, x};
		else{
			extend(u, l, r);
			u = new node{u->l, u->r, u->val};
			B m = l + (r - l >> 1);
			if(p < m) u->l = update<false>(p, x, u->l, l, m);
			else u->r = update<false>(p, x, u->r, m, r);
			u->val = merge(u->l->val, u->r->val, l, m, r);
		}
		if(First) state.push_back(u);
		return u;
	}
	template<bool First = true>
	Q query(B ql, B qr, node *u, B l = 0, B r = numeric_limits<B>::max()){
		if(First) r = n;
		if(qr <= l || r <= ql) return id;
		if(ql <= l && r <= qr) return u->val;
		extend(u, l, r);
		B m = l + (r - l >> 1);
		return merge(query<false>(ql, qr, u->l, l, m), query<false>(ql, qr, u->r, m, r), max(ql, l), clamp(m, ql, qr), min(qr, r));
	}
	// Two functions below assume the indices are partitioned ..., true, false, ...
	template<typename Pred, bool First = true>
	B partition_point_pref(Pred p, node *u, Q pval = {}, B l = 0, B r = numeric_limits<B>::max()){
		if(First){
			if(p(n, u->val)) return n;
			r = n, pval = id;
		}
		if(r - l == 1) return l;
		extend(u, l, r);
		B m = l + (r - l >> 1);
		if(p(m, merge(pval, u->l->val, 0, l, m))) return partition_point_pref<Pred, false>(p, u->r, merge(pval, u->l->val, 0, l, m), m, r);
		else return partition_point_pref<Pred, false>(p, u->l, pval, l, m);
	} // Returns the largest i with p(i, prefsum[0, i)) = true), 0 < i <= n, or 0 if all of them evalutes to false
	template<typename Pred, bool First = true>
	B partition_point_suff(Pred p, node *u, Q sval = {}, B l = 0, B r = numeric_limits<B>::max()){
		if(First){
			if(!p(0, u->val)) return 0;
			r = n, sval = id;
		}
		if(r - l == 1) return r;
		extend(u, l, r);
		B m = l + (r - l >> 1);
		if(p(m, merge(u->r->val, sval, m, r, n))) return partition_point_suff<Pred, false>(p, u->r, sval, m, r);
		else partition_point_suff<Pred, false>(p, u->l, merge(u->r->val, sval, m, r, n), l, m);
	} // Returns the smallest i with p(i, suffsum[i, n)) = false), 0 <= i < n, or n if all of them evalutes to true
};

// 156485479_3_3_1
// Fenwick Tree
// Only works on a commutative group
// O(n log n) preprocessing, O(log n) per query
template<typename T, typename BO, typename IO>
struct fenwick_tree{
	int n;
	BO bin_op;
	IO inv_op;
	T id;
	vector<T> val;
	template<typename IT>
	fenwick_tree(IT begin, IT end, BO bin_op, IO inv_op, T id): n(end - begin), bin_op(bin_op), inv_op(inv_op), id(id), val(n + 1, id){
		for(int i = 0; i < n; ++ i) update(i, *(begin ++));
	}
	fenwick_tree(int n, BO bin_op, IO inv_op, T id): n(n), bin_op(bin_op), inv_op(inv_op), id(id), val(n + 1, id){ }
	fenwick_tree(){ }
	template<bool increment = true>
	void update(int p, T x){
		if(!increment) x = inv_op(x, query(p, p + 1));
		for(++ p; p <= n; p += p & -p) val[p] = bin_op(val[p], x);
	}
	T sum(int p){
		T res = id;
		for(++ p; p > 0; p -= p & -p) res = bin_op(res, val[p]);
		return res;
	}
	T query(int l, int r){
		return inv_op(sum(r - 1), sum(l - 1));
	}
};

// 156485479_3_3_2
// Fenwick Tree Supporting Range Queries of The Same Type
// O(n log n) preprocessing, O(log n) per query
// Requires fenwick_tree
template<typename T, typename BO, typename IO, typename MO>
struct range_fenwick_tree{
	fenwick<T, BO, IO> tr1, tr2;
	BO bin_op;
	IO inv_op;
	MO multi_op;
	T id;
	range_fenwick_tree(int n, BO bin_op, IO inv_op, MO multi_op, T id):
		tr1(n, bin_op, inv_op, id),
		tr2(n, bin_op, inv_op, id),
		bin_op(bin_op), inv_op(inv_op), id(id){}
	void update(int l, int r, T x){
		tr1.update(l, x);
		tr1.update(r, inv_op(id, x));
		tr2.update(l, multi_op(x, l - 1));
		tr2.update(r, inv_op(id, multi_op(x, r - 1)));
	}
	T sum(int p){
		return inv_op(multi_op(tr1.sum(p), p), tr2.sum(p));
	}
	T query(int l, int r){
		return inv_op(sum(r - 1), sum(l - 1));
	}
};

// 156485479_3_3_3
// 2D Fenwick Tree ( Only for Commutative Group )
// O(nm log nm) preprocessing, O(log n log m) per query
template<typename T, typename BO, typename IO>
struct fenwick_tree_2d{
	int n, m;
	BO bin_op;
	IO inv_op;
	T id;
	vector<vector<T>> val;
	fenwick_tree_2d(const vector<vector<T>> &arr, BO bin_op, IO inv_op, T id): n(arr.size()), m(arr[0].size()), bin_op(bin_op), inv_op(inv_op), id(id), val(n + 1, vector<T>(m + 1)){
		for(int i = 0; i < n; ++ i) for(int j = 0; j < m; ++ j) update(i, j, arr[i][j]);
	}
	template<bool increment = true>
	void update(int p, int q, T x){
		if(!increment) x = inv_op(x, query(p, q, p + 1, q + 1));
		++ p, ++ q;
		for(int i = p; i <= n; i += i & -i) for(int j = q; j <= m; j += j & -j) val[i][j] = bin_op(val[i][j], x);
	}
	T sum(int p, int q){
		T res = id;
		++ p, ++ q;
		for(int i = p; i > 0; i -= i & -i) for(int j = q; j > 0; j -= j & -j) res = bin_op(res, val[i][j]);
		return res;
	}
	T query(int pl, int ql, int pr, int qr){
		-- pl, -- ql, -- pr, -- qr;
		return inv_op(bin_op(sum(pr, qr), sum(pl, ql)), bin_op(sum(pr, ql), sum(pl, qr)));
	}
};

// 156485479_3_4
// Wavelet Tree
// O(L log n) preprocessing, O(log n) per query
template<typename T>
struct node{
	int n;
	T low, high;
	node *l = 0, *r = 0;
	vector<int> freq;
	template<typename IT, typename Compare>
	node(IT begin, IT end, T low, T high, Compare cmp): n(end - begin), low(low), high(high){
		if(!n) return;
		if(low + 1 == high) return;
		T mid = low + (high - low >> 1);
		auto pred = [&](T x){ return cmp(x, mid); };
		freq.reserve(n + 1);
		freq.push_back(0);
		for(auto it = begin; it != end; ++ it) freq.push_back(freq.back() + pred(*it));
		auto inter = stable_partition(begin, end, pred);
		l = new node(begin, inter, low, mid, cmp);
		r = new node(inter, end, mid, high, cmp);
	}
};
template<typename T, typename Compare = less<>>
struct wavelet_tree{
	int n;
	node<T> *root;
	Compare cmp;
	template<typename IT>
	wavelet_tree(IT begin, IT end, Compare cmp = less<>()): n(end - begin), cmp(cmp){
		root = new node<T>(begin, end, *min_element(begin, end, cmp), *max_element(begin, end, cmp) + 1, cmp);
	}
	// Return the # of elements less than x in the range [ql, qr)
	int count(node<T> *u, int ql, int qr, T x){
		if(ql >= qr || !cmp(u->low, x)) return 0;
		if(!cmp(x, u->high)) return qr - ql;
		int lcnt = u->freq[ql], rcnt = u->freq[qr];
		return count(u->l, lcnt, rcnt, x) + count(u->r, ql - lcnt, qr - rcnt, x);
	}
	//Find the k-th element in the range [ql, qr) ( 0-indexed )
	T k_th(node<T> *u, int ql, int qr, int k){
		assert(0 <= k && k < u->n);
		if(u->low + 1 == u->high) return u->low;
		int lcnt = u->freq[ql], rcnt = u->freq[qr];
		if(k < rcnt - lcnt) return k_th(u->l, lcnt, rcnt, k);
		else return k_th(u->r, ql - lcnt, qr - rcnt, k - (rcnt - lcnt));
	}
	void print(){
		deque<node<T> *> q{root};
		while(!q.empty()){
			auto u = q.front();
			q.pop_front();
			cout << "range = [" << u->low << ", " << u->high << "), frequency: ";
			for(auto x: u->freq){
				cout << x << " ";
			}
			cout << "\n";
			if(u->l) q.push_back(u->l);
			if(u->r) q.push_back(u->r);
		}
	}
};

// 156485479_3_5
// Disjoint Set
// O(alpha(n)) per query where alpha(n) is the inverse ackermann function
// Credit: KACTL
struct disjoint_set{
	vector<int> p;
	disjoint_set(int n): p(n, -1){ }
	bool share(int a, int b){ return root(a) == root(b); }
	int sz(int u){ return -p[root(u)]; }
	int root(int u){ return p[u] < 0 ? u : p[u] = root(p[u]); }
	bool merge(int u, int v){
		u = root(u), v = root(v);
		if(u == v) return false;
		if(p[u] > p[v]) swap(u, v);
		p[u] += p[v], p[v] = u;
		return true;
	}
};
// Persistent Version
// O(log n)
struct rollback_disjoint_set{
	vector<int> p;
	vector<pair<int, int>> log;
	rollback_disjoint_set(int n): p(n, -1){ }
	bool share(int a, int b){ return root(a) == root(b); }
	int sz(int u){ return -p[root(u)]; }
	int root(int u){ return p[u] < 0 ? u : p[u] = root(p[u]); }
	bool merge(int u, int v){
		u = root(u), v = root(v);
		if(u == v) return false;
		if(p[u] > p[v]) swap(u, v);
		log.emplace_back(u, p[u]), log.emplace_back(v, p[v]);
		p[u] += p[v], p[v] = u;
		return true;
	}
	int time(){ return int(log.size()); }
	void reverse_to(int t = 0){
		for(int i = time(); i --> t; ) p[log[i].first] = log[i].second;
		log.resize(t);
	}
};

// 156485479_3_6
// Monotone Stack
// O(1) per operation
template<typename T = int, typename Compare = less<>>
struct monotone_stack: vector<T>{
	T init;
	Compare cmp;
	monotone_stack(T init = 0, Compare cmp = less{}): init(init), cmp(cmp){ }
	T push(T x){
		while(!this->empty() && !cmp(this->back(), x)) this->pop_back();
		this->push_back(x);
		return this->size() == 1 ? init : *next(this->rbegin());
	}
};

// 156485479_3_7
// Distinct Value Query, Less-than-k Query (Offline, Online)
// O(n log n) processing
// Requires fenwick_tree
// TYPE: {0: distinct value query, 1: less-than-k query with numbers in range [0, n), 2: arbitrary range less-than-k query}
template<typename T, int TYPE = 0>
struct offline_less_than_k_query{
	int n;
	vector<pair<T, int>> event;
	vector<tuple<T, int, int, int>> queries;
	vector<T> comp;
	template<typename IT>
	offline_less_than_k_query(IT begin, IT end): n(end - begin), event(n){
		if(TYPE == 0){
			map<T, int> q;
			for(int i = 0; begin != end; ++ begin, ++ i){
				event[i] = {(q.count(*begin) ? q[*begin] : -1), i};
				q[*begin] = i;
			}
		}
		else if(TYPE == 1) for(int i = 0; begin != end; ++ begin, ++ i) event[i] = {*begin, i};
		else{
			comp = {begin, end};
			sort(comp.begin(), comp.end()), comp.erase(unique(comp.begin(), comp.end()), comp.end());
			for(int i = 0; begin != end; ++ begin, ++ i) event[i] = {std::lower_bound(comp.begin(), comp.end(), *begin) - comp.begin(), i};
		}
	}
	void query(int i, int ql, int qr){ // For distinct value query
		assert(!TYPE);
		queries.emplace_back(ql, ql, qr, i);
	}
	void query(int i, int ql, int qr, T k){ // For less-than-k query
		assert(TYPE);
		queries.emplace_back(TYPE == 2 ? std::lower_bound(comp.begin(), comp.end(), k) - comp.begin() : k, ql, qr, i);
	}
	template<typename Action>
	void solve(Action ans){ // ans(index, answer)
		sort(queries.begin(), queries.end()), sort(event.begin(), event.end(), greater<>());
		fenwick_tree tr(n, plus<>(), minus<>(), 0);
		for(auto &[k, ql, qr, i]: queries){
			while(!event.empty() && event.back().first < k){
				tr.update(event.back().second, 1);
				event.pop_back();
			}
			ans(i, tr.query(ql, qr));
		}
	}
};
// Online
// Requires persistent_segment_tree
// TYPE: {0: distinct value query, 1: less-than-k query with numbers in range [0, n), 2: arbitrary range less-than-k query}
template<typename T, int TYPE = 0>
struct less_than_k_query{
	int n;
	vector<node<T> *> p;
	persistent_segment_tree<int, plus<>> tr;
	vector<T> comp;
	template<typename IT>
	less_than_k_query(IT begin, IT end): n(end - begin), p(n + 1), tr(n, plus<>(), 0){
		vector<pair<T, int>> event(n);
		if(TYPE == 0){
			map<T, int> q;
			for(int i = 0; begin != end; ++ begin, ++ i){
				event[i] = {(q.count(*begin) ? q[*begin] : -1), i};
				q[*begin] = i;
			}
		}
		else if(TYPE == 1) for(int i = 0; begin != end; ++ begin, ++ i) event[i] = {*begin, i};
		else{
			comp = {begin, end};
			sort(comp.begin(), comp.end()), comp.erase(unique(comp.begin(), comp.end()), comp.end());
			for(int i = 0; begin != end; ++ begin, ++ i) event[i] = {std::lower_bound(comp.begin(), comp.end(), *begin) - comp.begin(), i};
		}
		sort(event.begin(), event.end(), greater<>());
		tr.reserve(n);
		for(int i = 0; i <= n; ++ i){
			while(!event.empty() && event.back().first < i){
				tr.update(tr.back(), event.back().second, 1);
				event.pop_back();
			}
			p[i] = tr.back();
		}
	}
	// For distinct value query
	int query(int ql, int qr){
		assert(!TYPE);
		return tr.query(p[ql], ql, qr, 0, n);
	}
	// For less-than-k query
	int query(int ql, int qr, int k){
		assert(TYPE);
		return tr.query(p[TYPE == 2 ? std::lower_bound(comp.begin(), comp.end(), k) - comp.begin() : k], ql, qr, 0, n);
	}
};

// 156485479_3_8
// Mo's Algorithm
// O((N + Q) sqrt(N) F) where F is the processing time of ins and del.
// Credit: cp-algorithms.com
template<int B>
struct query{
	int l, r, ind;
	bool operator<(const query &otr) const{
		if(l / B != otr.l / B) return pair<int, int>(l, r) < pair<int, int>(otr.l, otr.r);
		return (l / B & 1) ? (r < otr.r) : (r > otr.r);
	}
};
template<typename T, typename Q, typename I, typename D, typename A>
vector<T> answer_query_offline(vector<Q> query, I ins, D del, A ans){
	sort(query.begin(), query.end());
	vector<T> res(query.size());
	int l = 0, r = 0;
	for(auto q: query){
		while(l > q.l) ins(-- l);
		while(r < q.r) ins(r ++);
		while(l < q.l) del(l ++);
		while(r > q.r) del(-- r);
		res[q.ind] = ans(q);
	}
	return res;
}
/*
	auto ins = [&](int i){
		
	};
	auto del = [&](int i){
		
	};
	auto ans = [&](const auto &q){

	};
*/

// 156485479_3_9
// Treap
// O(log N) per operation
// Credit: KACTL
struct treap{
	using Q = int; // Query Type
	using SS = int; // Subtree Sum Type
	using L = int; // Lazy Type
	struct node{
		node *l = 0, *r = 0;
		Q val;
		SS subtr_val;
		L lazy;
		int priority = rng(), sz = 1, ind = 0;
		node(Q val, L lazy, int ind = 0): val(val), subtr_val(val), lazy(lazy), ind(ind){ }
	};
	node *root = 0;
	void push(node *u){
		if(u && u->lazy){
			if(u->l){
				u->l->lazy += u->lazy;
				u->l->val += u->lazy;
				u->l->subtr_val += u->lazy * u->l->sz;
			}
			if(u->r){
				u->r->lazy += u->lazy;
				u->r->val += u->lazy;
				u->r->subtr_val += u->lazy * u->r->sz;
			}
			u->lazy = 0;
		}
	};
	void refresh(node *u){
		u->sz = (u->l ? u->l->sz : 0) + 1 + (u->r ? u->r->sz : 0);
		u->subtr_val = (u->l ? u->l->subtr_val : 0) + u->val + (u->r ? u->r->subtr_val : 0);
	};
	template<typename IT>
	treap(IT begin, IT end, L lazy_init = 0){
		root = build(begin, end, lazy_init);
	}
	treap(int n, Q val_init = 0, L lazy_init = 0){
		root = build(n, val_init, lazy_init);
	}
	void heapify(node *u){
		if(u){
			node *v = u;
			if(u->l && u->l->priority > v->priority) v = u->l;
			if(u->r && u->r->priority > v->priority) v = u->r;
			if(u != v){
				swap(u->priority, v->priority);
				heapify(v);
			}
		}
	}
	template<typename IT>
	node *build(IT begin, IT end, L lazy_init){
		if(begin == end) return 0;
		IT mid = begin + (end - begin >> 1);
		node *c = new node(*mid, lazy_init);
		c->l = build(begin, mid, lazy_init), c->r = build(mid + 1, end, lazy_init);
		heapify(c), refresh(c);
		return c;
	}
	node *build(int n, Q val_init, L lazy_init){
		if(!n) return 0;
		int m = n >> 1;
		node *c = new node(val_init, lazy_init);
		c->l = build(m, val_init, lazy_init), c->r = build(n - m - 1, val_init, lazy_init);
		heapify(c), refresh(c);
		return c;
	}
	int get_sz(node *u){
		return u ? u->sz : 0;
	}
	pair<node *, node *> split(node* u, Q k){
	// pair<node *, node *> split(node* u, int k){ // For the implicit treap
		if(!u) return { };
		push(u);
		if(u->val >= k){
		// if(get_sz(u->l) >= k){ // For the implicit treap
			auto [a, b] = split(u->l, k);
			u->l = b;
			refresh(u);
			return {a, u};
		}
		else{
			auto [a, b] = split(u->r, k);
			// auto [a, b] = split(u->r, k - get_sz(u->l) - 1); // For the implicit treap
			u->r = a;
			refresh(u);
			return {u, b};
		}
	}
	node *merge(node *u, node *v){
		if(!u || !v) return u ?: v;
		push(u), push(v);
		if(v->priority < u->priority){
			u->r = merge(u->r, v);
			refresh(u);
			return u;
		}
		else{
			v->l = merge(u, v->l);
			refresh(v);
			return v;
		}
	}
	node *insert(node *u, node *t){
	// node *insert(node *u, node *t, int pos){ // For the implicit treap
		if(!u) return t;
		auto [a, b] = split(u, t->val);
		// auto [a, b] = split(u, pos); // For the implicit treap
		return merge(merge(a, t), b);
	}
};
template<typename Action>
void for_each(treap::node *u, Action act){
	if(u){ for_each(u->l, act), act(u), for_each(u->r, act); }
}

// 156485479_3_10
// Splay Tree
// Amortized O(log n) per operation
// Credit: DGC
template<typename pnode>
struct splay_tree{
	pnode ch[2], p;
	bool rev;
	int sz;
	splay_tree(){ ch[0] = ch[1] = p = NULL; rev = 0; sz = 1; }
	friend int get_sz(pnode u) { return u ? u->sz : 0; }
	virtual void update(){
		if(ch[0]) ch[0]->push();
		if(ch[1]) ch[1]->push();
		sz = 1 + get_sz(ch[0]) + get_sz(ch[1]);
	}
	virtual void push(){
		if(rev){
			if(ch[0]) ch[0]->rev ^= 1;
			if(ch[1]) ch[1]->rev ^= 1;
			swap(ch[0], ch[1]); rev = 0;
		}
	}
	int dir(){
		if(!p) return -2; // root of LCT component
		if(p->ch[0] == this) return 0; // left child
		if(p->ch[1] == this) return 1; // right child
		return -1; // root of current splay tree
	}
	bool is_root(){ return dir() < 0; }
	friend void set_link(pnode u, pnode v, int d){
		if(v) v->p = u;
		if(d >= 0) u->ch[d] = v;
	}
	void rotate(){ // assume p and p->p propagated
		assert(!is_root());
		int x = dir(); pnode g = p;
		set_link(g->p, static_cast<pnode>(this), g->dir());
		set_link(g, ch[x^1], x);
		set_link(static_cast<pnode>(this), g, x^1);
		g->update(); update();
	}
	void splay(){ // bring this to top of splay tree
		while(!is_root() && !p->is_root()){
			p->p->push(), p->push(), push();
			dir() == p->dir() ? p->rotate() : rotate();
			rotate();
		}
		if(!is_root()) p->push(), push(), rotate();
		push();
	}
};
struct node: splay_tree<node *>{
	using splay_tree::ch;
	int val;
	long long aux_sum;
	node(): splay_tree(){ }
	void update() override{
		splay_tree::update();
		aux_sum = val;
		if(ch[0]) aux_sum += ch[0]->aux_sum;
		if(ch[1]) aux_sum += ch[1]->aux_sum;
	}
	void update_vsub(node* v, bool add){

	}
	void push() override{ // make sure push fix the node (don't call splay_tree::update)
		splay_tree::push();
	}
};

// 156485479_3_11
// Link Cut Trees
// Amortized O(log n) per operation
// Credit: DGC
// Requires splay_tree
template<typename node>
struct link_cut_tree{
	bool connected(node *u, node *v){ return lca(u, v) != NULL; }
	int depth(node *u){ access(u); return get_sz(u->ch[0]); }
	node *get_root(node *u){ // get root of LCT component
		access(u);
		while(u->ch[0]) u = u->ch[0], u->push();
		return access(u), u;
	}
	node *ancestor(node *u, int k){ // get k-th parent on path to root
		k = depth(u) - k;
		assert(k >= 0);
		for(; ; u->push()){
			int sz = get_sz(u->ch[0]);
			if(sz == k) return access(u), u;
			if(sz < k) k -= sz + 1, u = u->ch[1];
			else u = u->ch[0];
		}
		assert(0);
	}
	node *lca(node *u, node *v){
		if(u == v) return u;
		access(u); access(v);
		if(!u->p) return NULL;
		u->splay(); return u->p ?: u;
	}
	// make u parent of v
	void link(node *u, node *v){ assert(!connected(u, v)); make_root(v); access(u); set_link(v, u, 0); v->update(); }
	// cut u from its parent
	void cut(node *u){ access(u); u->ch[0]->p = NULL; u->ch[0] = NULL; u->update(); }
	void cut(node *u, node *v){ cut(depth(u) > depth(v) ? u : v); }
	void make_root(node *u){ access(u); u->rev ^= 1; access(u); assert(!u->ch[0] && !u->ch[1]); }
	void access(node *u){ // puts u on the preferred path, splay (right subtree is empty)
		for(node *v = u, *pre = NULL; v; v = v->p){
			v->splay(); // now update virtual children
			if(pre) v->update_vsub(pre, false);
			if(v->ch[1]) v->update_vsub(v->ch[1], true);
			v->ch[1] = pre; v->update(); pre = v;
		}
		u->splay(); assert(!u->ch[1]);
	}
	node *operator[](int i){ return &data[i]; }
	int operator[](node *i){ return i - &data[0]; }
	vector<node> data;
	link_cut_tree(int n): data(n){ }
	void print(){
		cout << "-----Current LCT info-----\n";
		for(auto u = 0; u < int(data.size()); ++ u){
			cout << "Node " << u << "\n" << "parent: ";
			if(data[u].p) cout << (*this)[data[u].p];
			cout << "\nSub aux tree size = " << get_sz(&data[u]) << "\nchilds = ";
			if(data[u].ch[0]) cout << (*this)[data[u].ch[0]] << " ";
			else cout << "x ";
			if(data[u].ch[1]) cout << (*this)[data[u].ch[1]];
			else cout << "x";
			// cout << "\nvalue = " << data[u].val << "\nsub aux tree value = " << data[u].aux_sum << "\n\n";
		}
		cout << "--------------------------\n\n";
	}
};

// 156485479_3_12
// Unital Sorter
// O(1) per operation
struct unital_sorter{
	int n, m; // # of items, maximum possible cnt
	vector<int> list, pos, cnt;
	vector<pair<int, int>> bound;
	unital_sorter(int n, int m): n(n), m(m), list(n), pos(n), cnt(n), bound(m + 1, {n, n}){
		bound[0] = {0, n};
		iota(list.begin(), list.end(), 0);
		iota(pos.begin(), pos.end(), 0);
	}
	void insert(int x){
		-- bound[cnt[x]].second;
		-- bound[cnt[x] + 1].first;
		int y = list[bound[cnt[x] ++].second];
		swap(pos[x], pos[y]);
		swap(list[pos[x]], list[pos[y]]);
	}
	void erase(int x){
		int y = list[bound[cnt[x] - 1].second];
		swap(pos[x], pos[y]);
		swap(list[pos[x]], list[pos[y]]);
		++ bound[cnt[x]].first;
		++ bound[-- cnt[x]].second;
	}
	void print(int X, int Y){
		cout << "List = ", copy(list.end() - X, list.end(), ostream_iterator<int>(cout, " ")), cout << "\n";
		cout << "Pos = ", copy(pos.begin(), pos.begin() + X, ostream_iterator<int>(cout, " ")), cout << "\n";
		cout << "Count = ", copy(cnt.begin(), cnt.begin() + X, ostream_iterator<int>(cout, " ")), cout << "\n";
		cout << "Bound = ";
		for(int i = 0; i < Y; ++ i) cout << "(" << bound[i].first << ", " << bound[i].second << ")";
		cout << endl;
	}
};

// 156485479_3_13
// AAA Tree < INCOMPLETE >

// 156485479_3_14
// Bit Trie
// O(n * bit) construction, O(bit) per query
template<typename T = int, int mx = 30>
struct trie{
	vector<array<int, 3>> next{{-1, -1, 0}};
	trie(){ }
	template<typename IT>
	trie(IT begin, IT end){
		for(; begin != end; ++ begin) insert(*begin);
	}
	void insert(T x){
		for(int bit = mx - 1, u = 0; bit >= 0; -- bit){
			if(!~next[u][!!(x & T(1) << bit)]){
				next[u][!!(x & T(1) << bit)] = int(next.size());
				next.push_back({-1, -1, 0});
			}
			u = next[u][!!(x & T(1) << bit)];
			++ next[u][2];
		}
	}
	void erase(T x){
		for(int bit = mx - 1, u = 0; bit >= 0; -- bit){
			u = next[u][!!(x & T(1) << bit)];
			-- next[u][2];
		}
	}
	T max_xor(T x){
		T res = 0;
		for(int bit = mx - 1, u = 0; bit >= 0; -- bit){
			if(!~next[u][!(x & T(1) << bit)] || !next[next[u][!(x & T(1) << bit)]][2]) u = next[u][!!(x & T(1) << bit)];
			else{
				res += T(1) << bit;
				u = next[u][!(x & T(1) << bit)];
			}
		}
		return res;
	}
};

// 156485479_3_15
// Query Tree
// If we have a data structure supporting insertion in true O(T(N)), we can delete
// from it in true O(T(N) log N) offline.
// Credit: https://cp-algorithms.com/data_structures/deleting_in_log_n.html
// The data structure must support .time() and .reverse_to(t)
template<typename rollback_DS, typename Element>
struct querytree{
	int n; // query count
	vector<vector<Element>> val;
	rollback_DS DS;
	querytree(int n, auto &CL): n(n), val(n << 1), DS(CL){ }
	void insert(int ql, int qr, Element e, int u, int l, int r){
		if(qr <= l || r <= ql) return;
		if(ql <= l && r <= qr){
			val[u].push_back(e);
			return;
		}
		else{
			int m = l + (r - l >> 1), v = u + 1, w = u + (m - l << 1);
			insert(ql, qr, e, v, l, m), insert(ql, qr, e, w, m, r);
		}
	}

	void insert(Element e){
		DS.push(e, 0, mx);
	} // Clearify how to modify the DS
	template<typename T>
	auto query(T x){
		return DS.query(x, 0, mx);
	} // Clearify how to get the answer

	template<typename B, typename T = int>
	vector<T> solve(const vector<B> &query_at, T init = 0){
		vector<T> res(n, init);
		function<void(int, int, int)> dfs = [&](int u, int l, int r){
			int timer = DS.time();
			for(auto e: val[u]) insert(e);
			if(r - l == 1) res[l] = query(query_at[l]);
			else{
				int m = l + (r - l >> 1), v = u + 1, w = u + (m - l << 1);
				dfs(v, l, m), dfs(w, m, r);
			}
			DS.reverse_to(timer);
		};
		dfs(0, 0, n);
		return res;
	}
};

// Example of DS: rollback lichao tree
// Used on Petrozavodsk Programming Camp 2020 Summer H3
const long long inf = numeric_limits<long long>::max() / 4 * 3;
struct line{
	long long a, b;
	line(long long a = 0, long long b = 0): a(a), b(b){ }
	long long eval(long long x){
		if(x == inf || b == inf || a == inf) return inf;
		x -= a, x *= x, x *= x;
		return x + b;
	}
	bool majorize(line X, long long L, long long R){
		return eval(L) <= X.eval(L) && eval(R) <= X.eval(R);
	}
};
const long long mx = 50010;
struct rollback_lichao{
	rollback_lichao *l = 0, *r = 0;
	line S;
	vector<tuple<int, rollback_lichao *, line>> &CL;
	rollback_lichao(auto &CL): CL(CL), S(line(0, inf)){ }
	~rollback_lichao(){
		delete l;
		delete r;
	}
	void mc(int i){
		if(i){
			if(!r){
				r = new rollback_lichao(CL);
				CL.emplace_back(2, this, S);
			}
		}
		else{
			if(!l){
				l = new rollback_lichao(CL);
				CL.emplace_back(1, this, S);
			}
		}
	}
	long long query(int X, int L, int R){
		long long ans = S.eval(X);
		int M = L + (R - L >> 1);
		if(X < M) return min(ans, l ? l->query(X, L, M) : inf);
		else return min(ans, r ? r->query(X, M, R) : inf);
	}
	void push(line X, int L, int R){
		if(X.majorize(S, L, R)) CL.emplace_back(0, this, S), swap(X, S);
		if(S.majorize(X, L, R)) return;
		if(S.eval(L) > X.eval(L)) CL.emplace_back(0, this, S), swap(X, S);
		int M = L + (R - L >> 1);
		if(X.eval(M) < S.eval(M)) CL.emplace_back(0, this, S), swap(X, S), mc(0), l->push(X, L, M);
		else mc(1), r->push(X, M, R);
	}
	int time(){
		return int(CL.size());
	}
	void reverse_to(int t){
		for(int i = time(); i --> t; ){
			auto [type, u, S] = CL.back(); CL.pop_back();
			if(!type) u->S = S;
			else if(type == 1){
				delete u->l;
				u->l = 0;
			}
			else{
				delete u->r;
				u->r = 0;
			}
		}
		CL.resize(t);
	}
};

// 156485479_4_1
// Strongly Connected Component ( Tarjan's Algorithm ) / Condensation
// Processes SCCs in reverse topological order
// O(V + E)
template<typename Graph, typename Process_SCC>
int SCC(const Graph &adj, Process_SCC f){
	int n = int(adj.size());
	vector<int> val(n), comp(n, -1), z, cur;
	int timer = 0, ncomps = 0;
	function<int(int)> dfs = [&](int u){
		int low = val[u] = ++ timer, v;
		z.push_back(u);
		for(auto v: adj[u]) if(comp[v] < 0) low = min(low, val[v] ?: dfs(v));
		if(low == val[u]){
			do{
				v = z.back(); z.pop_back();
				comp[v] = ncomps;
				cur.push_back(v);
			}while(v != u);
			f(cur);
			cur.clear();
			++ ncomps;
		}
		return val[u] = low;
	};
	for(int u = 0; u < n; ++ u) if(comp[u] < 0) dfs(u);
	return ncomps;
}
template<typename Graph>
tuple<vector<vector<int>>, vector<int>, vector<vector<int>>>
condensation(const Graph &adj){
	vector<vector<int>> scc;
	int n = int(adj.size()), k = SCC(adj, [&](const vector<int> &c){ scc.push_back(c); });
	reverse(scc.begin(), scc.end());
	vector<int> numb(n);
	vector<vector<int>> cadj(k);
	for(int i = 0; i < k; ++ i) for(auto u: scc[i]) numb[u] = i;
	for(int u = 0; u < n; ++ u) for(auto v: adj[u]) if(numb[u] ^ numb[v]) cadj[numb[u]].push_back(numb[v]);
	return {scc, numb, cadj};
}
// Strong Augmentation
// O(V + E)
// Requires SCC
template<typename Graph>
vector<array<int, 2>> strong_augmentation(const Graph &adj){
	auto [scc, numb, cadj] = condensation(adj);
	if(int(scc.size()) == 1) return {};
	int n = int(adj.size()), k = int(scc.size());
	vector<int> is_src(k, true), is_snk(k), src, snk;
	for(int u = 0; u < k; ++ u) for(auto v: cadj[u]) is_src[v] = false;
	for(int u = 0; u < k; ++ u){
		if(cadj[u].empty()) is_snk[u] = true, snk.push_back(u);
		if(is_src[u]) src.push_back(u);
	}
	int s = int(src.size()), t = int(snk.size());
	bool is_rev = false;
	if(s > t){
		is_rev = true;
		vector<vector<int>> rcadj(k);
		for(int u = 0; u < k; ++ u) for(auto v: cadj[u]) rcadj[v].push_back(u);
		swap(cadj, rcadj), swap(src, snk), swap(is_src, is_snk), swap(s, t);
	}
	vector<int> vis(k), ord_src, ord_snk;
	for(auto u: src) if(!vis[u]){
		int pu = -1;
		function<void(int)> get_sink = [&](int u){
			vis[u] = true;
			if(is_snk[u]) pu = u;
			for(auto v: cadj[u]) if(!~pu && !vis[v]) get_sink(v);
		};
		get_sink(u);
		if(~pu) ord_src.push_back(u), ord_snk.push_back(pu), vis[u] = vis[pu] = 2;
	}
	int cnt = int(ord_src.size());
	for(auto u: src) if(vis[u] ^ 2) ord_src.push_back(u);
	for(auto v: snk) if(vis[v] ^ 2) ord_snk.push_back(v);
	vector<array<int, 2>> res;
	auto add_edge = [&](int u, int v){
		if(is_rev) swap(u, v);
		res.push_back({scc[u][0], scc[v][0]});
	};
	for(int i = 1; i < cnt; ++ i) add_edge(ord_snk[i - 1], ord_src[i]);
	for(int i = cnt; i < s; ++ i) add_edge(ord_snk[i], ord_src[i]);
	if(s ^ t){
		add_edge(ord_snk[cnt - 1], ord_snk[s]);
		for(int i = s + 1; i < t; ++ i) add_edge(ord_snk[i - 1], ord_snk[i]);
		add_edge(ord_snk.back(), ord_src[0]);
	}
	else add_edge(ord_snk[cnt - 1], ord_src[0]);
	return res;
}

// 156485479_4_2
// Biconnected Components / adj[u]: list of [vertex, edgenum]
// O(V + E)
template<typename Graph, typename Process_BCC, typename Process_Bridge = function<void(int, int, int)>>
int BCC(const Graph &adj, Process_BCC f, Process_Bridge g = [](int u, int v, int e){ }){
	int n = int(adj.size());
	vector<int> num(n), st;
	int timer = 0, ncomps = 0;
	function<int(int, int)> dfs = [&](int u, int pe){
		int me = num[u] = ++ timer, top = me;
		for(auto [v, e]: adj[u]) if(e != pe){
			if(num[v]){
				top = min(top, num[v]);
				if(num[v] < me) st.push_back(e);
			}
			else{
				int si = int(st.size()), up = dfs(v, e);
				top = min(top, up);
				if(up == me){
					st.push_back(e);
					f(vector<int>(st.begin() + si, st.end())); // Processes edgelist
					st.resize(si);
					++ ncomps;
				}
				else if(up < me) st.push_back(e);
				else g(u, v, e);
			}
		}
		return top;
	};
	for(int u = 0; u < n; ++ u) if(!num[u]) dfs(u, -1);
	return ncomps;
}

// 156485479_4_3
// Articulation Points / WARNING: f(u) may be called multiple times for the same u.
// O(n + m)
template<typename Graph, typename Process_Articulation_Point>
void articulation_points(const Graph &adj, Process_Articulation_Point f){
	int n = int(adj.size()), timer = 0;
	vector<int> tin(n, -1), low(n);
	function<void(int, int)> dfs = [&](int u, int p){
		tin[u] = low[u] = timer ++;
		int child = 0;
		for(auto v: adj[u]){
			if(v != p){
				if(~tin[v]) low[u] = min(low[u], tin[v]);
				else{
					dfs(v, u);
					low[u] = min(low[u], low[v]);
					if(low[v] >= tin[u] && ~p) f(u);
					++ child;
				}
			}
		}
		if(!~p && child > 1) f(u);
	};
	for(int u = 0; u < n; ++ u) if(!~tin[u]) dfs(u, -1);
}

// 156485479_4_4_1
// Dinic's Maximum Flow Algorithm
// O(V^2 E) ( O(E min(V^2/3, E^1/2)) for unit network )
template<typename T>
struct flow_network{
	static constexpr T eps = (T)1e-9;
	int n;
	vector<vector<int>> adj;
	struct Edge{
		int from, to;
		T capacity, flow;
	};
	vector<Edge> edge;
	int source, sink;
	T flow = 0;
	flow_network(int n, int source, int sink): n(n), source(source), sink(sink), adj(n){ }
	void clear(){
		for(auto &e: edge) e.flow = 0;
		flow = 0;
	}
	int insert(int from, int to, T forward_cap, T backward_cap){
		int ind = (int)edge.size();
		adj[from].push_back(ind);
		edge.push_back({from, to, forward_cap, 0});
		adj[to].push_back(ind + 1);
		edge.push_back({to, from, backward_cap, 0});
		return ind;
	}
	void add_flow(int i, T f){
		edge[i].flow += f;
		edge[i ^ 1].flow -= f;
	}
};
template<typename T>
struct dinic{
	static constexpr T inf = numeric_limits<T>::max();
	flow_network<T> &g;
	vector<int> ptr, level, q;
	dinic(flow_network<T> &g): g(g), ptr(g.n), level(g.n), q(g.n){ }
	bool bfs(){
		fill(level.begin(), level.end(), -1);
		q[0] = g.sink;
		level[g.sink] = 0;
		int beg = 0, end = 1;
		while(beg < end){
			int i = q[beg ++];
			for(auto ind: g.adj[i]){
				auto &e = g.edge[ind];
				auto &re = g.edge[ind ^ 1];
				if(re.capacity - re.flow > g.eps && level[e.to] == -1){
					level[e.to] = level[i] + 1;
					if(e.to == g.source) return true;
					q[end ++] = e.to;
				}
			}
		}
		return false;
	}
	T dfs(int u, T w){
		if(u == g.sink) return w;
		int &j = ptr[u];
		while(j >= 0){
			int ind = g.adj[u][j];
			auto &e = g.edge[ind];
			if(e.capacity - e.flow > g.eps && level[e.to] == level[u] - 1){
				T F = dfs(e.to, min(e.capacity - e.flow, w));
				if(F > g.eps){
					g.add_flow(ind, F);
					return F;
				}
			}
			-- j;
		}
		return 0;
	}
	T max_flow(){
		while(bfs()){
			for(int i = 0; i < g.n; ++ i) ptr[i] = int(g.adj[i].size()) - 1;
			T sum = 0;
			while(1){
				T add = dfs(g.source, inf);
				if(add <= g.eps) break;
				sum += add;
			}
			if(sum <= g.eps) break;
			g.flow += sum;
		}
		return g.flow;
	}
	pair<T, vector<bool>> min_cut(){
		T cut = max_flow();
		vector<bool> res(g.n);
		for(int i = 0; i < g.n; ++ i) res[i] = (level[i] != -1);
		return {cut, res};
	}
};

// 156485479_4_4_2
// Minimum Cost Maximum Flow Algorithm
// O(Augmenting Paths) * O(SPFA)
// Credit: Tourist
template<typename T, typename C>
struct mcmf{
	static constexpr T eps = (T) 1e-9;
	struct Edge{
		int from, to;
		T capacity, flow;
		C cost;
	};
	vector<vector<int>> adj;
	vector<Edge> edge;
	vector<C> d;
	vector<bool> in_queue;
	vector<int> q, pe;
	int n, source, sink;
	T flow = 0;
	C cost = 0;
	mcmf(int n, int source, int sink): n(n), source(source), sink(sink), adj(n), d(n), in_queue(n), pe(n){ }
	void clear(){
		for(auto &e: edge) e.flow = 0;
		flow = 0;
	}
	int insert(int from, int to, T forward_cap, T backward_cap, C cost){
		assert(0 <= from && from < n && 0 <= to && to < n);
		int ind = int(edge.size());
		adj[from].push_back((int)edge.size());
		edge.push_back({from, to, forward_cap, 0, cost});
		adj[to].push_back((int)edge.size());
		edge.push_back({to, from, backward_cap, 0, -cost});
		return ind;
	}
	bool expath(){
		fill(d.begin(), d.end(), numeric_limits<C>::max());
		q.clear();
		q.push_back(source);
		d[source] = 0;
		in_queue[source] = true;
		int beg = 0;
		bool found = false;
		while(beg < (int)q.size()){
			int i = q[beg ++];
			if(i == sink) found = true;
			in_queue[i] = false;
			for(int id : adj[i]){
				const auto &e = edge[id];
				if(e.capacity - e.flow > eps && d[i] + e.cost < d[e.to]){
					d[e.to] = d[i] + e.cost;
					pe[e.to] = id;
					if(!in_queue[e.to]){
						q.push_back(e.to);
						in_queue[e.to] = true;
					}
				}
			}
		}
		if(found){
			T push = numeric_limits<T>::max();
			int v = sink;
			while(v != source){
				const auto &e = edge[pe[v]];
				push = min(push, e.capacity - e.flow);
				v = e.from;
			}
			v = sink;
			while(v != source){
				auto &e = edge[pe[v]];
				e.flow += push;
				auto &back = edge[pe[v] ^ 1];
				back.flow -= push;
				v = e.from;
			}
			flow += push;
			cost += push * d[sink];
		}
		return found;
	}
	pair<T, C> get_mcmf(){
		while(expath()){ }
		return {flow, cost};
	}
};

// 156485479_4_4_3
// Simple DFS Bipartite Matching
// Call solve() for maximum matching
// u from the left vertex set is matched with pu[u] on the right (-1 if not matched)
// v from the right vertex set is matched with pv[v] on the left (-1 if not matched)
// O(VE)
struct matching{
	vector<vector<int>> adj;
	vector<int> pu, pv, cur;
	int n, m, flow = 0, id = 0;
	matching(int n, int m): n(n), m(m), pu(n, -1), pv(m, -1), cur(n), adj(n){ }
	int insert(int from, int to){
		adj[from].push_back(to);
		return int(adj[from].size()) - 1;
	}
	bool dfs(int v){
		cur[v] = id;
		for(auto u: adj[v]){
			if(pv[u] == -1){
				pu[v] = u;
				pv[u] = v;
				return true;
			}
		}
		for(auto u: adj[v]){
			if(cur[pv[u]] != id && dfs(pv[u])){
				pu[v] = u;
				pv[u] = v;
				return true;
			}
		}
		return false;
	}
	int solve(){
		while(true){
			++ id;
			int augment = 0;
			for(int u = 0; u < n; ++ u) if(pu[u] == -1 && dfs(u)) ++ augment;
			if(!augment) break;
			flow += augment;
		}
		return flow;
	}
	int run_once(int v){
		if(pu[v] != -1) return 0;
		++ id;
		return dfs(v);
	}
	pair<vector<int>, vector<int>> minimum_vertex_cover(){
		solve();
		vector<int> L, R, visL(n), visR(m);
		function<void(int)> dfs = [&](int u){
			visL[u] = true;
			for(auto v: adj[u]){
				if(v != pu[u] && !visR[v]){
					visR[v] = true;
					if(~pv[v]) dfs(pv[v]);
				}
			}
		};
		for(int u = 0; u < n; ++ u) if(!visL[u] && pu[u] == -1) dfs(u);
		for(int u = 0; u < n; ++ u) if(!visL[u]) L.push_back(u);
		for(int v = 0; v < m; ++ v) if(visR[v]) R.push_back(v);
		return {L, R};
	}
};

// 156485479_4_4_4
// Hopcroft Karp Algorithm / Fast Bipartite Matching
// Call solve() for maximum matching
// u from the left vertex set is matched with pu[u] on the right (-1 if not matched)
// v from the right vertex set is matched with pv[v] on the left (-1 if not matched)
// O( sqrt(V) * E )
struct hopcroft_karp{
	int n, m, flow = 0;
	vector<vector<int>> adj;
	vector<int> pu, pv, U, V, cur, next;
	hopcroft_karp(int n, int m): n(n), m(m), adj(n), pu(n, -1), pv(m, -1), U(n), V(m){ }
	int insert(int from, int to){
		adj[from].push_back(to);
		return int(adj[from].size()) - 1;
	}
	bool bfs(){
		fill(U.begin(), U.end(), 0), fill(V.begin(), V.end(), 0);
		cur.clear();
		// Find the starting nodes for VFS (i.e. layer 0).
		for(auto u: pv) if(u != -1) U[u] = -1;
		for(int u = 0; u < n; ++ u) if(!U[u]) cur.push_back(u);
		// Find all layers using bfs.
		for(int layer = 1; ; ++ layer){
			bool islast = 0;
			next.clear();
			for(auto u: cur) for(auto v: adj[u]){
				if(pv[v] == -1){
					V[v] = layer;
					islast = 1;
				}
				else if(pv[v] != u && !V[v]){
					V[v] = layer;
					next.push_back(pv[v]);
				}
			}
			if(islast) return true;
			if(next.empty()) return false;
			for(auto u: next) U[u] = layer;
			cur.swap(next);
		}
	}
	bool dfs(int u, int L){
		if(U[u] != L) return false;
		U[u] = -1;
		for(auto v: adj[u]) if(V[v] == L + 1){
			V[v] = 0;
			if(pv[v] == -1 || dfs(pv[v], L + 1)){
				pu[u] = v;
				pv[v] = u;
				return true;
			}
		}
		return false;
	}
	int solve(){
		while(bfs()) for(int u = 0; u < n; ++ u) flow += dfs(u, 0);
		return flow;
	}
	pair<vector<int>, vector<int>> minimum_vertex_cover(){
		solve();
		vector<int> L, R;
		for(int u = 0; u < n; ++ u) if(!~U[u]) L.push_back(u);
		for(int v = 0; v < m; ++ v) if(V[v]) R.push_back(v);
		return {L, R};
	}
};

// 156485479_4_4_5
// Hungarian Algorithm / Minimum Weight Maximum Bipartite Matching ( WARNING: UNTESTED )
// O(V^2 E)
// Reads the adjacency matrix of the graph
template<typename Graph>
pair<long long, vector<int>> hungarian(const Graph &adj){
	if(adj.empty()) return {0, {}};
	int n = int(adj.size()) + 1, m = int(adj[0].size()) + 1;
	vector<long long> u(n), v(m);
	vector<int> p(m), ans(n - 1);
	for(int i = 1; i < n; ++ i){
		p[0] = i;
		int j0 = 0; // add "dummy" worker 0
		vector<long long> dist(m, numeric_limits<long long>::max());
		vector<int> pre(m, -1);
		vector<bool> done(m + 1);
		do{// dijkstra
			done[j0] = true;
			int i0 = p[j0], j1;
			long long delta = numeric_limits<long long>::max();
			for(int j = 1; j < m; ++ j) if(!done[j]){
				auto cur = adj[i0 - 1][j - 1] - u[i0] - v[j];
				if(cur < dist[j]) dist[j] = cur, pre[j] = j0;
				if(dist[j] < delta) delta = dist[j], j1 = j;
			}
			for(int j = 0; j < m; ++ j){
				if(done[j]) u[p[j]] += delta, v[j] -= delta;
				else dist[j] -= delta;
			}
			j0 = j1;
		}while(p[j0]);
		while(j0){ // update alternating path
			int j1 = pre[j0];
			p[j0] = p[j1], j0 = j1;
		}
	}
	for(int j = 1; j < m; ++ j) if(p[j]) ans[p[j] - 1] = j - 1;
	return {-v[0], ans}; // min cost
}

// 156485479_4_4_6
// Global Min Cut ( WARNING: UNTESTED )
// O(V^3)
template<typename Graph>
pair<int, vector<int>> global_min_cut(Graph adj){
	int n = int(adj.size());
	vector<int> used(n), cut, best_cut;
	int best_weight = -1;
	for(int phase = n - 1; phase >= 0; -- phase){
		vector<int> w = adj[0], added = used;
		int prev, k = 0;
		for(int i = 0; i < phase; ++ i){
			prev = k;
			k = -1;
			for(int j = 1; j < n; ++ j) if(!added[j] && (k == -1 || w[j] > w[k])) k = j;
			if(i == phase-1){
				for(int j = 0; j < n; ++ j) adj[prev][j] += adj[k][j];
				for(int j = 0; j < n; ++ j) adj[j][prev] = adj[prev][j];
				used[k] = true;
				cut.push_back(k);
				if(best_weight == -1 || w[k] < best_weight){
					best_cut = cut;
					best_weight = w[k];
				}
			}
			else{
				for(int j = 0; j < n; ++ j) w[j] += adj[k][j];
				added[k] = true;
			}
		}
	}
	return {best_weight, best_cut};
}

// 156485479_4_4_7
// Gomory-Hu Tree

// 156485479_4_4_8
// General Matching
// Fails with probability N/mod
template<typename E>
vector<array<int, 2>> generalMatching(int n, const E &edges, const int &mod){
	vector<vector<long long>> mat(n, vector<long long>(n)), A;
	for(auto pa: edges){
		int a = pa.first, b = pa.second, r = rng() % mod;
		mat[a][b] = r, mat[b][a] = (mod - r) % mod;
	}
	int r = matInv(A = mat), m = 2 * n - r, fi, fj;
	assert(r % 2 == 0);
	if(m != n) do{
		mat.resize(m, vector<long long>(m));
		for(int i = 0; i < n; ++ i){
			mat[i].resize(m);
			for(int j = n; j < m; ++ j){
				int r = rng() % mod;
				mat[i][j] = r, mat[j][i] = (mod - r) % mod;
			}
		}
	}while(matInv(A = mat) != m);
	vector<int> has(m, 1);
	vector<array<int, 2>> res;
	for(int it = 0; it < m >> 1; ++ it){
		for(int i = 0; i < m; ++ i) if(has[i])
			for(int j = i + 1; j < m; ++ j) if(A[i][j] && mat[i][j]){
				fi = i; fj = j;
				goto done;
			}
		assert(0);
		done:
		if(fj < n) res.push_back({fi, fj});
		has[fi] = has[fj] = 0;
		for(int sw = 0; sw < 2; ++ sw){
			long long a = modpow(A[fi][fj], mod - 2);
			for(int i = 0; i < m; ++ i) if(has[i] && A[i][fj]){
				long long b = A[i][fj] * a % mod;
				for(j,0,m) A[i][j] = (A[i][j] - A[fi][j] * b) % mod;
			}
			swap(fi,fj);
		}
	}
	return ret;
}

// 156485479_4_5_1
// LCA
// O(n log n) processing, O(1) per query
// Requires sparse_table
struct lca{
	int n, T = 0;
	vector<int> time, depth, path, res;
	sparse_table<int, function<int(int, int)>> rmq;
	template<typename Graph>
	lca(const Graph &adj, int root = 0): n(int(adj.size())), time(n), depth(n){
		dfs(adj, root, root);
		rmq = {res.begin(), res.end(), [&](int x, int y){ return min(x, y); }, numeric_limits<int>::max() / 2};
	}
	template<typename Graph>
	void dfs(const Graph &adj, int u, int p){
		time[u] = T ++;
		for(auto v: adj[u]) if(v != p){
			path.push_back(u), res.push_back(time[u]), depth[v] = depth[u] + 1;
			dfs(adj, v, u);
		}
	}
	int query(int u, int v){
		if(u == v) return u;
		tie(u, v) = minmax(time[u], time[v]);
		return path[rmq.query(u, v)];
	}
	int dist(int u, int v, int w = -1){ return depth[u] + depth[v] - 2 * depth[~w ? w : query(u, v)]; }
};
// For Weighted Tree
// Requires sparse_table
struct weighted_lca{
	int n, T = 0;
	vector<int> time, depth, path, res;
	vector<long long> dist;
	sparse_table<int, function<int(int, int)>> rmq;
	template<typename Graph>
	lca(const Graph &adj, int root = 0): n(int(adj.size())), time(n), depth(n), dist(n){
		dfs(adj, root, root);
		rmq = {res.begin(), res.end(), [&](int x, int y){ return min(x, y); }, numeric_limits<int>::max() / 2};
	}
	template<typename Graph>
	void dfs(const Graph &adj, int u, int p){
		time[u] = T ++;
		for(auto &[v, w]: adj[u]) if(v != p){
			path.push_back(u), res.push_back(time[u]), depth[v] = depth[u] + 1, dist[v] = dist[u] + w;
			dfs(adj, v, u);
		}
	}
	int query(int u, int v){
		if(u == v) return u;
		tie(u, v) = minmax(time[u], time[v]);
		return path[rmq.query(u, v)];
	}
	pair<int, long long> dist(int u, int v, int w = -1){
		int l = ~w ? w : query(u, v);
		return {depth[u] + depth[v] - 2 * depth[l], dist[u] + dist[v] - 2 * dist[l]};
	}
};

// 156485479_4_5_2
// Binary Lifting
// Also works for graphs with outdegree 1 for all vertices.
// O(n log n) preprocessing, O(log n) per lca query
// Credit: Benq
struct binary_lift{
	int n, lg;
	vector<vector<int>> up;
	vector<int> depth, visited;
	template<typename Graph>
	binary_lift(const Graph &adj): n(int(adj.size())), lg(__lg(n) + 1), depth(n), visited(n), up(n, vector<int>(__lg(n) + 2)){
		for(int u = 0; u < n; ++ u) if(!visited[u]) dfs(adj, u, u);
	}
	template<typename Graph>
	void dfs(const Graph &adj, int u, int p){
		visited[u] = true;
		up[u][0] = p;
		for(int i = 1; i <= lg; ++ i) up[u][i] = up[up[u][i - 1]][i - 1];
		for(auto v: adj[u]) if(v != p){
			depth[v] = depth[u] + 1;
			dfs(adj, v, u);
		}
	}
	int lca(int u, int v){
		if(depth[u] < depth[v]) swap(u, v);
		u = trace_up(u, depth[u] - depth[v]);
		for(int d = lg; d >= 0; -- d) if(up[u][d] != up[v][d]) u = up[u][d], v = up[v][d];
		return u == v ? u : up[u][0];
	}
	int dist(int u, int v, int w = -1){
		return depth[u] + depth[v] - 2 * depth[~w ? w : lca(u, v)];
	}
	int trace_up(int u, int dist){
		for(int d = lg; d >= 0; -- d) if(dist & (1 << d)) u = up[u][d];
		return u;
	}
};
// For Weighted Tree
template<typename T, typename U, typename BO>
struct weighted_binary_lift{
	int n, lg;
	BO bin_op;
	T id;
	vector<U> val;
	vector<vector<pair<int, T>>> up;
	vector<int> depth, visited;
	template<typename Graph>
	weighted_binary_lift(const Graph &adj, vector<U> val, BO bin_op, T id): n(int(adj.size())), bin_op(bin_op), id(id), lg(__lg(n) + 1), depth(n), visited(n), val(move(val)), up(n, vector<pair<int, T>>(lg + 1)){
		for(int u = 0; u < n; ++ u) if(!visited[u]) dfs(adj, u, u, id);
	}
	template<typename Graph>
	void dfs(const Graph &adj, int u, int p, T w){
		visited[u] = true;
		up[u][0] = {p, bin_op(val[u], w)};
		for(int i = 1; i <= lg; ++ i) up[u][i] = {
			up[up[u][i - 1].first][i - 1].first
			, bin_op(up[u][i - 1].second, up[up[u][i - 1].first][i - 1].second)
		};
		for(auto &[v, x]: adj[u]) if(v != p){
			depth[v] = depth[u] + 1;
			dfs(adj, v, u, x);
		}
	}
	pair<int, T> trace_up(int u, int dist){ // Node, Distance (Does not include weight of the Node)
		T res = id;
		for(int d = lg; d >= 0; -- d) if(dist & (1 << d)){
			res = bin_op(res, up[u][d].second), u = up[u][d].first;
		}
		return {u, res};
	}
	pair<int, T> query(int u, int v){ // LCA, Query Value
		if(depth[u] < depth[v]) swap(u, v);
		T res;
		tie(u, res) = trace_up(u, depth[u] - depth[v]);
		for(int d = lg; d >= 0; -- d) if(up[u][d].first != up[v][d].first){
			res = bin_op(res, up[u][d].second), u = up[u][d].first;
			res = bin_op(res, up[v][d].second), v = up[v][d].first;
		}
		if(u != v) res = bin_op(bin_op(res, up[u][0].second), up[v][0].second), u = up[u][0].first;
		return {u, bin_op(res, val[u])};
	}
	int dist(int u, int v, int w = -1){
		return depth[u] + depth[v] - 2 * depth[~w ? w : query(u, v).first];
	}
};

// 156485479_4_5_3
// Vertex Update Path Query
// O(N log N) processing, O(log N) per query
// Requires sparse_table and fenwick_tree
template<typename T, typename BO, typename IO>
struct vertex_update_path_query{
	int n;
	BO bin_op;
	IO inv_op;
	T id;
	vector<int> root, tin, tout, depth;
	vector<T> val;
	fenwick_tree<T, BO, IO> tr;
	sparse_table<int, function<int(int, int)>> st;
	template<typename Graph, typename IT>
	vertex_update_path_query(const Graph &adj, IT begin, IT end, BO bin_op, IO inv_op, T id, const vector<int> &s = {0}): n(int(adj.size())), val(begin, end), root(n), bin_op(bin_op), inv_op(inv_op), id(id), tin(n), tout(n), depth(n){
		vector<int> et;
		for(auto &r: s){
			function<void(int, int)> dfs = [&](int u, int p){
				root[u] = r, tin[u] = int(et.size());
				et.push_back(u);
				for(auto v: adj[u]) if(v ^ p) depth[v] = depth[u] + 1, dfs(v, u), et.push_back(u);
				tout[u] = int(et.size());
			};
			dfs(r, r);
		}
		vector<T> a(2 * n, id);
		for(int u = 0; u < n; ++ u) a[tin[u]] = bin_op(a[tin[u]], *(begin + u)), a[tout[u]] = inv_op(a[tout[u]], *(begin + u));
		tr = fenwick_tree(a.begin(), a.end(), bin_op, inv_op, id);
		st = sparse_table<int, function<int(int, int)>>(et.begin(), et.end(), [&](int u, int v){ return depth[u] < depth[v] ? u : v; }, -1);
	}
	template<typename Graph>
	vertex_update_path_query(const Graph &adj, BO bin_op, IO inv_op, T id, const vector<int> &s = {0}): n(int(adj.size())), val(n, id), root(n), bin_op(bin_op), inv_op(inv_op), id(id), tin(n), tout(n), depth(n){
		vector<int> et;
		for(auto &r: s){
			function<void(int, int)> dfs = [&](int u, int p){
				root[u] = r, tin[u] = int(et.size());
				et.push_back(u);
				for(auto v: adj[u]) if(v ^ p) depth[v] = depth[u] + 1, dfs(v, u), et.push_back(u);
				tout[u] = int(et.size());
			};
			dfs(r, r);
		}
		tr = fenwick_tree(2 * n, bin_op, inv_op, id);
		st = sparse_table(et.begin(), et.end(), [&](int u, int v){ return depth[u] < depth[v] ? u : v; }, -1);
	}
	bool is_ancestor_of(int u, int v){
		return tin[u] <= tin[v] && tout[v] <= tout[u];
	}
	int lca(int u, int v){
		if(tin[u] > tin[v]) swap(u, v);
		if(is_ancestor_of(u, v)) return u;
		return st.query(tout[u], tin[v]);
	}
	template<bool increment = true>
	void update(int u, T x){
		tr.template update<increment>(tin[u], x), tr.template update<increment>(tout[u], inv_op(id, x));
		val[u] = increment ? bin_op(val[u], x) : x;
	}
	T query(int u, int v){
		int w = lca(u, v), r = root[u];
		assert(is_ancestor_of(w, u) && is_ancestor_of(w, v));
		T x = tr.query(tin[r], tin[w] + 1);
		return inv_op(inv_op(bin_op(bin_op(tr.query(tin[r], tin[u] + 1), tr.query(tin[r], tin[v] + 1)), val[w]), x), x);
	}
};

// 156485479_4_5_4
// Heavy Light Decomposition / HLD
// O(N + M) processing, O(log^2 N) per query
// Credit: Benq
// Requires lazy_segment_tree or dynamic_lazy_segment_tree
template<typename DS, int VALS_IN_EDGES = 0>
struct heavy_light_decomposition{
	int n, root;
	vector<vector<int>> adj;
	vector<int> par, sz, depth, next, pos, rpos;
	DS *ds;

	using T = int;
	T bin_op(T x, T y){
		return min(x, y);
	}
	T id{numeric_limits<int>::max()};

	template<typename Graph>
	heavy_light_decomposition(const Graph &adj, DS *ds, int root = 0): n(int(adj.size())), adj(adj), root(root), par(n, -1), sz(n, 1), depth(n), next(n, root), pos(n), ds(ds){
		dfs_sz(root), dfs_hld(root);
	}
	void dfs_sz(int u){
		if(~par[u]) adj[u].erase(find(adj[u].begin(), adj[u].end(), par[u]));
		for(auto &v: adj[u]){
			par[v] = u, depth[v] = depth[u] + 1;
			dfs_sz(v);
			sz[u] += sz[v];
			if(sz[v] > sz[adj[u][0]]) swap(v, adj[u][0]);
		}
	}
	int timer = 0;
	void dfs_hld(int u){
		pos[u] = timer ++;
		rpos.push_back(u);
		for(auto &v: adj[u]){
			next[v] = (v == adj[u][0] ? next[u] : v);
			dfs_hld(v);
		}
	}
	int lca(int u, int v){
		for(; next[u] != next[v]; v = par[next[v]]) if(depth[next[u]] > depth[next[v]]) swap(u, v);
		return depth[u] < depth[v] ? u : v;
	}
	int dist(int u, int v, int w = -1){
		return depth[u] + depth[v] - 2 * depth[~w ? w : lca(u, v)];
	}
	template<typename Process>
	void processpath(int u, int v, Process act){
		for(; next[u] != next[v]; v = par[next[v]]){
			if(depth[next[u]] > depth[next[v]]) swap(u, v);
			act(pos[next[v]], pos[v] + 1);
		}
		if(depth[u] > depth[v]) swap(u, v);
		act(pos[u] + VALS_IN_EDGES, pos[v] + 1);
	}
	template<typename U>
	void updatepoint(int u, U val){
		ds->update(pos[u], val);
	} // For point update segtree
	template<typename U>
	void updatepath(int u, int v, U val){
		processpath(u, v, [this, &val](int l, int r){ ds->update(l, r, val); });
	} // For range update segtree
	template<typename U>
	void updatesubtree(int u, U val){
		ds->update(pos[u] + VALS_IN_EDGES, pos[u] + sz[u], val);
	} // For range update segtree
	T querypath(int u, int v){
		T res = id;
		processpath(u, v, [this, &res](int l, int r){ res = bin_op(res, ds->query(l, r)); });
		return res;
	}
	T querysubtree(int u){
		return ds->query(pos[u] + VALS_IN_EDGES, pos[u] + sz[u]);
	}
};

// 156485479_4_5_5
// Find all the centroids
// O(n)
vector<int> centroid(const vector<vector<int>> &adj){
	int n = int(adj.size());
	vector<int> sz(n, 1);
	function<void(int, int)> dfs_sz = [&](int u, int p){
		for(auto v: adj[u]) if(v != p){
			dfs_sz(v, u);
			sz[u] += sz[v];
		}
	};
	dfs_sz(0, -1);
	function<vector<int>(int, int)> dfs_cent = [&](int u, int p){
		for(auto v: adj[u]) if(v != p && sz[v] > n / 2) return dfs_cent(v, u);
		for(auto v: adj[u]) if(v != p && n - sz[v] <= n / 2) return vector<int>{u, v};
		return vector<int>{u};
	};
	return dfs_cent(0, -1);
}
// Centroid Decomposition
// O(n log n) processing
// Credit: Benq
struct centroid_decomposition{
	int n, root;
	vector<int> dead, sz, par, cpar;
	const vector<vector<int>> &adj;
	vector<vector<int>> cchild, dist;
	template<typename Graph>
	centroid_decomposition(const Graph &adj): n(int(adj.size())), adj(adj), dead(n), sz(n), par(n), cchild(n), cpar(n), dist(n){
		dfs_centroid(0, -1);
	}
	void dfs_sz(int u){
		sz[u] = 1;
		for(auto v: adj[u]) if(!dead[v] && v != par[u]){
			par[v] = u;
			dfs_sz(v);
			sz[u] += sz[v];
		}
	}
	int centroid(int u){
		par[u] = -1;
		dfs_sz(u);
		int s = sz[u];
		while(1){
			int w = 0, msz = 0;
			for(auto v: adj[u]) if(!dead[v] && v != par[u] && msz < sz[v]) w = v, msz = sz[v];
			if(msz * 2 <= s) return u;
			u = w;
		}
	}
	void dfs_dist(int u, int p){
		dist[u].push_back(dist[p].back() + 1);
		for(auto v: adj[u]) if(!dead[v] && v != p) dfs_dist(v, u);
	}
	void dfs_centroid(int u, int p){
		dead[u = centroid(u)] = true;
		cpar[u] = p;
		if(p != -1) cchild[p].push_back(u);
		else root = u;
		dist[u].push_back(0);
		for(auto v: adj[u]) if(!dead[v]) dfs_dist(v, u);
		for(auto v: adj[u]) if(!dead[v]) dfs_centroid(v, u);
	}
};

// 156485479_4_5_6
// AHU Algorithm ( Rooted Tree Isomorphism ) / Tree Isomorphism
// O(n)
void radix_sort(vector<pair<int, vector<int>>> &arr){
	int n = int(arr.size()), mxval = 0, mxsz = 1 + accumulate(arr.begin(), arr.end(), 0, [](int x, const pair<int, vector<int>> &y){return max(x, int(y.second.size()));});
	vector<vector<int>> occur(mxsz);
	for(int i = 0; i < n; ++ i){
		occur[arr[i].second.size()].push_back(i);
		for(auto x: arr[i].second) mxval = max(mxval, x);
	}
	++ mxval;
	for(int size = 1; size < mxsz; ++ size) for(int d = size - 1; d >= 0; -- d){
		vector<vector<int>> bucket(mxval);
		for(auto i: occur[size]) bucket[arr[i].second[d]].push_back(i);
		occur[size].clear();
		for(auto &b: bucket) for(auto i: b) occur[size].push_back(i);
	}
	vector<pair<int, vector<int>>> res;
	res.reserve(n);
	for(auto &b: occur) for(auto i: b) res.push_back(arr[i]);
	swap(res, arr);
}
bool isomorphic(const vector<vector<vector<int>>> &adj, const vector<int> &root){
	int n = int(adj[0].size());
	if(int(adj[1].size()) != n) return false;
	vector<vector<vector<int>>> occur(2);
	vector<vector<int>> depth(2, vector<int>(n)), par(2, vector<int>(n, -1));
	for(int k = 0; k < 2; ++ k){
		function<void(int, int)> dfs = [&](int u, int p){
			par[k][u] = p;
			for(auto v: adj[k][u]) if(v != p){
				depth[k][v] = depth[k][u] + 1;
				dfs(v, u);
			}
		};
		dfs(root[k], -1);
		int mxdepth = 1 + accumulate(depth[k].begin(), depth[k].end(), 0, [](int x, int y){return max(x, y);});
		occur[k].resize(mxdepth);
		for(int u = 0; u < n; ++ u) occur[k][depth[k][u]].push_back(u);
	}
	int mxdepth = int(occur[0].size());
	if(mxdepth != int(occur[1].size())) return false;
	for(int d = 0; d < mxdepth; ++ d) if(occur[0][d].size() != occur[1][d].size()) return false;
	vector<vector<int>> label(2, vector<int>(n)), pos(2, vector<int>(n));
	vector<vector<vector<int>>> sorted_list(mxdepth, vector<vector<int>>(2));
	for(int k = 0; k < 2; ++ k){
		sorted_list[mxdepth - 1][k].reserve(occur[k][mxdepth - 1].size());
		for(auto u: occur[k][mxdepth - 1]) sorted_list[mxdepth - 1][k].push_back(u);
	}
	for(int d = mxdepth - 2; d >= 0; -- d){
		vector<vector<pair<int, vector<int>>>> tuples(2);
		for(int k = 0; k < 2; ++ k){
			tuples[k].reserve(occur[k][d].size());
			for(auto u: occur[k][d]){
				pos[k][u] = int(tuples[k].size());
				tuples[k].emplace_back(u, vector<int>());
			}
			for(auto v: sorted_list[d + 1][k]){
				int u = par[k][v];
				tuples[k][pos[k][u]].second.push_back(label[k][v]);
			}
			radix_sort(tuples[k]);
		}
		for(int i = 0; i < int(tuples[0].size()); ++ i) if(tuples[0][i].second != tuples[1][i].second) return false;
		for(int k = 0; k < 2; ++ k){
			int cnt = 0;
			sorted_list[d][k].reserve(occur[k][d].size());
			sorted_list[d][k].push_back(tuples[k][0].first);
			for(int i = 1; i < int(tuples[k].size()); ++ i){
				int u = tuples[k][i].first;
				label[k][u] = (tuples[k][i - 1].second == tuples[k][i].second ? cnt : ++ cnt);
				sorted_list[d][k].push_back(u);
			}
		}
	}
	return true;
}
vector<int> centroid(const vector<vector<int>> &adj){
	int n = int(adj.size());
	vector<int> size(n, 1);
	function<void(int, int)> dfs_sz = [&](int u, int p){
		for(auto v: adj[u]) if(v != p){
			dfs_sz(v, u);
			size[u] += size[v];
		}
	};
	dfs_sz(0, -1);
	function<vector<int>(int, int)> dfs_cent = [&](int u, int p){
		for(auto v: adj[u]) if(v != p && size[v] > n / 2) return dfs_cent(v, u);
		for(auto v: adj[u]) if(v != p && n - size[v] <= n / 2) return vector<int>{u, v};
		return vector<int>{u};
	};
	return dfs_cent(0, -1);
}
bool isomorphic(const vector<vector<vector<int>>> &adj){
	vector<vector<int>> cent{centroid(adj[0]), centroid(adj[1])};
	if(cent[0].size() != cent[1].size()) return false;
	for(auto u: cent[0]) for(auto v: cent[1]) if(isomorphic(adj, vector<int>{u, v})) return true;
	return false;
}

// 156485479_4_7
// Minimum Spanning Forest
// O(m log n)
// Requires disjoint_set
template<typename T = long long>
struct minimum_spanning_forest{
	int n;
	vector<vector<pair<int, int>>> adj;
	vector<vector<int>> mst_adj;
	vector<int> mst_edge;
	vector<tuple<int, int, T>> edge;
	T cost = 0;
	minimum_spanning_forest(int n): n(n), adj(n), mst_adj(n){ }
	void insert(int u, int v, T w){
		adj[u].emplace_back(v, edge.size()), adj[v].emplace_back(u, edge.size());
		edge.emplace_back(u, v, w);
	}
	void init_kruskal(){
		int m = int(edge.size());
		vector<int> t(m);
		iota(t.begin(), t.end(), 0);
		sort(t.begin(), t.end(), [&](int i, int j){ return get<2>(edge[i]) < get<2>(edge[j]); });
		disjoint_set dsu(n);
		for(auto i: t){
			auto [u, v, w] = edge[i];
			if(dsu.merge(u, v)){
				cost += w;
				mst_edge.push_back(i);
				mst_adj[u].push_back(v), mst_adj[v].push_back(u);
			}
		}
	}
	void init_prim(){
		vector<bool> used(n);
		priority_queue<tuple<T, int, int, int>, vector<tuple<T, int, int, int>>, greater<>> q;
		for(int u = 0; u < n; ++ u) if(!used[u]){
			q.emplace(0, u, -1, -1);
			while(!q.empty()){
				auto [w, u, p, i] = q.top();
				q.pop();
				if(used[u]) continue;
				used[u] = true;
				if(p != -1){
					mst_edge.push_back(i);
					mst_adj[u].push_back(p), mst_adj[p].push_back(u);
				}
				cost += w;
				for(auto [v, i]: adj[u]) if(!used[v]) q.emplace(get<2>(edge[i]), v, u, i);
			}
		}
	}
};
// For dense graph
// O(n^2)
template<typename T = long long>
struct minimum_spanning_forest_dense{
	static constexpr T inf = numeric_limits<T>::max();
	int n, edgecnt = 0;
	vector<vector<T>> adjm;
	vector<vector<int>> adj;
	vector<vector<bool>> mst_adjm;
	T cost = 0;
	minimum_spanning_forest_dense(int n): n(n), adjm(n, vector<T>(n, inf)), adj(n), mst_adjm(n, vector<bool>(n)){ }
	void insert(int u, int v, T w){
		adjm[u][v] = adjm[v][u] = w;
		adj[u].push_back(v), adj[v].push_back(u);
	}
	void init_prim(){
		vector<bool> used(n), reached(n);
		vector<int> reach;
		vector<tuple<T, int, int>> t(n, {inf, -1, 0});
		for(int u = 0; u < n; ++ u) if(!used[u]){
			function<void(int)> dfs = [&](int u){
				reached[u] = true;
				reach.push_back(u);
				for(auto v: adj[u]) if(!reached[v]) dfs(v);
			};
			dfs(u);
			get<0>(t[reach[0]]) = 0;
			for(int tt = 0; tt < reach.size(); ++ tt){
				int u = -1;
				for(auto v: reach) if(!used[v] && (u == -1 || get<0>(t[v]) < get<0>(t[u]))) u = v;
				auto [w, p, ignore] = t[u];
				used[u] = true;
				cost += w;
				if(p != -1){
					mst_adjm[u][p] = mst_adjm[p][u] = true;
					++ edgecnt;
				}
				for(auto v: reach) if(adjm[u][v] < get<0>(t[v])) t[v] = {adjm[u][v], u, v};
			}
			reach.clear();
		}
	}
};

// 156485479_4_5_8
// Minimum Spanning Arborescence
// O(E log V)
// Credit: KACTL
// msa_edge contains the indices of edges forming msa rooted at 0.
// solve() returns -1 if no msa exists
// Requires rollback_disjoint_set
template<typename T = long long>
struct minimum_spanning_arborescence{
	struct etype{ int u, v, ind = 0; T w; };
	int n;
	vector<etype> edge;
	vector<int> msa_edge;
	T cost = 0;
	minimum_spanning_arborescence(int n): n(n){ }
	void insert(int u, int v, T w){ edge.push_back({u, v, int(edge.size()), w}); }
	struct node{ /// lazy skew heap node
		etype key;
		node *l, *r;
		T delta = 0;
		void push(){
			key.w += delta;
			if(l) l->delta += delta;
			if(r) r->delta += delta;
			delta = 0;
		}
		etype top(){ push(); return key; }
	};
	node *merge(node *u, node *v){
		if(!u || !v) return u ?: v;
		u->push(), v->push();
		if(u->key.w > v->key.w) swap(u, v);
		swap(u->l, (u->r = merge(v, u->r)));
		return u;
	}
	void pop(node *&u){ u->push(); u = merge(u->l, u->r); }
	T solve(){
		rollback_disjoint_set dsu(n);
		vector<node *> heap(n);
		for(auto &e: edge) heap[e.v] = merge(heap[e.v], new node{e});
		vector<int> seen(n, -1), path(n), par(n);
		seen[0] = 0;
		vector<etype> Q(n), in(n, {-1, -1}), comp;
		deque<tuple<int, int, vector<etype>>> cycles;
		for(int s = 0; s < n; ++ s){
			int u = s, qi = 0, v;
			while(seen[u] < 0){
				if(!heap[u]) return cost = -1;
				etype e = heap[u]->top();
				heap[u]->delta -= e.w, pop(heap[u]);
				Q[qi] = e, path[qi ++] = u, seen[u] = s;
				cost += e.w, u = dsu.root(e.u);
				if(seen[u] == s){
					node *cycle = 0;
					int end = qi, time = dsu.time();
					do cycle = merge(cycle, heap[v = path[-- qi]]);
					while(dsu.merge(u, v));
					u = dsu.root(u), heap[u] = cycle, seen[u] = -1;
					cycles.push_front({u, time, {&Q[qi], &Q[end]}});
				}
			}
			for(int i = 0; i < qi; ++ i) in[dsu.root(Q[i].v)] = Q[i];
		}
		for(auto &[u, t, comp]: cycles){
			dsu.reverse_to(t);
			etype inedge = in[u];
			for(auto &e: comp) in[dsu.root(e.v)] = e;
			in[dsu.root(inedge.v)] = inedge;
		}
		for(int u = 0; u < n; ++ u) if(u) msa_edge.push_back(in[u].ind);
		return cost;
	}
};

// 156485479_4_5_9
// Compressed Tree ( Virtual Tree, Auxiliary Tree )template<typename T, typename BO>
// O(S log S)
// Returns a list of (parent, original index) where parent of root = root
// Credit: KACTL
// Requires sparse_table and lca and weighted_lca
template<typename LCA>
vector<array<int, 2>> compressed_tree(LCA &lca, vector<int> &subset){
	static vector<int> rev; rev.resize(int(lca.time.size()));
	vector<int> li = subset, &T = lca.time;
	auto cmp = [&](int a, int b) { return T[a] < T[b]; };
	sort(li.begin(), li.end(), cmp);
	int m = int(li.size()) - 1;
	for(int i = 0; i < m; ++ i){
		int u = li[i], v = li[i + 1];
		li.push_back(lca.query(u, v));
	}
	sort(li.begin(), li.end(), cmp);
	li.erase(unique(li.begin(), li.end()), li.end());
	for(int i = 0; i < int(li.size()); ++ i) rev[li[i]] = i;
	vector<array<int, 2>> res = {{0, li[0]}};
	for(int i = 0; i < int(li.size()) - 1; ++ i){
		int u = li[i], v = li[i + 1];
		res.push_back({rev[lca.query(u, v)], v});
	}
	return res;
}

// 156485479_4_5_10
// Pruefer Code
// O(V) for both
// Number of labeled tree of N vertices: N^(N-2)
// Number of ways to make a graph of N vertices and K components of size
// s_1, ..., s_k connected: PI(s_i)N^(K-2)
// Credit: CP-Algorithms
template<typename Graph>
vector<int> pruefer_code(const Graph &adj){
	int n = int(adj.size());
	vector<int> parent(n);
	function<void(int, int)> dfs = [&](int u, int p){
		parent[u] = p;
		for(auto v: adj[u]) if(v != p) dfs(v, u);
	};
	dfs(n - 1, -1);
	int ptr = -1;
	vector<int> deg(n);
	for(int u = 0; u < n; ++ u){
		deg[u] = int(adj[u].size());
		if(deg[u] == 1 && ptr == -1) ptr = u;
	}
	vector<int> code(n - 2);
	for(int i = 0, leaf = ptr; i < n - 2; ++ i){
		int next = parent[leaf];
		code[i] = next;
		if(-- deg[next] == 1 && next < ptr) leaf = next;
		else{
			++ ptr;
			while(deg[ptr] != 1) ++ ptr;
			leaf = ptr;
		}
	}
	return code;
}
// Decode
template<typename Code>
vector<array<int, 2>> pruefer_decode(const Code &code){
	int n = int(code.size()) + 2;
	vector<int> deg(n, 1);
	for(auto u: code) ++ deg[u];
	int ptr = 0;
	while(deg[ptr] != 1) ++ ptr;
	int leaf = ptr;
	vector<array<int, 2>> edges;
	for(auto u: code){
		edges.emplace_back(leaf, u);
		if(-- deg[u] == 1 && u < ptr) leaf = u;
		else{
			++ ptr;
			while(deg[ptr] != 1) ++ ptr;
			leaf = ptr;
		}
	}
	edges.push_back({leaf, n - 1});
	return edges;
}

// 156485479_4_6_1
// Shortest Path Tree On Sparse Graph ( Dijkstra, Bellman Ford, SPFA )
template<typename T = long long, typename BO = plus<T>, typename Compare = less<T>>
struct shortest_path_tree{
	struct edge{
		int from, to;
		T cost;
	};
	int n;
	BO bin_op;
	Compare cmp;
	const T inf, id;
	vector<vector<int>> adj;
	vector<edge> edge;
	vector<T> dist;
	vector<int> parent;
	shortest_path_tree(int n, const T inf = numeric_limits<T>::max() / 8, BO bin_op = plus<T>(), T id = 0, Compare cmp = less<T>()): n(n), inf(inf), bin_op(bin_op), id(id), cmp(cmp), adj(n){ }
	void insert(int u, int v, T w){
		adj[u].push_back(int(edge.size()));
		edge.push_back({u, v, w});
	}
	void init(){
		dist.resize(n), parent.resize(n);
		fill(dist.begin(), dist.end(), inf), fill(parent.begin(), parent.end(), -1);
	}
	void init_bfs(const vector<int> S = {0}){
		init();
		deque<int> q;
		for(auto s: S){
			dist[s] = id;
			q.push_back(s);
		}
		while(!q.empty()){
			int u = q.front();
			q.pop_front();
			for(auto i: adj[u]){
				auto [_, v, w] = edge[i];
				if(dist[v] == inf){
					dist[v] = bin_op(dist[u], w);
					q.push_back(v);
				}
			}
		}
	}
	void init_dijkstra(const vector<int> S = {0}){
		init();
		auto qcmp = [&](const pair<T, int> &lhs, const pair<T, int> &rhs){
			return lhs.first == rhs.first ? lhs.second < rhs.second : cmp(rhs.first, lhs.first);
		};
		priority_queue<pair<T, int>, vector<pair<T, int>>, decltype(qcmp)> q(qcmp);
		for(auto s: S){
			dist[s] = id;
			q.push({id, s});
		}
		while(!q.empty()){
			auto [d, u] = q.top();
			q.pop();
			if(d != dist[u]) continue;
			for(int i: adj[u]){
				auto [u, v, w] = edge[i];
				if(cmp(bin_op(dist[u], w), dist[v])){
					dist[v] = bin_op(dist[u], w);
					parent[v] = i;
					q.push({dist[v], v});
				}
			}
		}
	}
	template<bool find_any_cycle = false>
	pair<vector<int>, vector<int>> init_bellman_ford(const vector<int> S = {0}){ // cycle {vertices, edges}
		if(find_any_cycle){
			fill(dist.begin(), dist.end(), id);
			fill(parent.begin(), parent.end(), -1);
		}
		else{
			init();
			for(auto s: S) dist[s] = id;
		}
		int x;
		for(int i = 0; i < n; ++ i){
			x = -1;
			for(int j = 0; j < edge.size(); ++ j){
				auto [u, v, w] = edge[j];
				if(cmp(dist[u], inf) && cmp(bin_op(dist[u], w), dist[v])){
					dist[v] = cmp(-inf, bin_op(dist[u], w)) ? bin_op(dist[u], w) : -inf;
					parent[v] = j;
					x = v;
				}
			}
		}
		if(x == -1) return {};
		else{
			int y = x;
			for(int i = 0; i < n; ++ i) y = parent[y];
			vector<int> vcycle, ecycle;
			for(int c = y; ; c = edge[parent[c]].from){
				vcycle.push_back(c), ecycle.push_back(parent[c]);
				if(c == y && vcycle.size() > 1) break;
			}
			return {{vcycle.rbegin(), vcycle.rend()}, {ecycle.rbegin(), ecycle.rend()}};
		}
	}
	bool init_spfa(const vector<int> S = {0}){ // returns false if cycle
		init();
		vector<int> cnt(n);
		vector<bool> inq(n);
		deque<int> q;
		for(auto s: S){
			dist[s] = id;
			q.push_back(s);
			inq[s] = true;
		}
		while(!q.empty()){
			int u = q.front();
			q.pop_front();
			inq[u] = false;
			for(int i: adj[u]){
				auto [u, v, w] = edge[i];
				if(cmp(bin_op(dist[u], w), dist[v])){
					dist[v] = bin_op(dist[u], w);
					parent[v] = i;
					if(!inq[v]){
						q.push_back(v);
						inq[v] = true;
						++ cnt[v];
						if(cnt[v] > n) return false;
					}
				}
			}
		}
		return true;
	}
	pair<vector<int>, vector<int>> path_from_root(int u){
		vector<int> vpath, epath;
		for(; parent[u] != -1; u = edge[parent[u]].from){
			vpath.push_back(u);
			epath.push_back(parent[u]);
		}
		vpath.push_back(u);
		return {{vpath.rbegin(), vpath.rend()}, {epath.rbegin(), epath.rend()}};
	}
};

// 156485479_4_6_2
// Shortest Path Tree On Dense Graph ( Dijkstra, Floyd Warshall )
template<typename T = long long, typename BO = plus<>, typename Compare = less<>>
struct shortest_path_tree_dense{
	struct etype{ int from, to; T cost; };
	int n;
	BO bin_op;
	Compare cmp;
	const T inf, id;
	vector<vector<int>> adj;
	vector<etype> edge;
	vector<T> dist;
	vector<int> parent;
	shortest_path_tree_dense(int n, const T inf = numeric_limits<T>::max() / 8, BO bin_op = plus<>(), T id = 0, Compare cmp = less<>()): n(n), adj(n), inf(inf), bin_op(bin_op), id(id), cmp(cmp){ }
	void insert(int u, int v, T w){
		adj[u].push_back(int(edge.size()));
		edge.push_back({u, v, w});
	}
	void init(){ dist.assign(n, inf), parent.assign(n, -1); }
	void init_bfs(const vector<int> &S = {0}){
		init();
		deque<int> q;
		for(auto s: S){
			dist[s] = id;
			q.push_back(s);
		}
		while(!q.empty()){
			int u = q.front();
			q.pop_front();
			for(auto i: adj[u]){
				auto [_, v, w] = edge[i];
				if(dist[v] == inf){
					dist[v] = bin_op(dist[u], w);
					q.push_back(v);
				}
			}
		}
	}
	void init_dijkstra(const vector<int> &S = {0}){
		init();
		auto qcmp = [&](const pair<T, int> &lhs, const pair<T, int> &rhs){
			return lhs.first == rhs.first ? lhs.second < rhs.second : cmp(rhs.first, lhs.first);
		};
		priority_queue<pair<T, int>, vector<pair<T, int>>, decltype(qcmp)> q(qcmp);
		for(auto s: S){
			dist[s] = id;
			q.push({id, s});
		}
		while(!q.empty()){
			auto [d, u] = q.top();
			q.pop();
			if(d != dist[u]) continue;
			for(int i: adj[u]){
				auto [u, v, w] = edge[i];
				if(cmp(bin_op(dist[u], w), dist[v])){
					dist[v] = bin_op(dist[u], w);
					parent[v] = i;
					q.push({dist[v], v});
				}
			}
		}
	}
	pair<vector<int>, vector<int>> init_bellman_ford(const vector<int> &S = {0}){ // cycle {vertices, edges}
		init();
		for(auto s: S) dist[s] = id;
		int x;
		for(int i = 0; i < n; ++ i){
			x = -1;
			for(int j = 0; j < edge.size(); ++ j){
				auto &[u, v, w] = edge[j];
				if(cmp(dist[u], inf) && cmp(bin_op(dist[u], w), dist[v])){
					dist[v] = cmp(-inf, bin_op(dist[u], w)) ? bin_op(dist[u], w) : -inf;
					parent[v] = j;
					x = v;
				}
			}
		}
		if(x == -1) return {};
		else{
			int y = x;
			for(int i = 0; i < n; ++ i) y = edge[parent[y]].from;
			vector<int> vcycle, ecycle;
			for(int c = y; ; c = edge[parent[c]].from){
				vcycle.push_back(c), ecycle.push_back(parent[c]);
				if(c == y && vcycle.size() > 1) break;
			}
			return {{vcycle.rbegin(), vcycle.rend()}, {ecycle.rbegin(), ecycle.rend()}};
		}
	}
	bool init_spfa(const vector<int> &S = {0}){ // returns false if cycle
		init();
		vector<int> cnt(n);
		vector<bool> inq(n);
		deque<int> q;
		for(auto s: S){
			dist[s] = id;
			q.push_back(s);
			inq[s] = true;
		}
		while(!q.empty()){
			int u = q.front();
			q.pop_front();
			inq[u] = false;
			for(int i: adj[u]){
				auto [u, v, w] = edge[i];
				if(cmp(bin_op(dist[u], w), dist[v])){
					dist[v] = bin_op(dist[u], w);
					parent[v] = i;
					if(!inq[v]){
						q.push_back(v);
						inq[v] = true;
						++ cnt[v];
						if(cnt[v] > n) return false;
					}
				}
			}
		}
		return true;
	}
	pair<vector<int>, vector<int>> path_from_root(int u){
		vector<int> vpath, epath;
		for(; parent[u] != -1; u = edge[parent[u]].from){
			vpath.push_back(u);
			epath.push_back(parent[u]);
		}
		vpath.push_back(u);
		return {{vpath.rbegin(), vpath.rend()}, {epath.rbegin(), epath.rend()}};
	}
};

// 156485479_4_7
// Topological Sort / Returns less than n elements if there's a cycle
// O(V + E)
template<typename Graph>
vector<int> toposort(const Graph &adj){
	int n = int(adj.size());
	vector<int> indeg(n), res;
	for(int u = 0; u < n; ++ u) for(auto v: adj[u]) ++ indeg[v];
	deque<int> q;
	for(int u = 0; u < n; ++ u) if (!indeg[u]) q.push_back(u);
	while(q.size() > 0){
		int u = q.front();
		q.pop_front();
		res.push_back(u);
		for(auto v: adj[u]) if (!(-- indeg[v])) q.push_back(v);
	}
	return res;
}
// Lexicographically Smallest Topological Sort / Return returns less than n elements if there's a cycle
// O(V log V + E)
template<typename Graph>
vector<int> toposort(const Graph &adj){
	int n = adj.size();
	vector<int> indeg(n), res;
	for(int u = 0; u < n; ++ u) for(auto v: adj[u]) ++ indeg[v];
	priority_queue<int, vector<int>, greater<>> q;
	for(int u = 0; u < n; ++ u) if (!indeg[u]) q.push(u);
	while(q.size() > 0){
		int u = q.top();
		q.pop();
		res.push_back(u);
		for(auto v: radj[u]) if (!(-- indeg[v])) q.push(v);
	}
	return res;
}

// 156485479_4_8
// Two Satisfiability / values hold the result
// O(V + E)
struct two_sat{
	int n;
	vector<vector<int>> adj;
	vector<int> value, val, comp, z;
	two_sat(int n = 0): n(n), adj(n << 1){ }
	int time = 0;
	int add_var(){
		adj.emplace_back();
		adj.emplace_back();
		return ++ n;
	}
	void either(int u, int v){
		u = max(2 * u, -1 - 2 * u);
		v = max(2 * v, -1 - 2 * v);
		adj[u].push_back(v ^ 1);
		adj[v].push_back(u ^ 1);
	}
	void set_value(int u){
		either(u, u);
	}
	void at_most_one(const vector<int> &arr){
		if(int(arr.size()) <= 1) return;
		int cur = ~arr[0];
		for(int u = 2; u < int(arr.size()); ++ u){
			int next = add_var();
			either(cur, ~arr[u]), either(cur, next), either(~arr[u], next);
			cur = ~next;
		}
		either(cur, ~arr[1]);
	}
	int dfs(int u){
		int low = val[u] = ++ time, v;
		z.push_back(u);
		for(auto v: adj[u]) if(!comp[v]) low = min(low, val[v] ?: dfs(v));
		++ time;
		if(low == val[u]) do{
			v = z.back();
			z.pop_back();
			comp[v] = time;
			if(value[v >> 1] == -1) value[v >> 1] = v & 1;
		}while(v != u);
		return val[u] = low;
	}
	bool solve(){
		value.assign(n, -1), val.assign(2 * n, 0), comp = val;
		for(int u = 0; u < n << 1; ++ u) if(!comp[u]) dfs(u);
		for(int u = 0; u < n; ++ u) if(comp[u << 1] == comp[u << 1 ^ 1]) return false;
		return true;
	}
};

// 156485479_4_9
// Euler Walk / adj list must be of form  [vertex, edge_index]
// O(n + m)
pair<vector<int>, vector<int>> euler_walk(const vector<vector<pair<int, int>>> &adj, int m, int source = 0){
	int n = int(adj.size());
	vector<int> deg(n), its(n), used(m), res_v, res_e;
	vector<pair<int, int>> q = {{source, -1}};
	++ deg[source]; // to allow Euler paths, not just cycles
	while(!q.empty()){
		auto [u, e] = q.back();
		int &it = its[u], end = int(adj[u].size());
		if(it == end){
			res_v.push_back(u), res_e.push_back(e), q.pop_back();
			continue;
		}
		auto [v, f] = adj[u][it ++];
		if(!used[f]){
			-- deg[u], ++ deg[v];
			used[f] = 1; q.emplace_back(v, f);
		}
	}
	for(auto d: deg) if(d < 0 || int(res_v.size()) != m + 1) return {};
	return {{res_v.rbegin(), res_v.rend()}, {res_e.rbegin() + 1, res_e.rend()}};
}

// UNTESTED
template<typename Graph>
auto euler_walk(const Graph &adj, int mx_e, int edge_cnt, const vector<int> &src = {0}){
	int n = int(adj.size()), tot = 0;
	vector<int> deg(n), its(n), used(mx_e), res_v, res_e;
	vector<pair<int, int>> q;
	vector<array<vector<int>, 2>> res;
	for(auto u: src){
		res_v.clear(), res_e.clear();
		q.push_back({u, -1});
		++ deg[u]; // to allow Euler paths, not just cycles
		while(!q.empty()){
			auto [u, e] = q.back();
			int &it = its[u], end = int(adj[u].size());
			if(it == end){
				res_v.push_back(u), res_e.push_back(e), q.pop_back();
				continue;
			}
			auto [v, f] = adj[u][it ++];
			if(!used[f]){
				-- deg[u], ++ deg[v];
				used[f] = 1; q.emplace_back(v, f);
			}
		}
		tot += int(res_v.size());
		res_e.pop_back(), reverse(res_v.begin(), res_v.end()), reverse(res_e.begin(), res_e.end());
		res.push_back({res_v, res_e}); // {{v0, v1, ..., vk}, {(v0->v1), (v1->v2), ...}}
	}
	if(tot != edge_cnt + int(src.size()) || any_of(deg.begin(), deg.end(), [&](int d){ return d < 0; })) return decltype(res){};
	return res;
}

// 156485479_4_10
// Dominator Tree
// O(n log n)
// Credit: Benq
struct dominator_tree{
	int n, source, timer = 0;
	vector<vector<int>> adj;
	// adj: dominator tree
	// Nodes below are labelled with dfs tree index 
	vector<vector<int>> radj, child, sdomChild;
	vector<int> label, rlabel, sdom, dom, par, best;
	template<typename Graph>
	dominator_tree(const Graph &init_adj, int source = 0): n(int(init_adj.size())), source(source), adj(n), radj(n), child(n), sdomChild(n), label(n, -1), rlabel(n), sdom(n), dom(n), par(n), best(n){
		dfs(init_adj, source);
		for(int i = timer - 1; i >= 0; -- i){
			for(auto j: radj[i]) sdom[i] = min(sdom[i], sdom[get(j)]);
			if(i) sdomChild[sdom[i]].push_back(i);
			for(auto j: sdomChild[i]){
				int k = get(j);
				if(sdom[j] == sdom[k]) dom[j] = sdom[j];
				else dom[j] = k;
			}
			for(auto j: child[i]) par[j] = i;
		}
		for(int i = 1; i < timer; ++ i){
			if(dom[i] != sdom[i]) dom[i] = dom[dom[i]];
			adj[rlabel[dom[i]]].push_back(rlabel[i]);
		}
	}
	template<typename Graph>
	void dfs(const Graph &adj, int u){ // create DFS tree
		rlabel[timer] = u, label[u] = sdom[timer] = par[timer] = best[timer] = timer;
		++ timer;
		for(auto v: adj[u]){
			if(!~label[v]){
				dfs(adj, v);
				child[label[u]].push_back(label[v]);
			}
			radj[label[v]].push_back(label[u]);
		}
	}
	int get(int i){// DSU with path compression, get vertex with smallest sdom on path to root
		if(par[i] != i){
			int j = get(par[i]);
			par[i] = par[par[i]];
			if(sdom[j] < sdom[best[i]]) best[i] = j;
		}
		return best[i];
	}
};

// 156485479_5_1
// Returns the starting position of the lexicographically minimal rotation
// O(n)
template<typename Str>
int min_rotation(Str s){
	int n = int(s.size());
	s += s;
	int a = 0;
	for(int b = 0; b < n; ++ b) for(int i = 0; i < n; ++ i){
		if(a + i == b || s[a + i] < s[b + i]){
			b += max(0, i - 1);
			break;
		}
		if(s[a + i] > s[b + i]){
			a = b;
			break;
		}
	}
	return a;
}

// 156485479_5_2
// Manacher's Algorithm ( Find All Palindromic Substrings )
// p[0][i]: half length of even pal substring around i, p[1][i]: odd rounded down
// O(N)
// Credit: KACTL
template<typename Str>
array<vector<int>, 2> manacher(const Str &s){
	int n = int(s.size());
	array<vector<int>, 2> p = {vector<int>(n + 1), vector<int>(n)};
	for(int z = 0; z < 2; ++ z){
		for(int i = 0, l = 0, r = 0; i < n; ++ i){
			int t = r - i + !z;
			if(i < r) p[z][i] = min(t, p[z][l + t]);
			int L = i - p[z][i], R = i + p[z][i] - !z;
			while(L >= 1 && R + 1 < n && s[L - 1] == s[R + 1]) ++ p[z][i], -- L, ++ R;
			if(R > r) l = L, r = R;
		}
	}
	return p;
}

// 156485479_5_3
// Suffix Array and Kasai's Algorithm for LCP
// O((N + W) log N) processing, O(1) per query
// Credit: KACTL
// Requires sparse_table
template<typename Str>
struct suffix_array{
	int n;
	vector<int> sa, rank, lcp;
	// sa[i]: indices of suffix of s+delim at position i, rank: inverse of sa, lcp[i]: lcp[0]=0, lcp[i] = lcp of suffices at i-1 and i
	sparse_table<int, function<int(int, int)>> rmq;
	suffix_array(const Str &s, int lim = 256): n(int(s.size()) + 1), sa(n), rank(n), lcp(n){
		int n = int(s.size()) + 1, k = 0, a, b;
		vector<int> x(s.begin(), s.end() + 1), y(n), ws(max(n, lim));
		iota(sa.begin(), sa.end(), 0);
		for(int j = 0, p = 0; p < n; j = max(1, j * 2), lim = p){
			p = j, iota(y.begin(), y.end(), n - j);
			for(int i = 0; i < n; ++ i) if(sa[i] >= j) y[p ++] = sa[i] - j;
			fill(ws.begin(), ws.end(), 0);
			for(int i = 0; i < n; ++ i) ws[x[i]] ++;
			for(int i = 1; i < lim; ++ i) ws[i] += ws[i - 1];
			for(int i = n; i --; ) sa[-- ws[x[y[i]]]] = y[i];
			swap(x, y), p = 1, x[sa[0]] = 0;
			for(int i = 1; i < n; ++ i){
				a = sa[i - 1], b = sa[i];
				x[b] = (y[a] == y[b] && y[a + j] == y[b + j]) ? p - 1 : p ++;
			}
		}
		for(int i = 1; i < n; ++ i) rank[sa[i]] = i;
		for(int i = 0, j; i < n - 1; lcp[rank[i ++]] = k) for(k && k --, j = sa[rank[i] - 1]; s[i + k] == s[j + k]; k++);
		rmq = sparse_table<int, function<int(int, int)>>(lcp.begin(), lcp.end(), [&](int x, int y){ return min(x, y); }, numeric_limits<int>::max());
	}
	int query(int i, int j){ // Find the length of lcp of suffices starting at i and j
		return i == j ? n - 1 - max(i, j) : rmq.query(min(rank[i], rank[j]) + 1, max(rank[i], rank[j]) + 1);
	}
};

// 156485479_5_4
// Z Function / for each position i > 0, returns the length of the longest prefix which is also a prefix starting at i
// O(n)
template<typename Str>
vector<int> z_function(const Str &s){
	int n = int(s.size());
	vector<int> z(n);
	for(int i = 1, l = 0, r = 1; i < n; ++ i){
		if(i < r) z[i] = min(r - i, z[i - l]);
		while(i + z[i] < n && s[z[i]] == s[i + z[i]]) ++ z[i];
		if(i + z[i] > r) l = i, r = i + z[i];
	}
	return z;
}

// 156485479_5_5
// Aho Corasic Automaton
// O(W) preprocessing, O(L) per query
// Credit: CP-Algorithms
template<typename Str, int lim = 128, typename Str::value_type PCH = '$'>
struct aho_corasic{
	typedef typename Str::value_type Char;
	struct node{
		int par, link = -1, elink = -1;
		Char cpar;
		vector<int> next, go;
		bool isleaf = false;
		node(int par = -1, Char pch = PCH): par(par), cpar(pch), next(lim, -1), go(lim, -1){ }
		long long val = 0;
		bool mark = false;
	};
	vector<node> state = vector<node>(1);
	int insert(const Str &s){
		int u = 0;
		for(auto &c: s){
			if(state[u].next[c] == -1){
				state[u].next[c] = int(state.size());
				state.emplace_back(u, c);
			}
			u = state[u].next[c];
		}
		state[u].isleaf = true;
		return u;
	}
	int get_link(int u){
		if(state[u].link == -1){
			if(!u || !state[u].par) state[u].link = 0;
			else state[u].link = go(get_link(state[u].par), state[u].cpar);
		}
		return state[u].link;
	}
	int get_elink(int u){
		if(state[u].elink == -1){
			if(!u || !get_link(u)) state[u].elink = 0;
			else if(state[get_link(u)].isleaf) state[u].elink = get_link(u);
			else state[u].elink = get_elink(get_link(u));
		}
		return state[u].elink;
	}
	int go(int u, const Char &c){
		if(state[u].go[c] == -1){
			if(state[u].next[c] != -1) state[u].go[c] = state[u].next[c];
			else state[u].go[c] = u ? go(get_link(u), c) : u;
		}
		return state[u].go[c];
	}
	int go(const Str &s){
		int u = 0;
		for(auto &c: s) u = go(u, c);
		return u;
	}
	void print(int u, Str s = ""){
		cout << "Node " << u << ": par = " << state[u].par << ", cpar = " << state[u].cpar << ", string: " << s << "\n";
		for(int c = 0; c < lim; ++ c){
			if(state[u].next[c] != -1){
				cout << u << " => ";
				print(state[u].next[c], s + Str{c});
			}
		}
	}
};

// 156485479_5_6
// Prefix Function / Prefix Automaton
// O(N) each
template<typename Str>
vector<int> prefix_function(const Str &s){
	int n = int(s.size());
	vector<int> p(n);
	for(int i = 1; i < n; ++ i){
		int j = p[i - 1];
		while(j > 0 && s[i] != s[j]) j = p[j - 1];
		if(s[i] == s[j]) ++ j;
		p[i] = j;
	}
	return p;
}
template<typename Str, int lim = 128>
pair<vector<int>, vector<vector<int>>> prefix_automaton(const Str &s){
	vector<int> p = prefix_function(s);
	int n = int(s.size());
	vector<vector<int>> aut(n, vector<int>(lim + 1));
	for(int i = 0; i < n; ++ i) for(int c = 0; c <= lim; ++ c){
		if(i > 0 && c != s[i]) aut[i][c] = aut[p[i - 1]][c];
		else aut[i][c] = i + (c == s[i]);
	}
	return {p, aut};
}

// 156485479_5_7
// Polynomial Hash
// O(n) processing, O(log n) for lcp, O(n) for search, O(1) for query
template<typename Str>
struct polyhash: vector<vector<long long>>{
	const int lim;
	const long long base, mod;
	vector<long long> p;
	polyhash(int lim, long long mod): lim(lim), p(lim, 1), mod(mod), base(rngll() % (long long)(0.4 * mod) + 0.3 * mod){
		for(int i = 1; i < lim; ++ i) p[i] = p[i - 1] * base % mod;
	}
	void insert(const Str &s){
		this->emplace_back(s.size() + 1);
		for(int i = 0; i < int(s.size()); ++ i) this->back()[i + 1] = (this->back()[i] * base + s[i]) % mod;
	}
	void extend(typename Str::value_type c, int i = 0){
		(*this)[i].push_back(((*this)[i].back() * base + c) % mod);
	}
	long long query(int ql, int qr, int i = 0){
		return ((*this)[i][qr] - (*this)[i][ql] * p[qr - ql] % mod + mod) % mod;
	}
	int lcp(int i, int j, int posi = 0, int posj = 0){ // returns the length
		int low = 0, high = min(int((*this)[i].size()) - posi, int((*this)[j].size()) - posj);
		while(high - low > 1){
			int mid = low + (high - low >> 1);
			query(posi, posi + mid, i) == query(posj, posj + mid, j) ? low = mid : high = mid;
		}
		return low;
	}
	int lcs(int i, int j, int posi, int posj){
		int low = 0, high = min(posi, posj) + 1;
		while(high - low > 1){
			int mid = low + (high - low >> 1);
			query(posi - mid, posi, i) == query(posj - mid, posj, j) ? low = mid : high = mid;
		}
		return low;
	}
	vector<int> search(const Str &s, bool FIND_ALL = true, int i = 0){
		int len = s.size();
		long long v = 0;
		for(auto c: s) v = (v * base + c) % mod;
		vector<int> res;
		for(int j = 0; j + len < (*this)[i].size(); ++ j) if(v == query(j, j + len, i)){
			res.push_back(j);
			if(!FIND_ALL) break;
		}
		return res;
	}
};
template<typename Str>
struct double_polyhash{
	pair<polyhash<Str>, polyhash<Str>> h;
	double_polyhash(int N, long long mod): h{polyhash<Str>(N, mod), polyhash<Str>(N, mod)}{ }
	void insert(const Str &s){
		h.first.insert(s), h.second.insert(s);
	}
	void extend(typename Str::value_type c, int i = 0){
		h.first.extend(c, i), h.second.extend(c, i);
	}
	pair<long long, long long> query(int ql, int qr, int i = 0){
		return {h.first.query(ql, qr, i), h.second.query(ql, qr, i)};
	}
	int lcp(int i, int j, int posi = 0, int posj = 0){ // returns the length
		int low = 0, high = min(int(h.first[i].size()) - posi, int(h.first[j].size()) - posj);
		while(high - low > 1){
			int mid = low + high >> 1;
			query(posi, posi + mid, i) == query(posj, posj + mid, j) ? low = mid : high = mid;
		}
		return low;
	}
	vector<int> search(const Str &s, bool FIND_ALL = true, int i = 0){
		int len = s.size();
		pair<long long, long long> v;
		for(auto c: s) v = {(v.first * h.first.base + c) % h.first.mod, (v.second * h.second.base + c) % h.second.mod};
		vector<int> res;
		for(int j = 0; j + len < h.first[i].size(); ++ j) if(v == query(j, j + len, i)){
			res.push_back(j);
			if(!FIND_ALL) break;
		}
		return res;
	}
};

// 156485479_5_8
// Suffix Automaton
// O(log ALP_SIZE) per extend call
template<typename Str>
struct suffix_automaton{
	typedef typename Str::value_type Char;
	struct node{
		int len = 0, link = -1, firstpos = -1;
		bool isclone = false;
		map<Char, int> next;
		vector<int> invlink;
		int cnt = -1;
	};
	vector<node> state = vector<node>(1);
	int last = 0;
	suffix_automaton(const Str &s){
		state.reserve(s.size());
		for(auto c: s) insert(c);
	}
	void insert(Char c){
		int cur = state.size();
		state.push_back({state[last].len + 1, -1, state[last].len});
		int p = last;
		while(p != -1 && !state[p].next.count(c)){
			state[p].next[c] = cur;
			p = state[p].link;
		}
		if(p == -1) state[cur].link = 0;
		else{
			int q = state[p].next[c];
			if(state[p].len + 1 == state[q].len) state[cur].link = q;
			else{
				int clone = state.size();
				state.push_back({state[p].len + 1, state[q].link, state[q].firstpos, true, state[q].next});
				while(p != -1 && state[p].next[c] == q){
					state[p].next[c] = clone;
					p = state[p].link;
				}
				state[q].link = state[cur].link = clone;
			}
		}
		last = cur;
	}
	void print(){
		for(int u = 0; u < state.size(); ++ u){
			cout << "--------------------------------\n";
			cout << "Node " << u << ": len = " << state[u].len << ", link = " << state[u].link;
			cout << ", firstpos = " << state[u].firstpos << ", cnt = " << state[u].cnt;
			cout << ", isclone = " << state[u].isclone;
			cout << "\ninvlink = " << state[u].invlink << "next = " << state[u].next;
			cout << "--------------------------------" << endl;
		}
	}
	pair<int, int> match(const Str &s){ // (Length of the longest prefix of s, state)
		int u = 0;
		for(int i = 0; i < s.size(); ++ i){
			if(!state[u].next.count(s[i])) return {i, u};
			u = state[u].next[s[i]];
		}
		return {s.size(), u};
	}
	vector<long long> distinct_substr_cnt(){
		vector<long long> dp(state.size());
		function<long long(int)> solve = [&](int u){
			if(dp[u]) return dp[u];
			dp[u] = 1;
			for(auto [c, v]: state[u].next) dp[u] += solve(v);
			return dp[u];
		};
		solve(0);
		return dp;
	}
	vector<long long> distinct_substr_len(){
		vector<long long> res(state.size()), dp(state.size());
		function<long long(int)> solve = [&](int u){
			if(dp[u]) return res[u];
			dp[u] = 1;
			for(auto [c, v]: state[u].next){
				res[u] += solve(v) + dp[v];
				dp[u] += dp[v];
			}
			return res[u];
		};
		solve(0);
		return res;
	}
	pair<Str, int> k_th_substr(long long k){
		vector<long long> dp(distinct_substr_cnt());
		assert(dp[0] >= k && k);
		Str res;
		int u = 0;
		for(; -- k; ) for(auto [c, v]: state[u].next){
			if(k > dp[v]) k -= dp[v];
			else{
				res.push_back(c);
				u = v;
				break;
			}
		}
		return {res, u};
	}
	pair<Str, int> smallest_substr(int length){
		Str res;
		int u = 0;
		for(; length --; ){
			assert(!state[u].next.empty());
			auto it = state[u].next.begin();
			res.push_back(it->first);
			u = it->second;
		}
		return {res, u};
	}
	pair<int, int> find_first(const Str &s){ // length, pos
		auto [l, u] = match(s);
		return {l, state[u].firstpos - int(s.size()) + 1};
	}
	void process_invlink(){
		for(int u = 1; u < int(state.size()); ++ u) state[state[u].link].invlink.push_back(u);
	}
	vector<int> find_all(const Str &s, bool invlink_init = false){
		auto [l, u] = match(s);
		if(l < int(s.size())) return{};
		vector<int> res;
		if(!invlink_init) process_invlink();
		function<void(int)> solve = [&](int u){
			if(!state[u].isclone) res.push_back(state[u].firstpos);
			for(auto v: state[u].invlink) solve(v);
		};
		solve(u);
		for(auto &x: res) x += 1 - int(s.size());
		sort(res.begin(), res.end());
		return res;
	}
	Str lcs(const Str &s){
		int u = 0, l = 0, best = 0, bestpos = 0;
		for(int i = 0; i < int(s.size()); ++ i){
			while(u && !state[u].next.count(s[i])){
				u = state[u].link;
				l = state[u].len;
			}
			if(state[u].next.count(s[i])){
				u = state[u].next[s[i]];
				++ l;
			}
			if(l > best){
				best = l;
				bestpos = i;
			}
		}
		return {s.begin() + bestpos - best + 1, s.begin() + bestpos + 1};
	}
	vector<int> process_lcs(const Str &s){ // list of length ending at the pos
		int u = 0, l = 0;
		vector<int> res(s.size());
		for(int i = 0; i < int(s.size()); ++ i){
			while(u && !state[u].next.count(s[i])){
				u = state[u].link;
				l = state[u].len;
			}
			if(state[u].next.count(s[i])){
				u = state[u].next[s[i]];
				++ l;
			}
			res[i] = l;
		}
		return res;
	}
	void process_cnt(bool invlink_init = false){
		for(int u = 0; u < int(state.size()); ++ u) state[u].cnt = (!state[u].isclone && u);
		if(!invlink_init) process_invlink();
		function<void(int)> solve = [&](int u){
			for(auto v: state[u].invlink){
				solve(v);
				state[u].cnt += state[v].cnt;
			}
		};
		solve(0);
	}
	int count(const string &s){
		assert(state[0].cnt != -1);
		return state[match(s).second].cnt;
	}
};
template<typename Str>
Str lcs(vector<Str> a){
	swap(a[0], *min_element(a.begin(), a.end(), [](const Str &s, const Str &t){ return s.size() < t.size(); }));
	vector<int> res(a[0].size());
	iota(res.begin(), res.end(), 1);
	for(int i = 1; i < a.size(); ++ i){
		auto t = suffix_automaton(a[i]).process_lcs(a[0]);
		for(int j = 0; j < int(a[0].size()); ++ j) ctmin(res[j], t[j]);
	}
	int i = max_element(res.begin(), res.end()) - res.begin();
	return {a[0].begin() + i + 1 - res[i], a[0].begin() + i + 1};
}

// 156485479_5_9
// Suffix Tree
// O(Alphabet_size * Length)
// Credit: KACTL
// TODO: Find a problem requiring suffix tree
// TODO: Change this into vector format
struct suffix_tree{
	enum { N = 200010, ALPHA = 26 }; // N ~ 2*maxlen+10
	int toi(char c){ return c - 'a'; }
	string a; // v = cur node, q = cur position
	int t[N][ALPHA],l[N],r[N],p[N],s[N],v=0,q=0,m=2;
	void ukkadd(int i, int c) { suff:
		if(r[v]<=q){
			if (t[v][c]==-1) { t[v][c]=m;  l[m]=i;
				p[m++]=v; v=s[v]; q=r[v];  goto suff; }
			v=t[v][c]; q=l[v];
		}
		if (q==-1 || c==toi(a[q])) q++; else {
			l[m+1]=i;  p[m+1]=m;  l[m]=l[v];  r[m]=q;
			p[m]=p[v];  t[m][c]=m+1;  t[m][toi(a[q])]=v;
			l[v]=q;  p[v]=m;  t[p[m]][toi(a[l[m]])]=m;
			v=s[p[m]];  q=l[m];
			while (q<r[m]) { v=t[v][toi(a[q])];  q+=r[v]-l[v]; }
			if (q==r[m])  s[m]=v;  else s[m]=m+2;
			q=r[v]-(q-r[m]);  m+=2;  goto suff;
		}
	}
	suffix_tree(string a) : a(a) {
		fill(r,r+N,sz(a));
		memset(s, 0, sizeof s);
		memset(t, -1, sizeof t);
		fill(t[1],t[1]+ALPHA,0);
		s[0] = 1; l[0] = l[1] = -1; r[0] = r[1] = p[0] = p[1] = 0;
		rep(i,0,sz(a)) ukkadd(i, toi(a[i]));
	}

	// example: find longest common substring (uses ALPHA = 28)
	pii best;
	int lcs(int node, int i1, int i2, int olen) {
		if (l[node] <= i1 && i1 < r[node]) return 1;
		if (l[node] <= i2 && i2 < r[node]) return 2;
		int mask = 0, len = node ? olen + (r[node] - l[node]) : 0;
		rep(c,0,ALPHA) if (t[node][c] != -1)
			mask |= lcs(t[node][c], i1, i2, len);
		if (mask == 3)
			best = max(best, {len, r[node] - len});
		return mask;
	}
	static pii LCS(string s, string t) {
		SuffixTree st(s + (char)('z' + 1) + t + (char)('z' + 2));
		st.lcs(0, sz(s), sz(s) + 1 + sz(t), 0);
		return st.best;
	}
};
	
// 156485479_5_10
// Palindrome Tree / Eertree
// O(len * lim)
template<typename Str, int lim = 128>
struct palindrome_tree{
	typedef typename Str::value_type Char;
	struct node{
		int len, link, depth, cnt = 0; // length, suffix link, depth of the node via suffix link
		vector<int> next;
		node(int len, int link, int depth): len(len), link(link), depth(depth), next(lim){ };
	};
	vector<int> s = vector<int>{-1};
	vector<node> state = vector<node>{{0, 1, 0}, {-1, 0, 0}};
	int lps = 1; // Node containing the longest palindromic suffix
	long long count = 0; // Number of non-empty palindromic substrings
	palindrome_tree(){ }
	palindrome_tree(const Str &s){
		for(auto c: s) push_back(c);
	}
	int get_link(int u){
		while(s[int(s.size()) - state[u].len - 2] != s.back()) u = state[u].link;
		return u;
	}
	void push_back(Char c){
		s.push_back(c), lps = get_link(lps);
		if(!state[lps].next[c]){
			state.push_back({state[lps].len + 2, state[get_link(state[lps].link)].next[c], state[state[get_link(state[lps].link)].next[c]].depth + 1});
			state[lps].next[c] = int(state.size()) - 1;
		}
		lps = state[lps].next[c], count += state[lps].depth, ++ state[lps].cnt;
	}
	void init_cnt(){
		function<void(int)> dfs = [&](int u){
			for(int c = 0; c < lim; ++ c) if(state[u].next[c]){
				dfs(state[u].next[c]);
				state[u].cnt += state[state[u].next[c]].cnt;
			}
		};
		dfs(0), dfs(1);
	}
	void print(){
		vector<pair<int, string>> q{{1, ""}, {0, ""}};
		while(!q.empty()){
			int u;
			string s;
			tie(u, s) = q.back(); q.pop_back();
			auto m = state[u];
			cerr << "Node " << u << " \"" << s << "\"\nlen = " << m.len << ", link = " << m.link << ", depth = " << m.depth << "\nnext: ";
			for(int c = 0; c < lim; ++ c) if(m.next[c]) cerr << "(" << char(c) << " -> " << m.next[c] << ") ";
			cerr << "\n\n";
			for(int c = lim - 1; c >= 0; -- c) if(m.next[c]) q.push_back({m.next[c], u == 1 ? string{char(c)} : char(c) + s + char(c)});
		}
	}
};

// 156485479_5_11
// Levenshtein Automaton

// 156485479_5_12
// Burrows Wheeler Transform
// O(|S| log |S|)
// Take all non-empty suffices of S+delim, sort it, then take the last characters.
// Requires sparse_table and suffix_array
template<typename Str>
Str bwt(const Str &s, typename Str::value_type delim = '$'){
	auto sa = suffix_array(s);
	Str res(s.size() + 1, delim);
	for(int i = 0; i <= int(s.size()); ++ i) res[i] = sa.sa[i] ? s[sa.sa[i] - 1] : delim;
	return res;
}

// Inverse Transform
// O(|S|)
// delim must be smaller than the characters on s
template<typename Str, int lim = 128>
Str ibwt(const Str &s, typename Str::value_type delim = '$'){
	int n = int(s.size());
	vector<int> t(lim), next(n);
	for(auto &c: s) ++ t[c + 1];
	for(int i = 1; i < lim; ++ i) t[i] += t[i - 1];
	for(int i = 0; i < n; ++ i) next[t[s[i]] ++] = i;
	int cur = next[0];
	Str res(n - 1, delim);
	for(int i = 0; i < n - 1; ++ i) res[i] = s[cur = next[cur]];
	return move(res);
};

// 156485479_5_13
// Main Lorentz Algorithm ( Find All Tandem ( Square ) Substrings )
// No two intervals with the same period intersect or touch.
// O(n log n)
// Credit: Benq
// Requires sparse_table and suffix_array
template<typename Str>
vector<array<int,3>> main_lorentz(const Str &s){
	int n = int(s.size());
	suffix_array sa(s), rsa(Str(s.rbegin(), s.rend()));
	vector<array<int,3>> res;
	for(int p = 1; p << 1 <= n; ++ p){ // do in O(n / p) for period p
		for(int i = 0, lst = -1; i + p <= n; i += p){
			int l = i - rsa.query(n - i - p, n - i), r = i - p + sa.query(i, i + p);
			if(l > r || l == lst) continue;
			res.push_back({lst = l, r + 1, p});
		} // s.substr(i, p) == s.substr(i + p, p) for each i in [l, r)
	}
	return res;
} 

// 156485479_6_1
// 2D Geometry Classes
template<typename T = long long> struct point{
	T x, y;
	int ind;
	template<typename U> point(const point<U> &otr): x(otr.x), y(otr.y), ind(otr.ind){ }
	template<typename U = T, typename V = T> point(U x = 0, V y = 0, int ind = -1): x(x), y(y), ind(ind){ }
	template<typename U> explicit operator point<U>() const{ return point<U>(static_cast<U>(x), static_cast<U>(y)); }
	T operator*(const point &otr) const{ return x * otr.x + y * otr.y; }
	T operator^(const point &otr) const{ return x * otr.y - y * otr.x; }
	point operator+(const point &otr) const{ return {x + otr.x, y + otr.y}; }
	point &operator+=(const point &otr){ return *this = *this + otr; }
	point operator-(const point &otr) const{ return {x - otr.x, y - otr.y}; }
	point &operator-=(const point &otr){ return *this = *this - otr; }
	point operator-() const{ return {-x, -y}; }
#define scalarop_l(op) friend point operator op(const T &c, const point &p){ return {c op p.x, c op p.y}; }
	scalarop_l(+) scalarop_l(-) scalarop_l(*) scalarop_l(/)
#define scalarop_r(op) point operator op(const T &c) const{ return {x op c, y op c}; }
	scalarop_r(+) scalarop_r(-) scalarop_r(*) scalarop_r(/)
#define scalarapply(op) point &operator op(const T &c){ return *this = *this op c; }
	scalarapply(+=) scalarapply(-=) scalarapply(*=) scalarapply(/=)
#define compareop(op) bool operator op(const point &otr) const{ return tie(x, y) op tie(otr.x, otr.y); }
	compareop(>) compareop(<) compareop(>=) compareop(<=) compareop(==) compareop(!=)
#undef scalarop_l
#undef scalarop_r
#undef scalarapply
#undef compareop
	double norm() const{ return sqrt(x * x + y * y); }
	T squared_norm() const{ return x * x + y * y; }
	double arg() const{ return atan2(y, x); } // [-pi, pi]
	point<double> unit() const{ return point<double>(x, y) / norm(); }
	point perp() const{ return {-y, x}; }
	point<double> normal() const{ return perp().unit(); }
	point<double> rotate(const double &theta) const{ return point<double>(x * cos(theta) - y * sin(theta), x * sin(theta) + y * cos(theta)); }
	point reflect_x() const{ return {x, -y}; }
	point reflect_y() const{ return {-x, y}; }
	point reflect(const point &o) const{ return {2 * o.x - x, 2 * o.y - y}; }
	bool operator||(const point &otr) const{ return !(*this ^ otr); }
};
template<typename T> istream &operator>>(istream &in, point<T> &p){ return in >> p.x >> p.y; }
template<typename T> ostream &operator<<(ostream &out, const point<T> &p){ return out << "(" << p.x << ", " << p.y << ")"; }
template<typename T> double distance(const point<T> &p, const point<T> &q){ return (p - q).norm(); }
template<typename T> T squared_distance(const point<T> &p, const point<T> &q){ return (p - q).squared_norm(); }
template<typename T, typename U, typename V> T ori(const point<T> &p, const point<U> &q, const point<V> &r){ return (q - p) ^ (r - p); }
template<typename IT> auto doubled_signed_area(IT begin, IT end){
	typename iterator_traits<IT>::value_type s = 0, init = *begin;
	for(; begin != prev(end); ++ begin) s += *begin ^ *next(begin);
	return s + (*begin ^ init);
}
template<typename T = long long> struct line{
	point<T> p, d; // p + d*t
	template<typename U = T, typename V = T> line(point<U> p = {0, 0}, point<V> q = {0, 0}, bool Two_Points = true): p(p), d(Two_Points ? q - p : q){ }
	template<typename U> line(point<U> d): p(), d(static_cast<point<T>>(d)){ }
	line(T a, T b, T c): p(a ? -c / a : 0, !a && b ? -c / b : 0), d(-b, a){ }
	template<typename U> explicit operator line<U>() const{ return line<U>(point<U>(p), point<U>(d), false); }
	point<T> q() const{ return p + d; }
	bool degen() const{ return d == point<T>(); }
	tuple<T, T, T> coef() const{ return {d.y, -d.x, d.perp() * p}; } // d.y (X - p.x) - d.x (Y - p.y) = 0
	bool operator||(const line<T> &L) const{ return d || L.d; }
};
template<typename T> bool on_line(const point<T> &p, const line<T> &L){
	if(L.degen()) return p == L.p;
	return (p - L.p) || L.d;
}
template<typename T> bool on_ray(const point<T> &p, const line<T> &L){
	if(L.degen()) return p == L.p;
	auto a = L.p - p, b = L.q() - p;
	return (a || b) && a * L.d <= 0;
}
template<typename T> bool on_segment(const point<T> &p, const line<T> &L){
	if(L.degen()) return p == L.p;
	auto a = L.p - p, b = L.q() - p;
	return (a || b) && a * b <= 0;
}
template<typename T> double distance_to_line(const point<T> &p, const line<T> &L){
	if(L.degen()) return distance(p, L.p);
	return abs((p - L.p) ^ L.d) / L.d.norm();
}
template<typename T> double distance_to_ray(const point<T> &p, const line<T> &L){
	if((p - L.p) * L.d <= 0) return distance(p, L.p);
	return distance_to_line(p, L);
}
template<typename T> double distance_to_segment(const point<T> &p, const line<T> &L){
	if((p - L.p) * L.d <= 0) return distance(p, L.p);
	if((p - L.q()) * L.d >= 0) return distance(p, L.q());
	return distance_to_line(p, L);
}
template<typename T> point<double> projection(const point<T> &p, const line<T> &L){ return static_cast<point<double>>(L.p) + (L.degen() ? point<double>() : (p - L.p) * L.d / L.d.norm() * static_cast<point<double>>(L.d)); }
template<typename T> point<double> reflection(const point<T> &p, const line<T> &L){ return 2.0 * projection(p, L) - static_cast<point<double>>(p); }
template<typename T> point<double> closest_point_on_segment(const point<T> &p, const line<T> &L){ return (p - L.p) * L.d <= 0 ? static_cast<point<double>>(L.p) : ((p - L.q()) * L.d >= 0 ? static_cast<point<double>>(L.q()) : projection(p, L)); }
template<int TYPE> struct EndpointChecker{ };
// For rays
template<> struct EndpointChecker<0>{ template<typename T> bool operator()(const T& a, const T& b) const{ return true; } }; // For ray
// For closed end
template<> struct EndpointChecker<1>{ template<typename T> bool operator()(const T& a, const T& b) const{ return a <= b; } }; // For closed end
// For open end
template<> struct EndpointChecker<2>{ template<typename T> bool operator()(const T& a, const T& b) const{ return a < b; } }; // For open end
// Assumes parallel lines do not intersect
template<int LA, int LB, int RA, int RB, typename T> pair<bool, point<double>> intersect_no_parallel_overlap(const line<T> &L, const line<T> &M){
	auto s = L.d ^ M.d;
	if(!s) return {false, point<double>()};
	auto ls = (M.p - L.p) ^ M.d, rs = (M.p - L.p) ^ L.d;
	if(s < 0) s = -s, ls = -ls, rs = -rs;
	bool intersect = EndpointChecker<LA>()(decltype(ls)(0), ls) && EndpointChecker<LB>()(ls, s) && EndpointChecker<RA>()(decltype(rs)(0), rs) && EndpointChecker<RB>()(rs, s);
	return {intersect, static_cast<point<double>>(L.p) + 1.0 * ls / s * static_cast<point<double>>(L.d)};
}
// Assumes parallel lines do not intersect
template<typename T> pair<bool, point<double>> intersect_closed_segments_no_parallel_overlap(const line<T> &L, const line<T> &M){
	return intersect_no_parallel_overlap<1, 1, 1, 1>(L, M);
}
// Assumes nothing
template<typename T> pair<bool, line<double>> intersect_closed_segments(const line<T> &L, const line<T> &M){
	auto s = L.d ^ M.d, ls = (M.p - L.p) ^ M.d;
	if(!s){
		if(ls) return {false, line<double>()};
		auto Lp = L.p, Lq = L.q(), Mp = M.p, Mq = M.q();
		if(Lp > Lq) swap(Lp, Lq);
		if(Mp > Mq) swap(Mp, Mq);
		line<double> res(max(Lp, Mp), min(Lq, Mq));
		return {res.d >= point<double>(), res};
	}
	auto rs = (M.p - L.p) ^ L.d;
	if(s < 0) s = -s, ls = -ls, rs = -rs;
	bool intersect = 0 <= ls && ls <= s && 0 <= rs && rs <= s;
	return {intersect, line<double>(static_cast<point<double>>(L.p) + 1.0 * ls / s * static_cast<point<double>>(L.d), point<double>())};
}
template<typename T> double distance_between_rays(const line<T> &L, const line<T> &M){
	if(L || M){
		if(L.d * M.d >= 0 || (M.p - L.p) * M.d <= 0) return distance_to_line(L.p, M);
		else return distance(L.p, M.p);
	}
	else{
		if(intersect_no_parallel_overlap<1, 0, 1, 0, long long>(L, M).first) return 0;
		else return min(distance_to_ray(L.p, M), distance_to_ray(M.p, L));
	}
}
template<typename T> double distance_between_segments(const line<T> &L, const line<T> &M){
	if(intersect_closed_segments(L, M).first) return 0;
	return min({distance_to_segment(L.p, M), distance_to_segment(L.q(), M), distance_to_segment(M.p, L), distance_to_segment(M.q(), L)});
}
template<typename P> struct compare_by_angle{
	const P origin;
	compare_by_angle(const P &origin = P()): origin(origin){ }
	bool operator()(const P &p, const P &q) const{ return ori(origin, p, q) > 0; }
};
template<typename It, typename P> void sort_by_angle(It begin, It end, const P &origin){
	begin = partition(begin, end, [&origin](const decltype(*begin) &point){ return point == origin; });
	auto pivot = partition(begin, end, [&origin](const decltype(*begin) &point) { return point > origin; });
	compare_by_angle<P> cmp(origin);
	sort(begin, pivot, cmp), sort(pivot, end, cmp);
}
/* Short Descriptions
struct point{
	T x, y;
	int ind;
	double norm()
	T squared_norm()
	double arg()
	point<double> unit()
	point perp()
	point<double> normal()
	point<double> rotate(double theta)
	point reflect_x()
	point reflect_y()
	point reflect(point o)
	bool operator||(point otr)
};
double distance(point p, point q)
T squared_distance(point p, point q)
T ori(point p, point q, point r){ return (q - p) ^ (r - p); }
auto doubled_signed_area(IT begin, IT end)
struct line{
	point p, d; // p + d*t
	line(point p = {0, 0}, point<V> q = {0, 0}, bool Two_Points = true)
	// two_points: pass through p and q
	// else: pass through p, slope q
	line(point<U> d) // pass through origin, slope d
	line(T a, T b, T c) // declare with ax + by + c = 0
	point<T> q()
	bool degen()
	tuple<T, T, T> coef()// d.y (X - p.x) - d.x (Y - p.y) = 0
};
bool on_line(point, line)
bool on_ray(point, line)
bool on_segment(point, line)
double distance_to_line(point, line)
double distance_to_ray(point, line)
double distance_to_segment(point, line)
point<double> projection(point, line)
point<double> reflection(point, line)
point<double> closest_point_on_segment(point, line)
// Endpoints: (0: rays), (1: closed), (2: open)
// Assumes parallel lines do not intersect
pair<bool, point<double>> intersect_no_parallel_overlap<EP1, EP2, EP3, EP4>(line, line)
// Assumes parallel lines do not intersect
pair<bool, point<double>> intersect_closed_segments_no_parallel_overlap(line, line)
// Assumes nothing
pair<bool, line<double>> intersect_closed_segments(line, line)
double distance_between_rays(line, line)
double distance_between_segments(line, line)
struct compare_by_angle
void sort_by_angle(It begin, It end, const P &origin) */
// 3D Geometry Classes
template<typename T = long long> struct point{
	T x, y, z;
	int ind;
	template<typename U> point(const point<U> &otr): x(otr.x), y(otr.y), z(otr.z), ind(otr.ind){ }
	template<typename U, typename V, typename W> point(const tuple<U, V, W> &p): x(get<0>(p)), y(get<1>(p)), z(get<2>(p)){ }
	template<typename U = T, typename V = T, typename W = T> point(U x = 0, V y = 0, W z = 0): x(x), y(y), z(z){ }
	template<typename U> explicit operator point<U>() const{ return point<U>(static_cast<U>(x), static_cast<U>(y), static_cast<U>(z)); }
	T operator*(const point &otr) const{ return x * otr.x + y * otr.y + z * otr.z; }
	point operator^(const point &otr) const{ return point(y * otr.z - z * otr.y, z * otr.x - x * otr.z, x * otr.y - y * otr.x); }
	point operator+(const point &otr) const{ return point(x + otr.x, y + otr.y, z + otr.z); }
	point &operator+=(const point &otr){ return *this = *this + otr; }
	point operator-(const point &otr) const{ return point(x - otr.x, y - otr.y, z - otr.z); }
	point &operator-=(const point &otr){ return *this = *this - otr; }
	point operator*(const T &c) const{ return point(x * c, y * c, z * c); }
	point &operator*=(const T &c) { return *this = *this * c; }
	point operator/(const T &c) const{ return point(x / c, y / c, z / c); }
	point &operator/=(const T &c) { return *this = *this / c; }
	point operator-() const{ return point(-x, -y, -z); }
	bool operator<(const point &otr) const{ return tie(x, y, z) < tie(otr.x, otr.y, otr.z); }
	bool operator>(const point &otr) const{ return tie(x, y, z) > tie(otr.x, otr.y, otr.z); }
	bool operator<=(const point &otr) const{ return tie(x, y, z) <= tie(otr.x, otr.y, otr.z); }
	bool operator>=(const point &otr) const{ return tie(x, y, z) >= tie(otr.x, otr.y, otr.z); }
	bool operator==(const point &otr) const{ return tie(x, y, z) == tie(otr.x, otr.y, otr.z); }
	bool operator!=(const point &otr) const{ return tie(x, y, z) != tie(otr.x, otr.y, otr.z); }
	double norm() const{ return sqrt(x * x + y * y + z * z); }
	T squared_norm() const{ return x * x + y * y + z * z; }
	point<double> unit() const{ return point<double>(x, y, z) / norm(); }
	point reflect_x() const{ return point(x, -y, -z); }
	point reflect_y() const{ return point(-x, y, -z); }
	point reflect_z() const{ return point(-x, -y, z); }
	point reflect_xy() const{ return point(x, y, -z); }
	point reflect_yz() const{ return point(-x, y, z); }
	point reflect_zx() const{ return point(x, -y, z); }
	bool operator||(const point &otr) const{ return *this ^ otr == point(); }
};
template<typename T> point<T> operator*(const T &c, const point<T> &p){ return point<T>(c * p.x, c * p.y, c * p.z); }
template<typename T> istream &operator>>(istream &in, point<T> &p){ return in >> p.x >> p.y >> p.z; }
template<typename T> ostream &operator<<(ostream &out, const point<T> &p){ return out << "(" << p.x << ", " << p.y << ", " << p.z << ")"; }
template<typename T> double distance(const point<T> &p, const point<T> &q){ return (p - q).norm(); }
template<typename T> T squared_distance(const point<T> &p, const point<T> &q){ return (p - q).squared_norm(); }
template<typename T, typename U, typename V, typename W> T ori(const point<T> &p, const point<U> &q, const point<V> &r, const point<W> &s){ return ((q - p) ^ (r - p)) * (s - p); }
template<typename T> T sextupled_signed_volume(const vector<vector<point<T>>> &arr){
	T s = 0;
	for(const auto &face: arr){
		assert(int(face.size()) >= 3);
		s += (face[int(face.size()) - 2] ^ face[int(face.size()) - 1]) * face[0] + (face[int(face.size()) - 1] ^ face[0]) * face[1];
		for(int i = 0; i + 3 <= int(face.size()); ++ i) s += (face[i] ^ face[i + 1]) * face[i + 2];
	}
	return s;
}
template<typename T = long long> struct line{
	point<T> p, d; // p + d*t
	template<typename U = T, typename V = T> line(point<U> p = {0, 0}, point<V> q = {0, 0}, bool Two_Points = true): p(p), d(Two_Points ? q - p : q){ }
	template<typename U> line(point<U> d): p(), d(static_cast<point<T>>(d)){ }
	template<typename U> explicit operator line<U>() const{ return line<U>(point<U>(p), point<U>(d), false); }
	point<T> q() const{ return p + d; }
	bool degen() const{ return d == point<T>(); }
	bool operator||(const line<T> &L){ return d || L.d; }
};
template<typename T> bool parallel(const line<T> &L, const line<T> &M){ return L.d ^ M.d == point<T>(); }
template<typename T> bool on_line(const point<T> &p, const line<T> &L){
	if(L.degen()) return p == L.p;
	return (p - L.p) || L.d;
}
template<typename T> bool on_ray(const point<T> &p, const line<T> &L){
	if(L.degen()) return p == L.p;
	auto a = L.p - p, b = L.q() - p;
	return (a || b) && a * L.d <= 0;
}
template<typename T> bool on_segment(const point<T> &p, const line<T> &L){
	if(L.degen()) return p == L.p;
	auto a = L.p - p, b = L.q() - p;
	return (a || b) && a * b <= 0;
}
template<typename T> double distance_to_line(const point<T> &p, const line<T> &L){
	if(L.degen()) return distance(p, L.p);
	return ((p - L.p) ^ L.d).norm() / L.d.norm();
}
template<typename T> double distance_to_ray(const point<T> &p, const line<T> &L){
	if((p - L.p) * L.d <= 0) return distance(p, L.p);
	return distance_to_line(p, L);
}
template<typename T> double distance_to_segment(const point<T> &p, const line<T> &L){
	if((p - L.p) * L.d <= 0) return distance(p, L.p);
	if((p - L.q()) * L.d >= 0) return distance(p, L.q());
	return distance_to_line(p, L);
}
template<typename T> point<double> projection(const point<T> &p, const line<T> &L){ return static_cast<point<double>>(L.p) + (L.degen() ? point<double>() : (p - L.p) * L.d / L.d.norm() * static_cast<point<double>>(L.d)); }
template<typename T> point<double> reflection(const point<T> &p, const line<T> &L){ return 2.0 * projection(p, L) - static_cast<point<double>>(p); }
template<typename T> point<double> closest_point_on_segment(const point<T> &p, const line<T> &L){ return (p - L.p) * L.d <= 0 ? static_cast<point<double>>(L.p) : ((p - L.q()) * L.d >= 0 ? static_cast<point<double>>(L.q()) : projection(p, L)); }

// 156485479_6_2
// Convex Hull and Minkowski Sum
// O(n log n) construction, O(n) if sorted.
template<typename Polygon, int type = 0> // type {0: both, 1: lower, 2: upper}
struct convex_hull: pair<Polygon, Polygon>{ // (Lower, Upper)
	convex_hull(Polygon arr = Polygon(), int type = 0, bool is_sorted = false): type(type){
		if(!is_sorted) sort(arr.begin(), arr.end()), arr.resize(unique(arr.begin(), arr.end()) - arr.begin());
#define ADDP(C, cmp) while(int(C.size()) > 1 && ori(C[int(C.size()) - 2], p, C.back()) cmp 0) C.pop_back(); C.push_back(p);
		for(auto &p: arr){
			if(type < 2){ ADDP(this->first, >=) }
			if(!(type & 1)){ ADDP(this->second, <=) }
		}
		reverse(this->second.begin(), this->second.end());
	}
	Polygon get_hull() const{
		if(type) return type == 1 ? this->first : this->second;
		if(this->first.size() <= 1) return this->first;
		Polygon res(this->first);
		res.insert(res.end(), ++ this->second.begin(), -- this->second.end());
		return res;
	}
	int min_element(const typename Polygon::value_type &p) const{
		assert(p.y >= 0 && !this->first.empty());
		int low = 0, high = this->first.size();
		while(high - low > 2){
			int mid1 = (2 * low + high) / 3, mid2 = (low + 2 * high) / 3;
			p * this->first[mid1] >= p * this->first[mid2] ? low = mid1 : high = mid2;
		}
		int res = low;
		for(int i = low + 1; i < high; i ++) if(p * this->first[res] > p * this->first[i]) res = i;
		return res;
	}
	int max_element(const typename Polygon::value_type &p) const{
		assert(p.y >= 0 && !this->second.empty());
		int low = 0, high = this->second.size();
		while(high - low > 2){
			int mid1 = (2 * low + high) / 3, mid2 = (low + 2 * high) / 3;
			p * this->second[mid1] <= p * this->second[mid2] ? low = mid1 : high = mid2;
		}
		int res = low;
		for(int i = low + 1; i < high; ++ i) if(p * this->second[res] < p * this->second[i]) res = i;
		return res;
	}
	Polygon linearize() const{
		if(type == 1) return this->first;
		if(type == 2){ Polygon res(this->second); reverse(res.begin(), res.end()); return res; }
		if(this->first.size() <= 1) return this->first;
		Polygon res;
		res.reserve(this->first.size() + this->second.size());
		merge(this->first.begin(), this->first.end(), ++ this->second.rbegin(), -- this->second.rend(), back_inserter(res));
		return res;
	}
	convex_hull operator^(const convex_hull &otr) const{ // Convex Hull Merge
		Polygon temp, A = linearize(), B = otr.linearize();
		temp.reserve(A.size() + B.size());
		merge(A.begin(), A.end(), B.begin(), B.end(), back_inserter(temp));
		temp.resize(unique(temp.begin(), temp.end()) - temp.begin());
		return {temp, type, true};
	}
	pair<Polygon, Polygon> get_boundary() const{
		Polygon L(this->first), R(this->second);
		for(int i = int(L.size()) - 1; i > 0; -- i) L[i] -= L[i - 1];
		for(int i = int(R.size()) - 1; i > 0; -- i) R[i] -= R[i - 1];
		return {L, R};
	}
	convex_hull operator+(const convex_hull &otr) const{ // Minkowski Sum
		assert(type == otr.type);
		convex_hull res(Polygon(), type);
		pair<Polygon, Polygon> A(this->get_boundary()), B(otr.get_boundary());
		compare_by_angle<typename Polygon::value_type> cmp;
#define PROCESS(COND, X) \
if(COND && !A.X.empty() && !B.X.empty()){ \
	res.X.reserve(A.X.size() + B.X.size()); \
	res.X.push_back(A.X.front() + B.X.front()); \
	merge(A.X.begin() + 1, A.X.end(), B.X.begin() + 1, B.X.end(), back_inserter(res.X), cmp); \
	for(int i = 1; i < int(res.X.size()); ++ i) res.X[i] += res.X[i - 1]; \
}
		PROCESS(type < 2, first)
		PROCESS(!(type & 1), second)
		return res;
	}
};

// 156485479_6_3
// KD Tree
// O(log n) for randomly distributed points
// Point Class must support x, y, less-compare, equal
template<typename T, typename point>
struct Node{
	static constexpr T inf = 1e9;
	point p; // if this is a leaf, the single point in it
	T xlow = inf, xhigh = -inf, ylow = inf, yhigh = -inf; // bounds
	Node *l = 0, *r = 0;
	Node(vector<point> &&arr): p(arr[0]){
		for(auto p: arr){
			xlow = min(xlow, p.x), xhigh = max(xhigh, p.x);
			ylow = min(ylow, p.y), yhigh = max(yhigh, p.y);
		}
		if(int(arr.size()) > 1){ // split on x if the box is wider than high (not best heuristic...)
			if(xhigh - xlow >= yhigh - ylow) sort(arr.begin(), arr.end());
			else sort(arr.begin(), arr.end(), [](point p, point q){ return p.y < q.y; });
			int mid = int(arr.size()) / 2;// divide by taking half the array for each child (not best performance with many duplicates in the middle)
			l = new Node({begin(arr), begin(arr) + mid});
			r = new Node({arr.begin() + mid, arr.end()});
		}
	}
	T distance(const point &p){ // min squared dist to point p
		T x = min(max(p.x, xlow), xhigh), y = min(max(p.y, ylow), yhigh); 
		return (x - p.x) * (x - p.x) + (y - p.y) * (y - p.y);
	}
};
template<typename T, typename point, bool IGNORE_ITSELF = true>
struct KDTree{
	static constexpr T inf = 1e9;
	Node<T, point> *root;
	template<typename Arr> KDTree(const Arr& arr): root(new Node<T, point>({arr.begin(), arr.end()})){ }
	pair<T, point> search(Node<T, point> *node, const point &p){
		if(!node->l){
			if(IGNORE_ITSELF && p == node->p) return {inf, point()};
			return {(p.x - node->p.x) * (p.x - node->p.x) + (p.y - node->p.y) * (p.y - node->p.y), node->p};
		}
		Node<T, point> *l = node->l, *r = node->r;
		T bl = l->distance(p), br = r->distance(p);
		if(bl > br) swap(br, bl), swap(l, r);
		auto best = search(l, p); // search closest side, other side if needed
		if(br < best.first) best = min(best, search(r, p));
		return best;
	}
	pair<T, point> query(const point &p){
		return search(root, p);
	}
};
struct P{
	long long x, y;
	bool operator<(const P &p) const{
		return x == p.x ? y < p.y : x < p.x;
	}
	bool operator==(const P &p) const{
		return x == p.x && y == p.y;
	}
};
istream &operator>>(istream &in, P &p){
	return in >> p.x >> p.y;
}
ostream &operator<<(ostream &out, const P &p){
	return out << "(" << p.x << ", " << p.y << ")";
}

// 156485479_6_4_1
// Find a Pair of Intersecting Segments

// 156485479_6_4_2
// Find the Closest Pair of Points
// O(N log N)
// Credit: KACTL
// Requires geometry
template<typename T, typename IT>
auto closest_pair(IT begin, IT end){
	using P = typename iterator_traits<IT>::value_type;
	auto a = vector<P>(begin, end);
	sort(a.begin(), a.end(), [](const P &p, const P &q){ return p.y < q.y; });
	if(auto it = a.begin(); (it = adjacent_find(a.begin(), a.end())) != a.end()) return tuple<T, P, P>{0, *it, *next(it)};
	tuple<T, P, P> res{numeric_limits<T>::max(), {}, {}};
	set<P> s;
	int j = 0;
	for(const auto &p: a){
		T d = 1 + sqrt(get<0>(res));
		while(a[j].y <= p.y - d) s.erase(a[j ++]);
		auto low = s.lower_bound({p.x - d, p.y}), high = s.upper_bound({p.x + d, p.y});
		for(; low != high; ++ low) res = min(res, {squared_distance(*low, p), *low, p});
		s.insert(p);
	}
	return res;
}

// 156485479_6_5
// Circle Class
template<typename T>
struct circle{
	T x, y, r;
	template<typename U>
	U sq(U x){
		return x * x;
	}
	T dis(const circle &c) const{
		return sqrt(sq(x - c.x) + sq(y - c.y));
	}
	bool share(const circle &c) const{
		T d = dis(c);
		return abs(r - c.r) <= d && d <= r + c.r;
	}
	vector<pair<double, double>> inter(const circle &c) const{ // Returns the list of intersection points
		if(!share(c)) return {};
		double R2 = sq(x - c.x) + sq(y - c.y);
		double A = 0.5, B = (sq(r) - sq(c.r)) / (2 * R2), C = 0.5 * sqrt(2 * (sq(r) + sq(c.r)) / R2 - sq(sq(r) - sq(c.r)) / sq(R2) - 1);
		return {
			{A * (x + c.x) + B * (c.x - x) + C * (c.y - y), A * (y + c.y) + B * (c.y - y) + C * (x - c.x)}
			, {A * (x + c.x) + B * (c.x - x) - C * (c.y - y), A * (y + c.y) + B * (c.y - y) - C * (x - c.x)}
		};
	}
};

// 156485479_7_1
// Maximum Independent Set
// http://ceur-ws.org/Vol-2098/paper12.pdf
template<int SZ>
bitset<SZ> maximum_independent_set_heuristic(vector<bitset<SZ>> adj){
	int n = int(adj.size()), Est = 0;
	vector<int> k(n), m(n);
	bitset<SZ> V0, S;
	V0.set();
	while(V0.any()){
		for(int u = V0._Find_first(); u < n; u = V0._Find_next(u)){
			static vector<int> cur;
			cur.clear();
			for(int v = adj[u]._Find_first(); v != n; v = adj[u]._Find_next(v)) cur.push_back(v);
			k[u] = int(cur.size()), m[u] = 0;
			for(int i = 0; i < k[u]; ++ i) for(int j = i + 1; j < k[u]; ++ j) if(!adj[cur[i]][cur[j]]) ++ m[u];
		}
		int v0 = V0._Find_first();
		for(int u = V0._Find_next(v0); u < n; u = V0._Find_next(u)){
			if(m[v0] > m[u] || m[v0] == m[u] && k[v0] < k[u]) v0 = u;
		}
		S.set(v0);
		Est += m[v0];
		static vector<int> cur;
		cur.clear();
		for(int u = adj[v0]._Find_first(); u != n; u = adj[v0]._Find_next(u)) cur.push_back(u);
		for(int u = V0._Find_first(); u < n; u = V0._Find_next(u)){
			adj[u].reset(v0);
			for(auto v: cur) adj[u].reset(v);
		}
		adj[v0].reset();
		V0.reset(v0);
		for(auto u: cur) V0.reset(u), adj[u].reset();
	}
	return S;
}

// 156485479_8_1
// Custom Hash Function for unordered_set and unordered map
struct custom_hash{
	static uint64_t splitmix64(uint64_t x){
		// http://xorshift.di.unimi.it/splitmix64.c
		x += 0x9e3779b97f4a7c15;
		x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
		x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
		return x ^ (x >> 31);
	}
	size_t operator()(uint64_t x) const {
		static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
		return splitmix64(x + FIXED_RANDOM);
	}
};
// KACTL Hash Function
# define M_PI 3.141592653589793238462643383279502884L
const int RANDOM = rng();
struct custom_hash{ // To use most bits rather than just the lowest ones:
	static const uint64_t C = long long(2e18 * M_PI) + 71; // large odd number
	long long operator()(long long x) const { return __builtin_bswap64((x^RANDOM)*C); }
};

/*
Speed test results
set                                    1e6: 670ms | 1e7: 10155ms
unordered_set                          1e6: 296ms | 1e7: 4320ms
unordered_set with custom hash         1e6: 358ms | 1e7: 4851ms
unordered_set with custom hash(narut)  1e6: 389ms | 1e7: 4850ms
unordered_set with custom hash(pajen)  1e6: 436ms | 1e7: 5022ms

map                                    1e6: 592ms | 1e7: 10420ms
unordered_map                          1e6: 373ms | 1e7: 4742ms
unordered_map with custom hash         1e6: 389ms | 1e7: 4913ms
unordered_map with custom hash(narut)  1e6: 327ms | 1e7: 4960ms
unordered_map with custom hash(pajen)  1e6: 389ms | 1e7: 4789ms

map           | 1e6: 576ms 31560KB | 5e6: 4757ms 156552KB | 1e7: 10498ms 313280KB
unodered_map  | 1e6: 327ms 32220KB | 5e6: 2121ms 147132KB | 1e7: 4835ms  295068KB
cc_hash_table | 1e6: 249ms 31916KB | 5e6: 2011ms 197140KB | 1e7: 4383ms  394588KB
gp_hash_table | 1e6: 109ms 36720KB | 5e6: 686ms  295516KB | 1e7: ????    MLE

constants
set / map                     : around 4
unordered_set / unordered_map : around 40
cc_hash_table                 : around 35
gp_hash_table                 : around 20
*/

// 156485479_8_2
// Bump Allocator
static char BUFF[220 << 20];
void *operator new(size_t s){
	static size_t i = sizeof BUFF;
	assert(s < i);
	return (void *)&BUFF[i -= s];
}
void operator delete(void *){ }

// 156485479_8_3
// Debugger

// DEBUG BEGIN
#ifdef LOCAL
template<typename L, typename R> ostream &operator<<(ostream &out, const pair<L, R> &p){
	return out << "(" << p.first << ", " << p.second << ")";
}
template<typename Tuple, size_t N> struct TuplePrinter{
	static ostream &print(ostream &out, const Tuple &t){ return TuplePrinter<Tuple, N-1>::print(out, t) << ", " << get<N-1>(t); }
};
template<typename Tuple> struct TuplePrinter<Tuple, 1>{
	static ostream &print(ostream &out, const Tuple& t){ return out << get<0>(t); }
};
template<typename... Args> ostream &print_tuple(ostream &out, const tuple<Args...> &t){
	return TuplePrinter<decltype(t), sizeof...(Args)>::print(out << "(", t) << ")";
}
template<typename ...Args> ostream &operator<<(ostream &out, const tuple<Args...> &t){
	return print_tuple(out, t);
}
template<typename T> ostream &operator<<(enable_if_t<!is_same<T, string>::value, ostream> &out, const T &arr){
	out << "{"; for(auto &x: arr) out << x << ", ";
	return out << (arr.size() ? "\b\b" : "") << "}";
}
template<size_t S> ostream &operator<<(ostream &out, const bitset<S> &b){
	for(int i = 0; i < S; ++ i) out << b[i];
	return out;
}
void debug_out(){ cerr << "\b\b " << endl; }
template<typename Head, typename... Tail>
void debug_out(Head H, Tail... T){ cerr << H << ", ", debug_out(T...); }
void debug2_out(){ cerr << "-----DEBUG END-----\n"; }
template<typename Head, typename... Tail>
void debug2_out(Head H, Tail... T){ cerr << "\n"; for(auto x: H) cerr << x << "\n"; debug2_out(T...); }
#define debug(...) cerr << "[" << #__VA_ARGS__ << "]: ", debug_out(__VA_ARGS__)
#define debug2(...) cerr << "----DEBUG BEGIN----\n[" << #__VA_ARGS__ << "]:", debug2_out(__VA_ARGS__)
#else
#define debug(...) 42
#define debug2(...) 42
#endif
// DEBUG END

// 156485479_8_4
// Random Generators
int rand_int(int low, int high){ // generate random integer in [low, high)
	return int(rng() % (high - low)) + low;
}
double rand_float(double low, double high){
	return rng() * 1.0 / numeric_limits<uint>::max() * (high - low) + low;
}
namespace graph_generator{
	vector<array<int, 2>> generate_tree(int n){
		vector<array<int, 2>> res;
		for(int u = 1; u < n; ++ u) res.push_back({u, rand_int(0, u)});
		return res;
	}
	vector<array<int, 3>> generate_weighted_tree(int n, int wlow = 1, int whigh = 6){
		vector<array<int, 3>> res;
		for(int u = 1; u < n; ++ u) res.push_back({u, rand_int(0, u), rand_int(wlow, whigh)});
		return res;
	}
	vector<array<int, 2>> generate_graph(int n, int m){
		vector<array<int, 2>> res(m);
		for(auto &[u, v]: res) u = rng() % n, v = rng() % n;
		return res;
	}
	vector<array<int, 2>> generate_simply_connected_graph(int n, int m){
		assert(n - 1 <= m && m <= 1LL * n * (n - 1) / 2);
		auto res = generate_tree(n);
		set<array<int, 2>> s;
		for(auto [u, v]: res) s.insert({u, v}), s.insert({v, u});
		for(int rep = n - 1; rep < m; ++ rep){
			int u, v;
			do{
				u = rng() % n, v = rng() % n;
			}while(u != v && !s.count({u, v}));
			s.insert({u, v}), s.insert({v, u});
			res.push_back({u, v});
		}
		return res;
	}
	vector<array<int, 3>> generate_weighted_simply_connected_graph(int n, int m, int wlow = 1, int whigh = 6){
		assert(n - 1 <= m && m <= 1LL * n * (n - 1) / 2);
		auto edges = generate_simply_connected_graph(n, m);
		vector<array<int, 3>> res(m);
		for(int i = 0; i < m; ++ i) res[i] = {{edges[i][0], edges[i][1], rand_int(wlow, whigh)}};
		return res;
	}
}
using namespace graph_generator;

// 156485479_8_5
// Barrett Reduction
// Calculate a mod b in range [0, 2b)
// Credit: KACTL
typedef unsigned long long ull;
struct barrett_reduction{
	ull b, m;
	barrett_reduction(ull b): b(b), m(-1ULL / b) {}
	ull reduce(ull a) { // a % b + (0 or b)
		return a - (ull)((__uint128_t(m) * a) >> 64) * b;
	}
};

// Sherman–Morrison formula
// A + u v^T is invertible if and only if 1 + v^T A u != 0
// (A + u v^T)^-1 = A^-1 - (A^-1 u v^T A^-1) / (1 + v^T A u)
/*
Polynomial
- FFT / NTT / chirp-z transform
- Multipoint evaluation
- Lagrange Interpolation
Number Theory
- Möbius inversion 
- Dirichlet Convolution
- Discrete logarithm / kth root
Linear Algebra
- Gaussian Elimination
- XOR basis
- Simplex algorithm
etc
- Burnside's Lemma
- Kitamasa
*/
