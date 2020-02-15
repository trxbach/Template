/******************************************************************************

                      This is Aeren's algorithm template                       
                         for competitive programming

******************************************************************************/
/******************************************************************************
Category


1. Number Theory
	1.1. Modular Exponentiation, Modular Inverse
		156485479_1_1
	1.2. Extended Euclidean Algorithm
		156485479_1_2
	1.3. Linear Sieve
		156485479_1_3
	1.4. Combinatorics
		156485479_1_4
	1.5. Euler Totient Function
		156485479_1_5
	1.6. Millar Rabin Primality Test
		156485479_1_6
	1.7. Pollard Rho and Factorization
		156485479_1_7
	1.8. Tonelli Shanks Algorithm ( Solution to x^2 = a mod p )
		156485479_1_8
	1.9. Chinese Remainder Theorem
		156485479_1_9
	1.10. Lehman Factorization
		156485479_1_10
	1.11. Mobius Function
		156485479_1_11
	1.12. Polynomial Class
		156485479_1_12


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
		2.3.2. Entries in some ring
			156485479_2_3_2
	2.4. Polynomial
		2.4.1. Convolution
			2.4.1.1 Addition Convolution
				2.4.1.1.1. Fast Fourier Transform
					156485479_2_4_1_1_1
				2.4.1.1.2. Number Theoric Transform
					156485479_2_4_1_1_2
			2.4.1.2. Bitwise XOR Convolution ( Fast Walsh Hadamard Transform )
				156485479_2_4_1_2
			2.4.1.3. Bitwise AND Convolution
				156485479_2_4_1_3
			2.4.1.4. Bitwise OR Convolution
				156485479_2_4_1_4
		2.4.2. Interpolation
			2.4.2.1. Slow Interpolation
				2.4.2.1.1.
					156485479_2_4_2_1_1
				2.4.2.1.2.
					156485479_2_4_2_1_2
			2.4.2.2. Fast Interpolation
				156485479_2_4_2_2 ( INCOMPLETE )
	2.5. Kadane
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
	2.7. Binary Search
		156485479_2_7
	2.8. BigInteger
		156485479_2_8
	2.9. Modular Arithmetics
		156485479_2_9


3. Data Structure
	3.1. Sparse Table
		156485479_3_1
	3.2. Segment Tree
		3.2.1. Simple Iterative Segment Tree
			156485479_3_2_1
		3.2.2. Iterative Segment Tree with Reversed Operation
			156485479_3_2_2
		3.2.3. Recursive Segment Tree
			156485479_3_2_3
		3.2.4. 2D Segment Tree
			156485479_3_2_4 ( INCOMPLETE )
		3.2.5. Lazy Dynamic Segment Tree
			156485479_3_2_5 
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
		156485479_3_4 ( NOT THROUGHLY TESTED YET )
	3.5. Disjoint Set
		156485479_3_5
	3.6. Monotone Stack
		156485479_3_6
	3.7. Less-than-k Query / Distinct Value Query
		156485479_3_7
	3.8. Mo's Algorithm
		156485479_3_8


4. Graph
	4.1. Strongly Connected Component ( Tarjan's Algorithm )
		156485479_4_1
	4.2. Biconnected Component
		156485479_4_2
	4.3. Articulation Points
		156485479_4_3
	4.4. Flow Network
		4.4.1. Dinic's Maximum Flow Algorithm
			156485479_4_4_1
		4.4.2. Minimum Cost Maximum Flow Algorithm
			156485479_4_4_2
	4.5. Tree Algorithms
		4.5.1. LCA
			156485479_4_5_1
		4.5.2. Binary Lifting
			4.5.2.1. Unweighted Tree
				156485479_4_5_2_1
			4.5.2.2. Weighted Tree
				156485479_4_5_2_2
		4.5.3. Heavy Light Decomposition
			156485479_4_5_3
		4.5.4. Centroid Decomposition
			156485479_4_5_4
		4.5.5. AHU Algorithm ( Rooted Tree Isomorphism ) / Tree Isomorphism
			156485479_4_5_5
	4.6. Shortest Path Tree
		4.6.1. On Sparse Graph ( Dijkstra, Bellman Ford, SPFA )
			156485479_4_6_1
		4.6.2. On Dense Graph ( Dijkstra, Floyd Warshall )
			156485479_4_6_2
	4.7. Minimum Spanning Forest
		156485479_4_7
	4.8. Topological Sort
		156485479_4_8
	4.9. Two Satisfiability
		156485479_4_9


5. String
	5.1. Lexicographically Minimal Rotation
		156485479_5_1
	5.2. Palindromic Substrings ( Manacher's Algorithm )
		156485479_5_2
	5.3. Suffix Array and Kasai's Algorithm
		156485479_5_3
	5.4. Z Function
		156485479_5_4
	5.5. Aho Corasic
		156485479_5_5
	5.6. Prefix Function / Prefix Automaton
		156485479_5_6
	5.7. Polynomial Hash
		156485479_5_7
	5.8. Suffix Automaton
		156485479_5_8
	5.9. Suffix Tree ( INCOMPLETE )
		156485479_5_9
	5.10. Palindrome Tree ( INCOMPLETE )
		156485479_5_10


6. Geometry
	6.1. 2D Geometry
		156485479_6_1
	6.2. Convex Hull and Minkowski Addition
		156485479_6_2


7. Miscellaneous
	7.1. Custom Hash Function for unordered_set and unordered map
		156485479_7_1
	7.2. Bump Allocator
		156485479_7_2

*******************************************************************************/

// 156485479_1_1
// Modular Exponentiation, Modular Inverse and Geometric Sum
// O(log e)
long long modexp(long long b, long long e, const long long &mod){
	long long res = 1;
	for(; e; b = b * b % mod, e >>= 1) if(e & 1) res = res * b % mod;
	return res;
}
long long modinv(long long a, long long m){
	long long u = 0, v = 1;
	while(a){
		long long t = m / a;
		m -= t * a; swap(a, m);
		u -= t * v; swap(u, v);
	}
	assert(m == 1);
	return u;
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
T binexp(T b, long long e, const T &id){
	T res = id;
	for(; e; b = b * b, e /= 2) if(e & 1) res = res * b;
	return res;
}
template<typename T>
T modinv(T a, T m){
	T u = 0, v = 1;
	while(a){
		T t = m / a;
		m -= t * a; swap(a, m);
		u -= t * v; swap(u, v);
	}
	assert(m == 1);
	return u;
}
template<typename T>
T bingeo(const T &b, long long e, const T &add_id, const T &mul_id){
	if(e < 2) return e ? mul_id : add_id;
	T res = mul_id, p = mul_id;
	for(long long bit = 1 << 30 - __builtin_clz(e); bit; bit >>= 1){
		res = res * (mul_id + p * b), p = p * p * b;
		if(bit & e) res = (res + (p = p * b));
	}
	return res;
}

// 156485479_1_2
// Extended Euclidean Algorithm
// O(max(log x, log y))
long long euclid(long long x, long long y, long long &a, long long &b){
	if(y){
		long long d = euclid(y, x % y, b, a);
		return b -= x / y * a, d;
	}
	return a = 1, b = 0, x;
}

// 156485479_1_3
// Run linear sieve up to n
// O(n)
pair<vector<int>, vector<int>> linearsieve(int n){
	vector<int> lpf(n + 1), prime;
	prime.reserve(n + 1);
	for(int i = 2; i <= n; ++ i){
		if(!lpf[i]) lpf[i] = i, prime.push_back(i);
		for(int j = 0; j < int(prime.size()) && prime[j] <= lpf[i] && i * prime[j] <= n; ++ j){
			lpf[i * prime[j]] = prime[j];
		}
	}
	return {lpf, prime};
}

// 156485479_1_4
// Combinatorics
// O(N) preprocessing, O(1) per query
struct combinatorics{
	const long long N, mod;
	vector<long long> inv, fact, invfact;
	combinatorics(long long N, long long mod):
		N(N), mod(mod), inv(N + 1), fact(N + 1), invfact(N + 1){
		inv[1] = 1, fact[0] = fact[1] = invfact[0] = invfact[1] = 1;
		for(long long i = 2; i <= N; i ++){
			inv[i] = (mod - mod / i * inv[mod % i] % mod) % mod;
			fact[i] = fact[i - 1] * i % mod;
			invfact[i] = invfact[i - 1] * inv[i] % mod;
		}
	}
	long long C(int n, int r){
		return n < r ? 0 : fact[n] * invfact[r] % mod * invfact[n - r] % mod;
	}
	long long P(int n, int r){
		return n < r ? 0 : fact[n] * invfact[n - r] % mod;
	}
	long long H(int n, int r){
		return c(n + r - 1, r);
	}
	long long Cat(int n, int k, int m){
		if(m <= 0) return 0;
		else if(k >= 0 && k < m) return c(n + k, k);
		else if(k < n + m) return (c(n + k, k) - c(n + k, k - m) + mod) % mod;
		else return 0;
	}
};

// 156485479_1_5
// Euler Totient Function
// O(sqrt(x))
long long phi(long long x){
	long long res = x;
	for(long long i = 2; i * i <= x; ++ i) if(x % i == 0){
		while(x % i == 0) x /= i;
		res -= res / i;
	}
	if(x > 1) res -= res / x;
	return res;
}
// Calculate phi(x) for all 1 <= x <= n
// O(n log n)
vector<int> process_phi(int n){
	vector<int> phi(n + 1);
	for(int i = 0; i <= n; ++ i) phi[i] = i & 1 ? i : i / 2;
	for(int i = 3; i <= n; i += 2) if(phi[i] == i) for(int j = i; j <= n; j += i) phi[j] -= phi[j] / i;
	return phi;
}

// 156485479_1_6
// Millar Rabin Primality Test
// O(log n) {constant is around 7}
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
	for(auto a: A){
		ull p = mod_pow(a, d, n), i = s;
		while(p != 1 && p != n - 1 && a % n && i --) p = mod_pow(p, p, n);
		if(p != n - 1 && i != s) return 0;
	}
	return 1;
}

// 156485479_1_7
// Pollard Rho Algorithm
// O(n^{1/4} log n)
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

// 156485479_1_8
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
	assert(modexp(a, (p - 1)/2, p) == 1);
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

// 156485479_1_9
// Chinese Remainder Theorem (Return a number x which satisfies x = a mod m & x = b mod n)
// All the values has to be less than 2^30
// O(log(m + n))
long long euclid(long long x, long long y, long long &a, long long &b){
	if(y){
		long long d = euclid(y, x % y, b, a);
		return b -= x / y * a, d;
	}
	return a = 1, b = 0, x;
}
long long crt_coprime(long long a, long long m, long long b, long long n){
	long long x, y; euclid(m, n, x, y);
	long long res = a * (y + m) % m * n + b * (x + n) % n * m;
	if(res >= m * n) res -= m * n;
	return res;
}
long long crt(long long a, long long m, long long b, long long n){
	long long d = gcd(m, n);
	if(((b -= a) %= n) < 0) b += n;
	if(b % d) return -1; // No solution
	return d * crt_coprime(0LL, m/d, b/d, n/d) + a;
}

// 156485479_1_10
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

// 156485479_1_11
// Mobius Function
// O(n)
pair<vector<int>, vector<int>> linearsieve(int n){
	vector<int> lpf(n + 1), prime;
	prime.reserve(n + 1);
	for(int i = 2; i <= n; ++ i){
		if(!lpf[i]) lpf[i] = i, prime.push_back(i);
		for(int j = 0; j < int(prime.size()) && prime[j] <= lpf[i] && i * prime[j] <= n; ++ j){
			lpf[i * prime[j]] = prime[j];
		}
	}
	return {lpf, prime};
}
tuple<vector<int>, vector<int>, vector<int>> process_mobius(int n){
	auto [lpf, prime] = linearsieve(n);
	vector<int> mobius(n + 1, 1);
	for(int i = 2; i <= n; ++ i) mobius[i] = (i / lpf[i] % lpf[i] ? -mobius[i / lpf[i]] : 0);
	return {lpf, prime, mobius}
}

// 156485479_1_12
// Polynomial Class
namespace algebra {
	int mod;
	const int inf = 1e9;
	const int magic = 500; // threshold for sizes to run the naive algo
	namespace fft {
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
			for(size_t i = 0; i < a.size(); ++ i){
				for(size_t j = 0; j < int(b.size()); ++ j){
					res[i + j] += a[i] * b[j];
				}
			}
			a = res;
		}
		template<typename T>
		void mul(vector<T> &a, const vector<T> &b){
			if(int(min(a.size(), b.size())) < magic){
				mul_slow(a, b);
				return;
			}
			init();
			static const int shift = 15, mask = (1 << shift) - 1;
			size_t n = a.size() + b.size() - 1;
			while(__builtin_popcount(n) != 1) ++ n;
			a.resize(n);
			static point A[maxn], B[maxn];
			static point C[maxn], D[maxn];
			for(size_t i = 0; i < n; ++ i){
				A[i] = point(a[i] & mask, a[i] >> shift);
				if(i < b.size()) {
					B[i] = point(b[i] & mask, b[i] >> shift);
				} else {
					B[i] = 0;
				}
			}
			fft(A, C, n); fft(B, D, n);
			for(size_t i = 0; i < n; i++) {
				point c0 = C[i] + conj(C[(n - i) % n]);
				point c1 = C[i] - conj(C[(n - i) % n]);
				point d0 = D[i] + conj(D[(n - i) % n]);
				point d1 = D[i] - conj(D[(n - i) % n]);
				A[i] = c0 * d0 - point(0, 1) * c1 * d1;
				B[i] = c0 * d1 + d0 * c1;
			}
			fft(A, C, n); fft(B, D, n);
			reverse(C + 1, C + n);
			reverse(D + 1, D + n);
			int t = 4 * n;
			for(size_t i = 0; i < n; i++) {
				long long A0 = llround(real(C[i]) / t);
				T A1 = llround(imag(D[i]) / t);
				T A2 = llround(imag(C[i]) / t);
				a[i] = A0 + (A1 << shift) + (A2 << 2 * shift);
			}
			return;
		}
	}
	template<typename T>
	T bpow(T x, size_t n) {
		return n ? n % 2 ? x * bpow(x, n - 1) : bpow(x * x, n / 2) : T(1);
	}
	template<typename T>
	T bpow(T x, size_t n, T m) {
		return n ? n % 2 ? x * bpow(x, n - 1, m) % m : bpow(x * x % m, n / 2, m) : T(1);
	}
	template<typename T>
	T gcd(const T &a, const T &b) {
		return b == T(0) ? a : gcd(b, a % b);
	}
	template<typename T>
	T nCr(T n, int r) { // runs in O(r)
		T res(1);
		for(int i = 0; i < r; i++) {
			res *= (n - T(i));
			res /= (i + 1);
		}
		return res;
	}

	struct modular {
		long long r;
		modular() : r(0) {}
		modular(long long rr) : r(rr) {if(abs(r) >= mod) r %= mod; if(r < 0) r += mod;}
		modular inv() const {return bpow(*this, mod - 2);}
		modular operator * (const modular &t) const {return (r * t.r) % mod;}
		modular operator / (const modular &t) const {return *this * t.inv();}
		modular operator += (const modular &t) {r += t.r; if(r >= mod) r -= mod; return *this;}
		modular operator -= (const modular &t) {r -= t.r; if(r < 0) r += mod; return *this;}
		modular operator + (const modular &t) const {return modular(*this) += t;}
		modular operator - (const modular &t) const {return modular(*this) -= t;}
		modular operator *= (const modular &t) {return *this = *this * t;}
		modular operator /= (const modular &t) {return *this = *this / t;}
		
		bool operator == (const modular &t) const {return r == t.r;}
		bool operator != (const modular &t) const {return r != t.r;}
		
		operator long long() const {return r;}
	};

	istream& operator >> (istream &in, modular &x) {
		return in >> x.r;
	}
	
	
	template<typename T>
	struct poly {
		vector<T> a;
		
		void normalize() { // get rid of leading zeroes
			while(!a.empty() && a.back() == T(0)) {
				a.pop_back();
			}
		}
		
		poly(){}
		poly(T a0) : a{a0}{normalize();}
		poly(vector<T> t) : a(t){normalize();}
		
		poly operator += (const poly &t) {
			a.resize(max(a.size(), t.a.size()));
			for(size_t i = 0; i < t.a.size(); i++) {
				a[i] += t.a[i];
			}
			normalize();
			return *this;
		}
		poly operator -= (const poly &t) {
			a.resize(max(a.size(), t.a.size()));
			for(size_t i = 0; i < t.a.size(); i++) {
				a[i] -= t.a[i];
			}
			normalize();
			return *this;
		}
		poly operator + (const poly &t) const {return poly(*this) += t;}
		poly operator - (const poly &t) const {return poly(*this) -= t;}
		
		poly mod_xk(size_t k) const { // get same polynomial mod x^k
			k = min(k, a.size());
			return vector<T>(begin(a), begin(a) + k);
		}
		poly mul_xk(size_t k) const { // multiply by x^k
			poly res(*this);
			res.a.insert(begin(res.a), k, 0);
			return res;
		}
		poly div_xk(size_t k) const { // divide by x^k, dropping coefficients
			k = min(k, a.size());
			return vector<T>(begin(a) + k, end(a));
		}
		poly substr(size_t l, size_t r) const { // return mod_xk(r).div_xk(l)
			l = min(l, a.size());
			r = min(r, a.size());
			return vector<T>(begin(a) + l, begin(a) + r);
		}
		poly inv(size_t n) const { // get inverse series mod x^n
			assert(!is_zero());
			poly ans = a[0].inv();
			size_t a = 1;
			while(a < n) {
				poly C = (ans * mod_xk(2 * a)).substr(a, 2 * a);
				ans -= (ans * C).mod_xk(a).mul_xk(a);
				a *= 2;
			}
			return ans.mod_xk(n);
		}
		
		poly operator *= (const poly &t) {fft::mul(a, t.a); normalize(); return *this;}
		poly operator * (const poly &t) const {return poly(*this) *= t;}
		
		poly reverse(size_t n, bool rev = 0) const { // reverses and leaves only n terms
			poly res(*this);
			if(rev) { // If rev = 1 then tail goes to head
				res.a.resize(max(n, res.a.size()));
			}
			std::reverse(res.a.begin(), res.a.end());
			return res.mod_xk(n);
		}
		
		pair<poly, poly> divmod_slow(const poly &b) const { // when divisor or quotient is small
			vector<T> A(a);
			vector<T> res;
			while(A.size() >= b.a.size()) {
				res.push_back(A.back() / b.a.back());
				if(res.back() != T(0)) {
					for(size_t i = 0; i < b.a.size(); i++) {
						A[A.size() - i - 1] -= res.back() * b.a[b.a.size() - i - 1];
					}
				}
				A.pop_back();
			}
			std::reverse(begin(res), end(res));
			return {res, A};
		}
		
		pair<poly, poly> divmod(const poly &b) const { // returns quotiend and remainder of a mod b
			if(deg() < b.deg()) {
				return {poly{0}, *this};
			}
			int d = deg() - b.deg();
			if(min(d, b.deg()) < magic) {
				return divmod_slow(b);
			}
			poly D = (reverse(d + 1) * b.reverse(d + 1).inv(d + 1)).mod_xk(d + 1).reverse(d + 1, 1);
			return {D, *this - D * b};
		}
		
		poly operator / (const poly &t) const {return divmod(t).first;}
		poly operator % (const poly &t) const {return divmod(t).second;}
		poly operator /= (const poly &t) {return *this = divmod(t).first;}
		poly operator %= (const poly &t) {return *this = divmod(t).second;}
		poly operator *= (const T &x) {
			for(auto &it: a) {
				it *= x;
			}
			normalize();
			return *this;
		}
		poly operator /= (const T &x) {
			for(auto &it: a) {
				it /= x;
			}
			normalize();
			return *this;
		}
		poly operator * (const T &x) const {return poly(*this) *= x;}
		poly operator / (const T &x) const {return poly(*this) /= x;}
		
		void print() const {
			for(auto it: a) {
				cout << it << ' ';
			}
			cout << endl;
		}
		T eval(T x) const { // evaluates in single point x
			T res(0);
			for(int i = int(a.size()) - 1; i >= 0; i--) {
				res *= x;
				res += a[i];
			}
			return res;
		}
		
		T& lead() { // leading coefficient
			return a.back();
		}
		int deg() const { // degree
			return a.empty() ? -inf : a.size() - 1;
		}
		bool is_zero() const { // is polynomial zero
			return a.empty();
		}
		T operator [](int idx) const {
			return idx >= (int)a.size() || idx < 0 ? T(0) : a[idx];
		}
		
		T& coef(size_t idx) { // mutable reference at coefficient
			return a[idx];
		}
		bool operator == (const poly &t) const {return a == t.a;}
		bool operator != (const poly &t) const {return a != t.a;}
		
		poly deriv() { // calculate derivative
			vector<T> res;
			for(int i = 1; i <= deg(); i++) {
				res.push_back(T(i) * a[i]);
			}
			return res;
		}
		poly integr() { // calculate integral with C = 0
			vector<T> res = {0};
			for(int i = 0; i <= deg(); i++) {
				res.push_back(a[i] / T(i + 1));
			}
			return res;
		}
		size_t leading_xk() const { // Let p(x) = x^k * t(x), return k
			if(is_zero()) {
				return inf;
			}
			int res = 0;
			while(a[res] == T(0)) {
				res++;
			}
			return res;
		}
		poly log(size_t n) { // calculate log p(x) mod x^n
			assert(a[0] == T(1));
			return (deriv().mod_xk(n) * inv(n)).integr().mod_xk(n);
		}
		poly exp(size_t n) { // calculate exp p(x) mod x^n
			if(is_zero()) {
				return T(1);
			}
			assert(a[0] == T(0));
			poly ans = T(1);
			size_t a = 1;
			while(a < n) {
				poly C = ans.log(2 * a).div_xk(a) - substr(a, 2 * a);
				ans -= (ans * C).mod_xk(a).mul_xk(a);
				a *= 2;
			}
			return ans.mod_xk(n);
			
		}
		poly pow_slow(size_t k, size_t n) { // if k is small
			return k ? k % 2 ? (*this * pow_slow(k - 1, n)).mod_xk(n) : (*this * *this).mod_xk(n).pow_slow(k / 2, n) : T(1);
		}
		poly pow(size_t k, size_t n) { // calculate p^k(n) mod x^n
			if(is_zero()) {
				return *this;
			}
			if(k < magic) {
				return pow_slow(k, n);
			}
			int i = leading_xk();
			T j = a[i];
			poly t = div_xk(i) / j;
			return bpow(j, k) * (t.log(n) * T(k)).exp(n).mul_xk(i * k).mod_xk(n);
		}
		poly mulx(T x) { // component-wise multiplication with x^k
			T cur = 1;
			poly res(*this);
			for(int i = 0; i <= deg(); i++) {
				res.coef(i) *= cur;
				cur *= x;
			}
			return res;
		}
		poly mulx_sq(T x) { // component-wise multiplication with x^{k^2}
			T cur = x;
			T total = 1;
			T xx = x * x;
			poly res(*this);
			for(int i = 0; i <= deg(); i++) {
				res.coef(i) *= total;
				total *= cur;
				cur *= xx;
			}
			return res;
		}
		vector<T> chirpz_even(T z, int n) { // P(1), P(z^2), P(z^4), ..., P(z^2(n-1))
			int m = deg();
			if(is_zero()) {
				return vector<T>(n, 0);
			}
			vector<T> vv(m + n);
			T zi = z.inv();
			T zz = zi * zi;
			T cur = zi;
			T total = 1;
			for(int i = 0; i <= max(n - 1, m); i++) {
				if(i <= m) {vv[m - i] = total;}
				if(i < n) {vv[m + i] = total;}
				total *= cur;
				cur *= zz;
			}
			poly w = (mulx_sq(z) * vv).substr(m, m + n).mulx_sq(z);
			vector<T> res(n);
			for(int i = 0; i < n; i++) {
				res[i] = w[i];
			}
			return res;
		}
		vector<T> chirpz(T z, int n) { // P(1), P(z), P(z^2), ..., P(z^(n-1))
			auto even = chirpz_even(z, (n + 1) / 2);
			auto odd = mulx(z).chirpz_even(z, n / 2);
			vector<T> ans(n);
			for(int i = 0; i < n / 2; i++) {
				ans[2 * i] = even[i];
				ans[2 * i + 1] = odd[i];
			}
			if(n % 2 == 1) {
				ans[n - 1] = even.back();
			}
			return ans;
		}
		template<typename iter>
		vector<T> eval(vector<poly> &tree, int v, iter l, iter r) { // auxiliary evaluation function
			if(r - l == 1) {
				return {eval(*l)};
			} else {
				auto m = l + (r - l) / 2;
				auto A = (*this % tree[2 * v]).eval(tree, 2 * v, l, m);
				auto B = (*this % tree[2 * v + 1]).eval(tree, 2 * v + 1, m, r);
				A.insert(end(A), begin(B), end(B));
				return A;
			}
		}
		vector<T> eval(vector<T> x) { // evaluate polynomial in (x1, ..., xn)
			int n = x.size();
			if(is_zero()) {
				return vector<T>(n, T(0));
			}
			vector<poly> tree(4 * n);
			build(tree, 1, begin(x), end(x));
			return eval(tree, 1, begin(x), end(x));
		}
		template<typename iter>
		poly inter(vector<poly> &tree, int v, iter l, iter r, iter ly, iter ry) { // auxiliary interpolation function
			if(r - l == 1) {
				return {*ly / a[0]};
			} else {
				auto m = l + (r - l) / 2;
				auto my = ly + (ry - ly) / 2;
				auto A = (*this % tree[2 * v]).inter(tree, 2 * v, l, m, ly, my);
				auto B = (*this % tree[2 * v + 1]).inter(tree, 2 * v + 1, m, r, my, ry);
				return A * tree[2 * v + 1] + B * tree[2 * v];
			}
		}
	};
	template<typename T>
	poly<T> operator * (const T& a, const poly<T>& b) {
		return b * a;
	}
	
	template<typename T>
	poly<T> xk(int k) { // return x^k
		return poly<T>{1}.mul_xk(k);
	}

	template<typename T>
	T resultant(poly<T> a, poly<T> b) { // computes resultant of a and b
		if(b.is_zero()) {
			return 0;
		} else if(b.deg() == 0) {
			return bpow(b.lead(), a.deg());
		} else {
			int pw = a.deg();
			a %= b;
			pw -= a.deg();
			T mul = bpow(b.lead(), pw) * T((b.deg() & a.deg() & 1) ? -1 : 1);
			T ans = resultant(b, a);
			return ans * mul;
		}
	}
	template<typename iter>
	poly<typename iter::value_type> kmul(iter L, iter R) { // computes (x-a1)(x-a2)...(x-an) without building tree
		if(R - L == 1) {
			return vector<typename iter::value_type>{-*L, 1};
		} else {
			iter M = L + (R - L) / 2;
			return kmul(L, M) * kmul(M, R);
		}
	}
	template<typename T, typename iter>
	poly<T> build(vector<poly<T>> &res, int v, iter L, iter R) { // builds evaluation tree for (x-a1)(x-a2)...(x-an)
		if(R - L == 1) {
			return res[v] = vector<T>{-*L, 1};
		} else {
			iter M = L + (R - L) / 2;
			return res[v] = build(res, 2 * v, L, M) * build(res, 2 * v + 1, M, R);
		}
	}
	template<typename T>
	poly<T> inter(vector<T> x, vector<T> y) { // interpolates minimum polynomial from (xi, yi) pairs
		int n = x.size();
		vector<poly<T>> tree(4 * n);
		return build(tree, 1, begin(x), end(x)).deriv().inter(tree, 1, begin(x), end(x), begin(y), end(y));
	}
};

using namespace algebra;
typedef poly<modular> polym;
mod = 1e9 + 7;

// 156485479_2_1
// Linear Recurrence Relation Solver / Berlekamp - Massey Algorithm
// O(N^2 log n) / O(N^2)
long long modinv(long long a, long long m){
	long long u = 0, v = 1;
	while(a){
		long long t = m / a;
		m -= t * a; swap(a, m);
		u -= t * v; swap(u, v);
	}
	assert(m == 1);
	return u;
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

// 156485479_2_2_2
// Find a solution of the system of linear equations. Return -1 if no sol, rank otherwise.
// O(n^2 m)
long long modinv(long long a, long long m){
	long long u = 0, v = 1;
	while(a){
		long long t = m / a;
		m -= t * a; swap(a, m);
		u -= t * v; swap(u, v);
	}
	assert(m == 1);
	return u;
}
int solve_linear_equations(const vector<vector<long long>> &AA, vector<long long> &x, const vector<long long> &bb, long long mod){
	auto A = AA;
	auto b = bb;
	int n = int(A.size()), m = int(A[0].size()), rank = 0, br, bc;
	vector<int> col(m);
	iota(col.begin(), col.end(), 0);
	for(auto &x: A) for(auto &y: x) y %= mod;
	for(int i = 0; i < n; ++ i){
		long long v, bv = 0;
		for(int r = i; r < n; ++ r) for(int c = i; c < m; ++ c) if((v = abs(A[r][c])) > bv) br = r, bc = c, bv = v;
		if(!bv){
			for(int j = i; j < n; ++ j) if(abs(b[j])) return -1;
			break;
		}
		swap(A[i], A[br]), swap(b[i], b[br]), swap(col[i], col[bc]);
		for(int j = 0; j < n; ++ j) swap(A[j][i], A[j][bc]);
		bv = modinv(A[i][i], mod);
		for(int j = i + 1; j < n; ++ j){
			long long fac = A[j][i] * bv % mod;
			b[j] = (b[j] - fac * b[i] % mod + mod) % mod;
			for(int k = i + 1; k < m; ++ k) A[j][k] = (A[j][k] - fac * A[i][k] % mod + mod) % mod;
		}
		++ rank;
	}
	x.resize(m);
	for(int i = rank; i --; ){
		b[i] = b[i] * modinv(A[i][i], mod) % mod;
		x[col[i]] = b[i];
		for(int j = 0; j < i; ++ j) b[j] = (b[j] - A[j][i] * b[i] % mod + mod) % mod;
	}
	return rank;
}

// 156485479_2_2_3
// Find a solution of the system of linear equations. Return -1 if no sol, rank otherwise.
// O(n^2 m)
typedef bitset<1000> bs;
int solve_linear_equations(const vector<bs> &AA, bs& x, const vector<int> &bb, int m){
	vector<bs> A(AA);
	vector<int> b(bb);
	int n = int(A.size()), rank = 0, br;
	vector<int> col(m);
	iota(col.begin(), col.end(), 0);
	for(int i = 0; i < n; ++ i){
		for(br = i; br < n; ++ br) if(A[br].any()) break;
		if (br == n){
			for(int j = i; j < n; ++ j) if(b[j]) return -1;
			break;
		}
		int bc = (int)A[br]._Find_next(i-1);
		swap(A[i], A[br]);
		swap(b[i], b[br]);
		swap(col[i], col[bc]);
		for(int j = 0; j < n; ++ j) if(A[j][i] != A[j][bc]) A[j].flip(i); A[j].flip(bc);
		for(int j = i + 1; j < n; ++ j) if(A[j][i]) b[j] ^= b[i], A[j] ^= A[i];
		++ rank;
	}
	x = bs();
	for (int i = rank; i --;){
		if (!b[i]) continue;
		x[col[i]] = 1;
		for(int j = 0; j < i; ++ j) b[j] ^= A[j][i];
	}
	return rank;
}

// 156485479_2_3_1
// Matrix for Z_p
struct matrix: vector<vector<long long>>{
	int N, M;
	const long long mod;
	matrix(int N, int M, long long mod, bool is_id = false): N(N), M(M), mod(mod){
		resize(N, vector<long long>(M));
		if(is_id) for(int i = 0; i < min(N, M); ++ i) (*this)[i][i] = 1;
	}
	matrix(const vector<vector<long long>> &arr, long long mod): N(arr.size()), M(arr[0].size()), mod(mod){
		resize(N);
		for(int i = 0; i < N; ++ i) (*this)[i] = arr[i];
	}
	bool operator==(const matrix &otr) const{
		if(N != otr.N || M != otr.M) return false;
		for(int i = 0; i < N; ++ i) for(int j = 0; j < M; ++ j) if((*this)[i][j] != otr[i][j]) return false;
		return true;
	}
	matrix operator=(const matrix &otr){
		N = otr.N, M = otr.M;
		resize(N);
		for(int i = 0; i < N; ++ i) (*this)[i] = otr[i];
		return *this;
	}
	matrix operator+(const matrix &otr) const{
		matrix res(N, M, mod);
		for(int i = 0; i < N; ++ i) for(int j = 0; j < M; ++ j) res[i][j] = ((*this)[i][j] + otr[i][j]) % mod;
		return res;
	}
	matrix operator+=(const matrix &otr){
		*this = *this + otr;
		return *this;
	}
	matrix operator*(const matrix &otr) const{
		assert(M == otr.N);
		int L = otr.M;
		matrix res(N, M, mod);
		for(int i = 0; i < N; ++ i) for(int j = 0; j < L; ++ j) for(int k = 0; k < M; ++ k) (res[i][j] += (*this)[i][k] * otr[k][j]) %= mod;
		return res;
	}
	matrix operator*=(const matrix &otr){
		*this = *this * otr;
		return *this;
	}
	matrix operator^(long long e) const{
		assert(N == M);
		matrix res(N, N, mod, 1), b(*this);
		for(; e; b *= b, e >>= 1) if(e & 1) res *= b;
		return res;
	}
	long long det() const{
		assert(N == M);
		vector<vector<long long>> temp = *this;
		long long res = 1;
		for(int i = 0; i < N; ++ i){
			for(int j = i + 1; j < N; ++ j){
				while(temp[j][i]){
					long long t = temp[i][i] / temp[j][i];
					if(t) for(int k = i; i < N; ++ k) temp[i][k] = (temp[i][k] - temp[j][k] * t) % mod;
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
// Matrix for general ring
// T must support +, *, !=, <<, >>
template<typename T>
struct matrix: vector<vector<T>>{
	int N, M;
	const T add_id, mul_id; // multiplicative identity
	matrix(int N, int M, const T &add_id, const T &mul_id, bool is_id = false): N(N), M(M), add_id(add_id), mul_id(mul_id){
		this->resize(N, vector<T>(M, add_id));
		if(is_id) for(int i = 0; i < min(N, M); ++ i) (*this)[i][i] = mul_id;
	}
	matrix(const vector<vector<T>> &arr, const T &add_id, const T &mul_id): N(arr.size()), M(arr[0].size()), add_id(add_id), mul_id(mul_id){
		this->resize(N, vector<T>(M, add_id));
		for(int i = 0; i < N; ++ i) for(int j = 0; j < M; ++ j) (*this)[i][j] = arr[i][j];
	}
	bool operator==(const matrix &otr) const{
		if(N != otr.N || M != otr.M) return false;
		for(int i = 0; i < N; ++ i) for(int j = 0; j < M; ++ j) if((*this)[i][j] != otr[i][j]) return false;
		return true;
	}
	matrix operator=(const matrix &otr){
		N = otr.N, M = otr.M;
		this->resize(N);
		for(int i = 0; i < N; ++ i) (*this)[i] = otr[i];
		return *this;
	}
	matrix operator+(const matrix &otr) const{
		matrix res(N, M, add_id, mul_id);
		for(int i = 0; i < N; ++ i) for(int j = 0; j < M; ++ j) res[i][j] = (*this)[i][j] + otr[i][j];
		return res;
	}
	matrix operator+=(const matrix &otr){
		*this = *this + otr;
		return *this;
	}
	matrix operator*(const matrix &otr) const{
		assert(M == otr.N);
		int L = otr.M;
		matrix res(N, M, add_id, mul_id);
		for(int i = 0; i < N; ++ i) for(int j = 0; j < L; ++ j) for(int k = 0; k < M; ++ k) res[i][j] = res[i][j] + (*this)[i][k] * otr[k][j];
		return res;
	}
	matrix operator*=(const matrix &otr){
		*this = *this * otr;
		return *this;
	}
	matrix operator^(long long e) const{
		assert(N == M);
		matrix res(N, N, add_id, mul_id, true), b(*this);
		for(; e; b *= b, e >>= 1) if(e & 1) res *= b;
		return res;
	}
};

// 156485479_2_4_1_1_1
// Fast Fourier Transformation.
// Size must be a power of two.
// O(n log n)
typedef complex<double> cd;
const double PI = acos(-1);
void fft(vector<cd> &f, bool invert){
	int n = int(f.size());
	for(int i = 1, j = 0; i < n; ++ i){
		int bit = n >> 1;
		for(; j & bit; bit >>= 1) j ^= bit;
		j ^= bit;
		if(i < j) swap(f[i], f[j]);
	}
	for(int len = 2; len <= n; len <<= 1){
		double theta = 2 * PI / len * (invert ? -1 : 1);
		cd w(cos(theta), sin(theta));
		for(int i = 0; i < n; i += len){
			cd wj(1);
			for(int j = 0; j < len / 2; ++ j, wj *= w){
				cd u = f[i + j], v = wj * f[i + j + len / 2];
				f[i + j] = u + v, f[i + j + len / 2] = u - v;
			}
		}
	}
	if(invert) for(auto &c: f) c /= n;
}
vector<long long> polymul(const vector<long long> &a, const vector<long long> &b){
	vector<cd> f(a.begin(), a.end()), g(b.begin(), b.end());
	int n = 1;
	while(n < int(a.size() + b.size())) n <<= 1;
	f.resize(n), g.resize(n);
	fft(f, false), fft(g, false);
	for(int i = 0; i < n; ++ i) f[i] *= g[i];
	fft(f, true);
	vector<long long> res(n);
	for(int i = 0; i < n; ++ i) res[i] = round(f[i].real());
	while(!res.empty() && !res.back()) res.pop_back();
	return res;
}

// 156485479_2_4_1_1_2
// Number Theoric Transformation. Use (998244353, 15311432, 1 << 23) or (7340033, 5, 1 << 20)
// Size must be a power of two
// O(n log n)
long long modinv(long long a, long long m){
	long long u = 0, v = 1;
	while(a){
		long long t = m / a;
		m -= t * a; swap(a, m);
		u -= t * v; swap(u, v);
	}
	assert(m == 1);
	return u;
}
const long long mod = 998244353, root = 15311432, root_pw = 1 << 23, root_1 = modinv(root, mod);
vector<long long> ntt(const vector<long long> &arr, bool invert){
    int n = int(arr.size());
    vector<long long> a{arr};
    for(int i = 1, j = 0; i < n; ++ i){
        int bit = n >> 1;
        for(; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if(i < j) swap(a[i], a[j]);
    }
    for(int len = 2; len <= n; len <<= 1){
        long long wlen = invert ? root_1 : root;
        for(int i = len; i < root_pw; i <<= 1) wlen = wlen * wlen % mod;
        for(int i = 0; i < n; i += len){
            long long w = 1;
            for(int j = 0; j < len / 2; ++ j){
                long long u = a[i + j], v = a[i + j + len / 2] * w % mod;
                a[i + j] = u + v < mod ? u + v : u + v - mod;
                a[i + j + len / 2] = u - v >= 0 ? u - v : u - v + mod;
                w = w * wlen % mod;
            }
        }
    }
    if(invert){
        long long n_1 = modinv(n, mod);
        for(auto &x: a) x = x * n_1 % mod;
    }
    return a;
}

// 156485479_2_4_1_2
// Bitwise XOR Transformation ( Fast Walsh Hadamard Transformation, FWHT ).
// Size must be a power of two.
// Transformation   1  1     Inversion   1  1     
//     Matrix       1 -1      Matrix     1 -1   TIMES  1/2
// O(n log n)
template<typename T>
vector<T> xort(const vector<T> &P, bool inverse){
	vector<T> p(P);
	int n = int(p.size());
	for(int len = 1; 2 * len <= n; len <<= 1){
		for(int i = 0; i < n; i += 2 * len){
			for(int j = 0; j < len; ++ j){
				T u = p[i + j], v = p[i + j + len];
				p[i + j] = u + v, p[i + j + len] = u - v;
			}
		}
	}
	if(inverse) for(int i = 0; i < n; ++ i) p[i] /= n;
	return p;
}

// 156485479_2_4_1_3
// Bitwise AND Transformation.
// Size must be a power of two.
// Transformation   0  1     Inversion   -1  1
//     Matrix       1  1      Matrix      1  0
// O(n log n)
template<typename T>
vector<T> andt(const vector<T> &P, bool inverse){
	vector<T> p(P);
	int n = int(p.size());
	for(int len = 1; 2 * len <= n; len <<= 1){
		for(int i = 0; i < n; i += 2 * len){
			for(int j = 0; j < len; ++ j){
				T u = p[i + j], v = p[i + j + len];
				if(!inverse) p[i + j] = v, p[i + j + len] = u + v;
				else p[i + j] = -u + v, p[i + j + len] = u;
			}
		}
	}
	return p;
}

// 156485479_2_4_1_4
// Bitwise OR Transformation.
// Size must be a power of two
// Transformation   1  1     Inversion    0  1
//     Matrix       1  0      Matrix      1 -1
// O(n log n)
template<typename T>
vector<T> ort(const vector<T> &P, bool inverse){
	vector<T> p(P);
	int n = int(p.size());
	for(int len = 1; 2 * len <= n; len <<= 1){
		for(int i = 0; i < n; i += 2 * len){
			for(int j = 0; j < len; ++ j){
				T u = p[i + j], v = p[i + j + len];
				if(!inverse) p[i + j] = u + v, p[i + j + len] = u;
				else p[i + j] = v, p[i + j + len] = u - v;
			}
		}
	}
	return p;
}

// 156485479_2_4_2_1_1
// Polynomial Interpolation
// O(n ^ 2)
vector<double> interpolate(vector<double> x, vector<double> y){
	int n = int(x.size());
	vector<double> res(n), temp(n);
	for(int k = 0; k < n; ++ k) for(int i = k + 1; i < n; ++ i){
		y[i] = (y[i] - y[k]) / (x[i] - x[k]);
	}
	double last = 0; temp[0] = 1;
	for(int k = 0; k < n; ++ k) for(int i = 0; i < n; ++ i){
		res[i] += y[k] * temp[i];
		swap(last, temp[i]);
		temp[i] -= last * x[k];
	}
	return res;
}

// 156485479_2_4_2_1_2
// Polynomial Interpolation
// O(n ^ 2)
long long modinv(long long a, long long m){
	long long u = 0, v = 1;
	while(a){
		long long t = m / a;
		m -= t * a; swap(a, m);
		u -= t * v; swap(u, v);
	}
	assert(m == 1);
	return u;
}
vector<long long> interpolate(vector<long long> x, vector<long long> y, long long mod){
	int n = int(x.size());
	vector<long long> res(n), temp(n);
	for(int k = 0; k < n; ++ k){
		for(int i = k + 1; i < n; ++ i){
			y[i] = (y[i] - y[k]) * modinv(x[i] - x[k], mod) % mod;
		}
	}
	long long last = 0; temp[0] = 1;
	for(int k = 0; k < n; ++ k){
		for(int i = 0; i < n; ++ i){
			res[i] = (res[i] + y[k] * temp[i]) % mod;
			swap(last, temp[i]);
			temp[i] = (temp[i] - last * x[k] % mod + mod) % mod;
		}
	}
	return res;
}

// 156485479_2_4_2_2
// Polynomial Interpolation
// O(n log n)
// (INCOMPLETE!)

// 156485479_2_5
// Kadane
// O(N)
template<typename T>
T kadane(const vector<T> &arr){
	int n = int(arr.size());
	T lm = 0, gm = 0;
	for(int i = 0; i < n; ++ i){
		lm = max(arr[i], arr[i] + lm);
		gm = max(gm, lm);
	}
	return gm;
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
	// (for doubles, use inf = 1/.0, div(a,b) = a/b)
	static constexpr long long inf = numeric_limits<long long>::max();
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
struct line{
	long long d, k;
	line(long long d = 0, long long k = -(long long)9e18): d(d), k(k){ }
	long long eval(long long x){ return d * x + k; }
	bool majorize(line X, long long L, long long R){ return eval(L) >= X.eval(L) && eval(R) >= X.eval(R); }
};
template<bool GET_MAX = true>
struct lichao{
	lichao *l = NULL, *r = NULL;
	line S;
	lichao(): S(line()){ }
	void rm(){
		if(l) l->rm();
		if(r) r->rm();
		delete this;
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
// Recurrence relation of form dp_next[i] = min{j in [0, i)} (dp[j] + C[j][i])
// Must satisfy opt[j] <= opt[j + 1]
// Special case: for all a<=b<=c<=d, C[a][c] + C[b][d] <= C[a][d] + C[b][d] ( C is a Monge array )
// O(N log N)
template<typename T>
void DCDP(vector<T> &dp, vector<T> &dp_next, const vector<vector<T>> &C, int low, int high, int optl, int optr){
	if(low >= high) return;
	int mid = low + high >> 1;
	pair<T, int> res{numeric_limits<T>::max(), -1};
	for(int i = optl; i < min(mid, optr); ++ i) res = min(res, {dp[i] + C[i][mid], i});
	dp_next[mid] = res.first;
	DCDP(dp, dp_next, C, low, mid, optl, res.second + 1);
	DCDP(dp, dp_next, C, mid + 1, high, res.second, optr);
}

// 156485479_2_6_3
// Knuth DP Optimization
// Recurrence relation of form dp[i][j] = min{k in [i, j)} (dp[i][k] + dp[k][j] + C[i][j])
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
template<typename Pred>
long long custom_binary_search(long long low, long long high, const long long &step, Pred p, bool is_left = true){
	assert(low < high && (high - low) % step == 0);
	const long long rem = (low % step + step) % step;
	if(is_left){
		while(high - low > step){
			long long mid = low + (high - low >> 1);
			mid = mid / step * step + rem;
			p(mid) ? low = mid : high = mid;
		}
		return low;
	}
	else{
		while(high - low > step){
			long long mid = low + (high - low >> 1);
			mid = mid / step * step + rem;
			p(mid) ? high = mid : low = mid;
		}
		return high;
	}
}
template<typename DP, bool GET_MAX = true>
pair<long long, vector<int>> LagrangeDP(int n, DP f, long long k, long long low, long long high){
	long long resp, resq;
	vector<int> prevp(n + 1), cntp(n + 1), prevq(n + 1), cntq(n + 1);
	auto pred = [&](long long lambda){
		swap(resp, resq), swap(prevp, prevq), swap(cntp, cntq);
		resp = f(lambda, prevp, cntp);
		return GET_MAX ? cntp.back() <= k : cntp.back() >= k;
	};
	long long lambda = custom_binary_search(2 * low - 1, 2 * high + 1, 2, pred);
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

// 156485479_2_7
// Binary Search
// O(log(high - low)) applications of p
template<typename Pred>
long long custom_binary_search(long long low, long long high, Pred p, bool is_left = true){
	assert(low < high);
	if(is_left){
		while(high - low > 1){
			long long mid = low + (high - low >> 1);
			p(mid) ? low = mid : high = mid;
		}
		return low;
	}
	else{
		while(high - low > 1){
			long long mid = low + (high - low >> 1);
			p(mid) ? high = mid : low = mid;
		}
		return high;
	}
}
// Binary search for numbers with the same remainder mod step
template<typename Pred>
long long custom_binary_search(long long low, long long high, const long long &step, Pred p, bool is_left = true){
	assert(low < high && (high - low) % step == 0);
	const long long rem = (low % step + step) % step;
	if(is_left){
		while(high - low > step){
			long long mid = low + (high - low >> 1);
			mid = mid / step * step + rem;
			p(mid) ? low = mid : high = mid;
		}
		return low;
	}
	else{
		while(high - low > step){
			long long mid = low + (high - low >> 1);
			mid = mid / step * step + rem;
			p(mid) ? high = mid : low = mid;
		}
		return high;
	}
}

// 156485479_2_8
// Big Integer
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
template<typename T>
T modinv(T a, T m){
	T u = 0, v = 1;
	while(a){
		T t = m / a;
		m -= t * a; swap(a, m);
		u -= t * v; swap(u, v);
	}
	assert(m == 1);
	return u;
}
template<typename T>
struct Z_p{
	using Type = typename decay<decltype(T::value)>::type;
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
	typename enable_if<is_same<typename Z_p<U>::Type, int>::value, Z_p>::type& operator*=(const Z_p& rhs){
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
	typename enable_if<is_same<typename Z_p<U>::Type, int64_t>::value, Z_p>::type& operator*=(const Z_p &rhs){
		int64_t q = static_cast<int64_t>(static_cast<long double>(value) * rhs.value / mod());
		value = normalize(value * rhs.value - q * mod());
		return *this;
	}
	template<typename U = T>
	typename enable_if<!is_integral<typename Z_p<U>::Type>::value, Z_p>::type& operator*=(const Z_p &rhs){
		value = normalize(value * rhs.value);
		return *this;
	}
	Z_p &operator/=(const Z_p &otr){ return *this *= Z_p(modinv(otr.value, mod())); }
	template<typename U> friend const Z_p<U> &abs(const Z_p<U> &v){ return v; }
	template<typename U> friend bool operator==(const Z_p<U> &lhs, const Z_p<U> &rhs);
	template<typename U> friend bool operator<(const Z_p<U> &lhs, const Z_p<U> &rhs);
	template<typename U> friend istream &operator>>(istream &in, Z_p<U> &number);
	Type value;
};
template<typename T> bool operator==(const Z_p<T> &lhs, const Z_p<T> &rhs){ return lhs.value == rhs.value; }
template<typename T, typename U> bool operator==(const Z_p<T>& lhs, U rhs){ return lhs == Z_p<T>(rhs); }
template<typename T, typename U> bool operator==(U lhs, const Z_p<T> &rhs){ return Z_p<T>(lhs) == rhs; }
template<typename T> bool operator!=(const Z_p<T> &lhs, const Z_p<T> &rhs){ return !(lhs == rhs); }
template<typename T, typename U> bool operator!=(const Z_p<T> &lhs, U rhs){ return !(lhs == rhs); }
template<typename T, typename U> bool operator!=(U lhs, const Z_p<T> &rhs){ return !(lhs == rhs); }
template<typename T> bool operator<(const Z_p<T> &lhs, const Z_p<T> &rhs){ return lhs.value < rhs.value; }
template<typename T> Z_p<T> operator+(const Z_p<T> &lhs, const Z_p<T> &rhs){ return Z_p<T>(lhs) += rhs; }
template<typename T, typename U> Z_p<T> operator+(const Z_p<T> &lhs, U rhs){ return Z_p<T>(lhs) += rhs; }
template<typename T, typename U> Z_p<T> operator+(U lhs, const Z_p<T> &rhs){ return Z_p<T>(lhs) += rhs; }
template<typename T> Z_p<T> operator-(const Z_p<T> &lhs, const Z_p<T> &rhs){ return Z_p<T>(lhs) -= rhs; }
template<typename T, typename U> Z_p<T> operator-(const Z_p<T>& lhs, U rhs){ return Z_p<T>(lhs) -= rhs; }
template<typename T, typename U> Z_p<T> operator-(U lhs, const Z_p<T> &rhs){ return Z_p<T>(lhs) -= rhs; }
template<typename T> Z_p<T> operator*(const Z_p<T> &lhs, const Z_p<T> &rhs){ return Z_p<T>(lhs) *= rhs; }
template<typename T, typename U> Z_p<T> operator*(const Z_p<T>& lhs, U rhs){ return Z_p<T>(lhs) *= rhs; }
template<typename T, typename U> Z_p<T> operator*(U lhs, const Z_p<T> &rhs){ return Z_p<T>(lhs) *= rhs; }
template<typename T> Z_p<T> operator/(const Z_p<T> &lhs, const Z_p<T> &rhs) { return Z_p<T>(lhs) /= rhs; }
template<typename T, typename U> Z_p<T> operator/(const Z_p<T>& lhs, U rhs) { return Z_p<T>(lhs) /= rhs; }
template<typename T, typename U> Z_p<T> operator/(U lhs, const Z_p<T> &rhs) { return Z_p<T>(lhs) /= rhs; }
template<typename T> istream &operator>>(istream &in, Z_p<T> &number){
	typename common_type<typename Z_p<T>::Type, int64_t>::type x;
	in >> x;
	number.value = Z_p<T>::normalize(x);
	return in;
}
template<typename T> ostream &operator<<(ostream &out, const Z_p<T> &number){ return out << number(); }
constexpr int mod = (int)1e9 + 7;
using Zp = Z_p<integral_constant<decay<decltype(mod)>::type, mod>>;

// 156485479_3_1
// Sparse Table
// The binary operator must be idempotent and associative
// O(N log N) preprocessing, O(1) per query
template<typename T, typename BO = function<T(T, T)>>
struct sparse_table{
	int N;
	BO bin_op;
	vector<vector<T>> val;
	sparse_table(const vector<T> &arr, BO bin_op = [](T x, T y){return min(x, y);}): N(arr.size()), bin_op(bin_op){
		int t = 1, d = 1;
		while(t < N) t *= 2, ++ d;
		val.assign(d, arr);
		for(int i = 0; i < d - 1; ++ i) for(int j = 0; j < N; ++ j){
			val[i + 1][j] = bin_op(val[i][j], val[i][min(N - 1, j + (1 << i))]);
		}
	}
	sparse_table(){ }
	T query(int l, int r){
		int d = 31 - __builtin_clz(r - l);
		return bin_op(val[d][l], val[d][r - (1 << d)]);
	}
	sparse_table &operator=(const sparse_table &otr){
		N = otr.N, bin_op = otr.bin_op;
		val = otr.val;
		return *this;
	}
};

// 156485479_3_2_1
// Iterative Segment Tree
// O(N) processing, O(log N) per query
template<typename T, typename BO>
struct segment{
	int N;
	BO bin_op;
	const T id;
	vector<T> val;
	segment(const vector<T> &arr, BO bin_op, T id): N(arr.size()), bin_op(bin_op), id(id), val(N << 1, id){
		for(int i = 0; i < N; ++ i) val[i + N] = arr[i];
		for(int i = N - 1; i > 0; -- i) val[i] = bin_op(val[i << 1], val[i << 1 | 1]);
	}
	void set(int p, T x){
		for(p += N, val[p] = x; p > 1; p >>= 1) val[p >> 1] = bin_op(val[p], val[p ^ 1]);
	}
	void update(int p, T x){
		for(p += N, val[p] = bin_op(val[p], x); p > 1; p >>= 1) val[p >> 1] = bin_op(val[p], val[p ^ 1]);
	}
	T query(int l, int r){
		if(l >= r) return id;
		T resl = id, resr = id;
		for(l += N, r += N; l < r; l >>= 1, r >>= 1){
			if(l & 1) resl = bin_op(resl, val[l ++]);
			if(r & 1) resr = bin_op(val[-- r], resr);
		}
		return bin_op(resl, resr);
	}
};

// 156485479_3_2_2
// Iterative Segment Tree with Reversed Operation ( Commutative Operation Only )
// O(N) Preprocessing, O(1) per query
template<typename T, typename BO>
struct segment{
	int N;
	BO bin_op;
	T id;
	vector<T> val;
	segment(const vector<T> &arr, BO bin_op, T id): N(arr.size()), bin_op(bin_op), id(id), val(N << 1, id){
		for(int i = 0; i < N; ++ i) val[i + N] = arr[i];
	}
	void update(int l, int r, T x){
		for(l += N, r += N; l < r; l >>= 1, r >>= 1){
			if(l & 1) val[l ++] = bin_op(val[l], x);
			if(r & 1) val[r] = bin_op(val[-- r], x);
		}
	}
	T query(int p){
		T res = id;
		for(p += N; p > 0; p >>= 1) res = bin_op(res, val[p]);
		return res;
	}
	void push(){
		for(int i = 1; i < N; ++ i){
			val[i << 1] = bin_op(val[i << 1], val[i]);
			val[i << 1 | 1] = bin_op(val[i << 1 | 1], val[i]);
			val[i] = id;
		}
	}
};

// 156485479_3_2_3
// Simple Recursive Segment Tree
// O(N) preprocessing, O(log N) per query
template<typename T, typename BO>
struct segment{
	int N;
	BO bin_op;
	const T id;
	vector<T> val;
	segment(const vector<T> &arr, BO bin_op, T id): N(arr.size()), bin_op(bin_op), id(id), val(N << 2, id){
		build(arr, 1, 0, N);
	}
	void build(const vector<T> &arr, int u, int left, int right){
		if(left + 1 == right) val[u] = arr[left];
		else{
			int mid = left + right >> 1;
			build(arr, u << 1, left, mid);
			build(arr, u << 1 ^ 1, mid, right);
			val[u] = bin_op(val[u << 1], val[u << 1 ^ 1]);
		}
	}
	T pq(int u, int left, int right, int ql, int qr){
		if(qr <= left || right <= ql) return id;
		if(ql == left && qr == right) return val[u];
		int mid = left + right >> 1;
		return bin_op(pq(u << 1, left, mid, ql, qr), pq(u << 1 ^ 1, mid, right, ql, qr));
	}
	T query(int ql, int qr){
		return pq(1, 0, N, ql, qr);
	}
	void pu(int u, int left, int right, int ind, T x){
		if(left + 1 == right) val[u] = x;
		else{
			int mid = left + right >> 1;
			if(ind < mid) pu(u << 1, left, mid, ind, x);
			else pu(u << 1 ^ 1, mid, right, ind, x);
			val[u] = bin_op(val[u << 1], val[u << 1 ^ 1]);
		}
	}
	void update(int ind, T x){
		pu(1, 0, N, ind, x);
	}
	// Below assumes T is an ordered field and node stores positive values
	template<typename IO>
	int plb(int u, int left, int right, T x, IO inv_op){
		if(left + 1 == right) return right;
		int mid = left + right >> 1;
		if(val[u << 1] < x) return plb(u << 1 ^ 1, mid, right, inv_op(x, val[u << 1]), inv_op);
		else return plb(u << 1, left, mid, x, inv_op);
	}
	template<typename IO>
	int lower_bound(T x, IO inv_op){ // min i such that query[0, i) >= x
		if(val[1] < x) return N + 1;
		else return plb(1, 0, N, x, inv_op);
	}
	template<typename IO>
	int lower_bound(int i, T x, IO inv_op){
		return lower_bound(bin_op(x, query(0, min(i, N))), inv_op);
	}
	template<typename IO>
	int pub(int u, int left, int right, T x, IO inv_op){
		if(left + 1 == right) return right;
		int mid = left + right >> 1;
		if(x < val[u << 1]) return pub(u << 1, left, mid, x, inv_op);
		else return pub(u << 1 ^ 1, mid, right, inv_op(x, val[u << 1]), inv_op);
	}
	template<typename IO>
	int upper_bound(T x, IO inv_op){ // min i such that query[0, i) > x
		if(x < val[1]) return pub(1, 0, N, x, inv_op);
		else return N + 1;
	}
	template<typename IO>
	int upper_bound(int i, T x, IO inv_op){
		return upper_bound(bin_op(x, query(0, min(i, N))), inv_op);
	}
};

// 156485479_3_2_4
// Iterative 2D Segment Tree ( Only for commutative group )
// O(NM log NM) processing, O(log NM) per query
template<typename T, typename BO>
struct segment{
	int N, M;
	BO bin_op;
	const T id;
	vector<vector<T>> val;
	segment(const vector<vector<T>> &arr, BO bin_op, T id): N(arr.size()), M(arr[0].size()), bin_op(bin_op), id(id), val(N << 1, vector<T>(M << 1, id)){
		for(int i = 0; i < N; ++ i) for(int j = 0; j < N; ++ j) val[i + N][j + N] = arr[i][j];
		for(int i = N - 1; i > 0; -- i) for(int j = 0; j < N; ++ j) val[i][j + N] = bin_op(val[i << 1][j + N], val[i << 1 | 1][j + N]);
		for(int i = 1; i < N << 1; ++ i) for(int j = N - 1; j > 0; -- j) val[i][j] = bin_op(val[i][j << 1], val[i][j << 1 | 1]);
	}
	void set(int p, int q, T x){
		val[p += N][q += M] = x;
		for(int j = q; j >>= 1; ) val[p][j] = bin_op(val[p][j << 1], val[p][j << 1 | 1]);
		for(int i = p; i >>= 1; ){
			val[i][q] = bin_op(val[i << 1][q], val[i << 1 | 1][q]);
			for(int j = q; j >>= 1; ) val[i][j] = bin_op(val[i][j << 1], val[i][j << 1 | 1]);
		}
	}
	void update(int p, int q, T x){
		p += N, q += N, val[p][q] = bin_op(val[p][q], x);
		for(int j = q; j >>= 1; ) val[p][j] = bin_op(val[p][j << 1], val[p][j << 1 | 1]);
		for(int i = p; i >>= 1; ){
			val[i][q] = bin_op(val[i << 1][q], val[i << 1 | 1][q]);
			for(int j = q; j >>= 1; ) val[i][j] = bin_op(val[i][j << 1], val[i][j << 1 | 1]);
		}
	}
	T query(int pl, int ql, int pr, int qr){
		if(pl >= pr || ql >= qr) return id;
		T res = id;
		for(int il = pl + N, ir = pr + N; il < ir; il >>= 1, ir >>= 1){
			if(il & 1){
				for(int jl = ql + N, jr = qr + N; jl < jr; jl >>= 1, jr >>= 1){
					if(jl & 1) res = bin_op(res, val[il][jl ++]);
					if(jr & 1) res = bin_op(res, val[il][-- jr]);
				}
				++ il;
			}
			if(ir & 1){
				-- ir;
				for(int jl = ql + N, jr = qr + N; jl < jr; jl >>= 1, jr >>= 1){
					if(jl & 1) res = bin_op(res, val[ir][jl ++]);
					if(jr & 1) res = bin_op(res, val[ir][-- jr]);
				}
			}
		}
		return res;
	}
};

// 156485479_3_2_5
// Lazy Dynamic Segment Tree
// O(1) or O(N) preprocessing, O(log L) or O(log N) per query
template<typename T, typename LOP, typename QOP, typename AOP, typename INIT = function<T(int, int)>>
struct segment{
	segment *l = 0, *r = 0;
	int low, high;
	LOP lop;                // Lazy op(L, L -> L)
	QOP qop;                // Query op(Q, Q -> Q)
	AOP aop;                // Apply op(Q, L, leftend, rightend -> Q)
	const vector<T> id;     // Lazy id(L), Query id(Q), Disable constant(Q)
	T lset, lazy, val;
	INIT init;
	segment(int low, int high, LOP lop, QOP qop, AOP aop, const vector<T> &id, INIT init): low(low), high(high), lop(lop), qop(qop), aop(aop), id(id), init(init){
		lazy = id[0], lset = id[2], val = init(low, high);
	}
	segment(const vector<T> &arr, int low, int high, LOP lop, QOP qop, AOP aop, const vector<T> &id): low(low), high(high), lop(lop), qop(qop), aop(aop), id(id){
		lazy = id[0], lset = id[2];
		if(high - low > 1){
			int mid = low + (high - low) / 2;
			l = new segment(arr, low, mid, lop, qop, aop, id);
			r = new segment(arr, mid, high, lop, qop, aop, id);
			val = qop(l->val, r->val);
		}
		else val = arr[low];
	}
	void push(){
		if(!l){
			int mid = low + (high - low) / 2;
			l = new segment(low, mid, lop, qop, aop, id);
			r = new segment(mid, high, lop, qop, aop, id);
		}
		if(lset != id[2]){
			l->set(low, high, lset);
			r->set(low, high, lset);
			lset = id[2];
		}
		else if(lazy != id[0]){
			l->update(low, high, lazy);
			r->update(low, high, lazy);
			lazy = id[0];
		}
	}
	void set(int ql, int qr, T x){
		if(qr <= low || high <= ql) return;
		if(ql <= low && high <= qr){
			lset = x;
			lazy = id[0];
			val = aop(id[1], x, low, high);
		}
		else{
			push();
			l->set(ql, qr, x);
			r->set(ql, qr, x);
			val = qop(l->val, r->val);
		}
	}
	void update(int ql, int qr, T x){
		if(qr <= low || high <= ql) return;
		if(ql <= low && high <= qr){
			if(lset != id[2]) lset = lop(lset, x);
			else lazy = lop(lazy, x);
			val = aop(val, x, low, high);
		}
		else{
			push();
			l->update(ql, qr, x);
			r->update(ql, qr, x);
			val = qop(l->val, r->val);
		}
	}
	T query(int ql, int qr){
		if(qr <= low || high <= ql) return id[1];
		if(ql <= low && high <= qr) return val;
		push();
		return qop(l->query(ql, qr), r->query(ql, qr));
	}
	// Below assumes T is an ordered field and node stores positive values
	template<typename IO>
	int plb(T x, IO inv_op){
		if(low + 1 == high) return high;
		push();
		if(l->val < x) return r->plb(inv_op(x, l->val), inv_op);
		else return l->plb(x, inv_op);
	}
	template<typename IO>
	int lower_bound(T x, IO inv_op){ // min i such that query[0, i) >= x
		if(val < x) return high + 1;
		else return plb(x, inv_op);
	}
	template<typename IO>
	int lower_bound(int i, T x, IO inv_op){
		return lower_bound(qop(x, query(low, min(i, high))), inv_op);
	}
	template<typename IO>
	int pub(T x, IO inv_op){
		if(low + 1 == high) return high;
		push();
		if(x < l->val) return l->pub(x, inv_op);
		else return r->pub(inv_op(x, l->val), inv_op);
	}
	template<typename IO>
	int upper_bound(T x, IO inv_op){ // min i such that query[0, i) > val
		if(x < val) return pub(x, inv_op);
		else return high + 1;
	}
	template<typename IO>
	int upper_bound(int i, T x, IO inv_op){
		return upper_bound(qop(x, query(low, min(i, high))), inv_op);
	}
};

// 156485479_3_2_6
// Persistent Segment Tree
// O(N) preprocessing, O(log N) per query
template<typename T>
struct node{
	node *l = 0, *r = 0;
	T val;
	node(T val): val(val){}
	node(node *l, node *r, function<T(T, T)> bin_op, T id): l(l), r(r), val(id){
		if(l) val = bin_op(l->val, val);
		if(r) val = bin_op(val, r->val);
	}
};
template<typename T, typename BO>
struct segment: vector<node<T> *>{
	int N;
	BO bin_op;
	const T id;
	segment(const vector<T> &arr, BO bin_op, T id): N(arr.size()), bin_op(bin_op), id(id){
		this->push_back(build(arr, 0, N));
	}
	node<T> *build(const vector<T> &arr, int left, int right){
		if(left + 1 == right) return new node<T>(arr[left]);
		int mid = left + right >> 1;
		return new node<T>(build(arr, left, mid), build(arr, mid, right), bin_op, id);
	}
	T pq(node<T> *u, int left, int right, int ql, int qr){
		if(qr <= left || right <= ql) return id;
		if(ql <= left && right <= qr) return u->val;
		int mid = left + right >> 1;
		return bin_op(pq(u->l, left, mid, ql, qr), pq(u->r, mid, right, ql, qr));
	}
	T query(node<T> *u, int ql, int qr){
		return pq(u, 0, N, ql, qr);
	}
	node<T> *ps(node<T> *u, int left, int right, int p, int x){
		if(left + 1 == right) return new node<T>(x);
		int mid = left + right >> 1;
		if(mid > p) return new node<T>(ps(u->l, left, mid, p, x), u->r, bin_op, id);
		else return new node<T>(u->l, ps(u->r, mid, right, p, x), bin_op, id);
	}
	void set(node<T> *u, int p, int x){
		this->push_back(ps(u, 0, N, p, x));
	}
	// Below assumes T is an ordered field and node stores positive values
	template<typename IO>
	int plb(node<T> *u, int left, int right, T x, IO inv_op){
		if(left + 1 == right) return right;
		int mid = left + right >> 1;
		if(u->l->val < x) return plb(u->r, mid, right, inv_op(x, u->l->val), inv_op);
		else return plb(u->l, left, mid, x, inv_op);
	}
	template<typename IO>
	int lower_bound(node<T> *u, T x, IO inv_op){ // min i such that query[0, i) >= x
		if(u->val < x) return N + 1;
		else return plb(u, 0, N, x, inv_op);
	}
	template<typename IO>
	int lower_bound(node<T> *u, int i, T x, IO inv_op){
		return lower_bound(u, bin_op(x, query(u, 0, min(i, N))), inv_op);
	}
	template<typename IO>
	int pub(node<T> *u, int left, int right, T x, IO inv_op){
		if(left + 1 == right) return right;
		int mid = left + right >> 1;
		if(x < u->l->val) return pub(u->l, left, mid, x, inv_op);
		else return pub(u->r, mid, right, inv_op(x, u->l->val), inv_op);
	}
	template<typename IO>
	int upper_bound(node<T> *u, T x, IO inv_op){ // min i such that query[0, i) > x
		if(x < u->val) return pub(u, 0, N, x, inv_op);
		else return N + 1;
	}
	template<typename IO>
	int upper_bound(node<T> *u, int i, T x, IO inv_op){
		return upper_bound(u, bin_op(x, query(u, 0, min(i, N))), inv_op);
	}
};

// 156485479_3_3_1
// Fenwick Tree
// Only works on a commutative group
// O(N log N) preprocessing, O(log N) per query
template<typename T, typename BO, typename IO>
struct fenwick{
	int N;
	BO bin_op;
	IO inv_op;
	const T id;
	vector<T> val;
	fenwick(const vector<T> &arr, BO bin_op, IO inv_op, T id): N(arr.size()), bin_op(bin_op), inv_op(inv_op), id(id), val(N + 1, id){
		for(int i = 0; i < N; ++ i) update(i, arr[i]);
	}
	void set(int p, T x){
		for(x = inv_op(x, query(p, p + 1)), ++ p; p <= N; p += p & -p) val[p] = bin_op(val[p], x);
	}
	void update(int p, T x){
		for(++ p; p <= N; p += p & -p) val[p] = bin_op(val[p], x);
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
// O(N log N) preprocessing, O(log N) per query
template<typename T, typename BO, typename IO>
struct fenwick{
	int N;
	BO bin_op;
	IO inv_op;
	const T id;
	vector<T> val;
	fenwick(const vector<T> &arr, BO bin_op, IO inv_op, T id): N(arr.size()), bin_op(bin_op), inv_op(inv_op), id(id), val(N + 1, id){
		for(int i = 0; i < N; ++ i) update(i, arr[i]);
	}
	void set(int p, T x){
		for(x = inv_op(x, query(p, p + 1)), ++ p; p <= N; p += p & -p) val[p] = bin_op(val[p], x);
	}
	void update(int p, T x){
		for(++ p; p <= N; p += p & -p) val[p] = bin_op(val[p], x);
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
template<typename T, typename BO, typename IO, typename MO>
struct rangefenwick{
	fenwick<T, BO, IO> tr1, tr2;
	BO bin_op;
	IO inv_op;
	MO multi_op;
	const T id;
	rangefenwick(int N, BO bin_op, IO inv_op, MO multi_op, T id):
		tr1(vector<T>(N, id), bin_op, inv_op, id),
		tr2(vector<T>(N, id), bin_op, inv_op, id),
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
// O(NM log NM) preprocessing, O(log N log M) per query
template<typename T, typename BO, typename IO>
struct fenwick{
	int N, M;
	BO bin_op;
	IO inv_op;
	const T id;
	vector<vector<T>> val;
	fenwick(const vector<vector<T>> &arr, BO bin_op, IO inv_op, T id): N(arr.size()), M(arr[0].size()), bin_op(bin_op), inv_op(inv_op), id(id), val(N + 1, vector<T>(M + 1)){
		for(int i = 0; i < N; ++ i) for(int j = 0; j < M; ++ j) update(i, j, arr[i][j]);
	}
	void set(int p, int q, T x){
		x = inv_op(x, query(p, q, p + 1, q + 1)), ++ p, ++ q;
		for(int i = p; i <= N; i += i & -i) for(int j = q; j <= N; j += j & -j) val[i][j] = bin_op(val[i][j], x);
	}
	void update(int p, int q, T x){
		++ p, ++ q;
		for(int i = p; i <= N; i += i & -i) for(int j = q; j <= N; j += j & -j) val[i][j] = bin_op(val[i][j], x);
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
// Wavelet Tree ( WARNING: NOT THOROUGHLY TESTED YET )
// O(L log N) preprocessing, O(log N) per query
struct node: vector<int>{
	int N, low, high;
	node *l = NULL, *r = NULL;
	node(vector<int>::iterator bg, vector<int>::iterator ed, int low, int high, function<bool(int, int)> cmp):
		N(ed - bg), low(low), high(high){
		if(!N) return;
		if(low + 1 == high){
			this->resize(N + 1);
			iota(this->begin(), this->end(), 0);
			return;
		}
		int mid = low + high >> 1;
		auto pred = [&](int x){return cmp(x, mid);};
		this->reserve(N + 1);
		this->push_back(0);
		for(auto it = bg; it != ed; it ++){
			this->push_back(this->back() + pred(*it));
		}
		auto p = stable_partition(bg, ed, pred);
		l = new node(bg, p, low, mid, cmp);
		r = new node(p, ed, mid, high, cmp);
	}
};
struct wavelet{
	int N;
	node *root;
	function<bool(int, int)> cmp;
	vector<int> arr;
	wavelet(const vector<int> &other, function<bool(int, int)> cmp = less<int>()):
		N(other.size()), arr(other), cmp(cmp){
		root = new node(arr.begin(), arr.end(), *min_element(arr.begin(), arr.end(), cmp), *max_element(arr.begin(), arr.end(), cmp) + 1, cmp);
	}
	//Count elements less than val in the range [l, r)
	int count(node *node, int ql, int qr, int val){
		if(ql >= qr || !cmp(node->low, val)) return 0;
		if(!cmp(val, node->high)) return qr - ql;
		int Lcnt = (*node)[ql], Rcnt = (*node)[qr];
		return count(node->l, Lcnt, Rcnt, val) + count(node->r, ql - Lcnt, qr - Rcnt, val);
	}
	//Find the kth element in the range [l, r)
	int kth(node *node, int ql, int qr, int k){
		if(k > node->N) return node->high;
		if(k <= 0) return node->low - 1;
		if(node->low + 1 == node->high) return node->low;
		int Lcnt = (*node)[ql], Rcnt = (*node)[qr];
		if(k <= node->l->N) return kth(node->l, Lcnt, Rcnt, k);
		else return kth(node->r, ql - Lcnt, qr - Rcnt, k - node->l->N);
	}
};

// 156485479_3_5
// Disjoint Set
// O(alpha(n)) per query where alpha(n) is the inverse ackermann function
struct disjoint{
	int N;
	vector<int> parent, size;
	// vector<pair<int, int>> Log_parent, Log_size; // For persistency
	disjoint(int N): N(N), parent(N), size(N, 1){ iota(parent.begin(), parent.end(), 0); }
	void expand(){
		++ N;
		parent.push_back(parent.size());
		size.push_back(1);
	}
	int root(int u){
		// Log_parent.emplace_back(u, parent[u]);
		return parent[u] == u ? u : parent[u] = root(parent[u]);
	}
	bool merge(int u, int v){
		u = root(u), v = root(v);
		if(u == v) return false;
		if(size[u] < size[v]) swap(u, v);
		// Log_parent.emplace_back(v, parent[v]);
		parent[v] = u;
		// Log_size.emplace_back(u, size[u]);
		size[u] += size[v];
		return true;
	}
	bool share(int u, int v){ return root(parent[u]) == root(parent[v]); }
	/*void reverse(int p, int s){
		while(parent.size() != p) reverse_parent();
		while(size.size() != s) reverse_size();
	}
	void reverse_parent(){
		auto [u, p] = Log_parent.back();
		Log_parent.pop_back();
		parent[u] = p;
	}
	void reverse_size(){
		auto [u, s] = Log_size.back();
		Log_size.pop_back();
		size[u] = s;
	}*/
};

// 156485479_3_6
// Monotone Stack
// O(1) per operation
template<typename T = int, typename Compare = function<bool(T, T)>>
struct monotone_stack: vector<T>{
	T init;
	Compare cmp;
	monotone_stack(T init = 0, Compare cmp = less<T>{}): init(init), cmp(cmp){ }
	T push(T x){
		while(!this->empty() && !cmp(this->back(), x)) this->pop_back();
		this->push_back(x);
		return this->size() == 1 ? init : *-- -- this->end();
	}
};

// 156485479_3_7
// Less-than-k Query, Distinct Value Query (Offline, Online)
// O(N log N) processing
template<typename T, typename BO, typename IO>
struct fenwick{
	int N;
	BO bin_op;
	IO inv_op;
	const T id;
	vector<T> val;
	fenwick(const vector<T> &arr, BO bin_op, IO inv_op, T id): N(arr.size()), bin_op(bin_op), inv_op(inv_op), id(id), val(N + 1, id){
		for(int i = 0; i < N; ++ i) update(i, arr[i]);
	}
	void set(int p, T x){
		for(x = inv_op(x, query(p, p + 1)), ++ p; p <= N; p += p & -p) val[p] = bin_op(val[p], x);
	}
	void update(int p, T x){
		for(++ p; p <= N; p += p & -p) val[p] = bin_op(val[p], x);
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
template<typename T>
struct offline_less_than_k_query{
	int N;
	vector<pair<T, int>> event;
	vector<tuple<T, int, int, int>> queries;
	offline_less_than_k_query(const vector<T> &arr, bool IS_DVQ = true): N(arr.size()), event(N){
		if(IS_DVQ){
			map<T, int> q;
			for(int i = 0; i < N; ++ i){
				event[i] = {(q.count(arr[i]) ? q[arr[i]] : -1), i};
				q[arr[i]] = i;
			}
		}
		else for(int i = 0; i < N; ++ i) event[i] = {arr[i], i};
	}
	void query(int i, int ql, int qr){ // For distinct value query
		queries.emplace_back(ql, ql, qr, i);
	}
	void query(int i, int ql, int qr, T k){ // For less-than-k query
		queries.emplace_back(k, ql, qr, i);
	}
	template<typename Action>
	void solve(Action ans){ // ans(index, answer)
		sort(queries.begin(), queries.end()), sort(event.begin(), event.end(), greater<pair<T, int>>());
		fenwick tr(vector<int>(N), plus<int>(), minus<int>(), 0);
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
template<typename T>
struct node{
	node *l = 0, *r = 0;
	T val;
	node(T val): val(val){}
	node(node *l, node *r, function<T(T, T)> bin_op, T id): l(l), r(r), val(id){
		if(l) val = bin_op(l->val, val);
		if(r) val = bin_op(val, r->val);
	}
};
template<typename T, typename BO>
struct segment: vector<node<T> *>{
	int N;
	BO bin_op;
	const T id;
	segment(const vector<T> &arr, BO bin_op, T id): N(arr.size()), bin_op(bin_op), id(id){
		this->push_back(build(arr, 0, N));
	}
	node<T> *build(const vector<T> &arr, int left, int right){
		if(left + 1 == right) return new node<T>(arr[left]);
		int mid = left + right >> 1;
		return new node<T>(build(arr, left, mid), build(arr, mid, right), bin_op, id);
	}
	T pq(node<T> *u, int left, int right, int ql, int qr){
		if(qr <= left || right <= ql) return id;
		if(ql <= left && right <= qr) return u->val;
		int mid = left + right >> 1;
		return bin_op(pq(u->l, left, mid, ql, qr), pq(u->r, mid, right, ql, qr));
	}
	T query(node<T> *u, int ql, int qr){
		return pq(u, 0, N, ql, qr);
	}
	node<T> *ps(node<T> *u, int left, int right, int p, int x){
		if(left + 1 == right) return new node<T>(x);
		int mid = left + right >> 1;
		if(mid > p) return new node<T>(ps(u->l, left, mid, p, x), u->r, bin_op, id);
		else return new node<T>(u->l, ps(u->r, mid, right, p, x), bin_op, id);
	}
	void set(node<T> *u, int p, int x){
		this->push_back(ps(u, 0, N, p, x));
	}
	// Below assumes T is an ordered field and node stores positive values
	template<typename IO>
	int plb(node<T> *u, int left, int right, T x, IO inv_op){
		if(left + 1 == right) return right;
		int mid = left + right >> 1;
		if(u->l->val < x) return plb(u->r, mid, right, inv_op(x, u->l->val), inv_op);
		else return plb(u->l, left, mid, x, inv_op);
	}
	template<typename IO>
	int lower_bound(node<T> *u, T x, IO inv_op){ // min i such that query[0, i) >= x
		if(u->val < x) return N + 1;
		else return plb(u, 0, N, x, inv_op);
	}
	template<typename IO>
	int lower_bound(node<T> *u, int i, T x, IO inv_op){
		return lower_bound(u, bin_op(x, query(u, 0, min(i, N))), inv_op);
	}
	template<typename IO>
	int pub(node<T> *u, int left, int right, T x, IO inv_op){
		if(left + 1 == right) return right;
		int mid = left + right >> 1;
		if(x < u->l->val) return pub(u->l, left, mid, x, inv_op);
		else return pub(u->r, mid, right, inv_op(x, u->l->val), inv_op);
	}
	template<typename IO>
	int upper_bound(node<T> *u, T x, IO inv_op){ // min i such that query[0, i) > x
		if(x < u->val) return pub(u, 0, N, x, inv_op);
		else return N + 1;
	}
	template<typename IO>
	int upper_bound(node<T> *u, int i, T x, IO inv_op){
		return upper_bound(u, bin_op(x, query(u, 0, min(i, N))), inv_op);
	}
};
template<typename T>
struct less_than_k_query{ // for less-than-k query, it only deals with numbers in range [0, N)
	int N;
	vector<node<T> *> p;
	segment<int, plus<int>> tr;
	less_than_k_query(const vector<T> &arr, bool IS_DVQ = true): N(arr.size()), p(N + 1), tr(vector<int>(N), plus<int>{}, 0){
		vector<pair<T, int>> event(N);
		if(IS_DVQ){
			map<T, int> q;
			for(int i = 0; i < N; ++ i){
				event[i] = {(q.count(arr[i]) ? q[arr[i]] : -1), i};
				q[arr[i]] = i;
			}
		}
		else for(int i = 0; i < N; ++ i) event[i] = {arr[i], i};
		sort(event.begin(), event.end(), greater<pair<int, int>>{});
		tr.reserve(N);
		for(int i = 0; i <= N; ++ i){
			while(!event.empty() && event.back().first < i){
				tr.set(tr.back(), event.back().second, 1);
				event.pop_back();
			}
			p[i] = tr.back();
		}
	}
	// For distinct value query
	int query(int ql, int qr){
		return tr.query(p[ql], ql, qr);
	}
	int lower_bound(int ql, int cnt){ // min i such that # of distinct in [l, l + i) >= cnt
		return tr.lower_bound(p[ql], ql, cnt, minus<int>());
	}
	int upper_bound(int ql, int cnt){ // min i such that # of distinct in [l, l + i) > cnt
		return tr.upper_bound(p[ql], ql, cnt, minus<int>());
	}
	// For less-than-k query
	int query(int ql, int qr, int k){
		return tr.query(p[k], ql, qr);
	}
	int lower_bound(int ql, int k, int cnt){ // min i such that ( # of elements < k in [l, l + i) ) >= cnt
		return tr.lower_bound(p[k], ql, cnt, minus<int>());
	}
	int upper_bound(int ql, int k, int cnt){ // min i such that ( # of elements < k in [l, l + i) ) > cnt
		return tr.upper_bound(p[k], ql, cnt, minus<int>());
	}
};

// 156485479_3_8
// Mo's Algorithm
// O((N + Q) sqrt(N) F) where F is the processing time of ins and del.
template<int B>
struct Query{
	int l, r, ind;
	bool operator<(const Query &otr) const{
		if(l / B != otr.l / B) return pair<int, int>(l, r) < pair<int, int>(otr.l, otr.r);
		return (l / B & 1) ? (r < otr.r) : (r > otr.r);
	}
};
template<typename T, typename Q, typename I, typename D, typename A>
vector<T> answer_query_offline(const vector<T> &arr, vector<Q> query, I ins, D del, A ans){
	sort(query.begin(), query.end());
	vector<T> res(query.size());
	l = 0, r = 0;
	for(auto q: query){
		while(l > q.l) ins(-- l);
		while(r < q.r) ins(r ++);
		while(l < q.l) del(l ++);
		while(r > q.r) del(-- r);
		res[q.ind] = ans();
	}
	return res;
}

// 156485479_4_1
// Strongly Connected Component ( Tarjan's Algorithm ) / Processes SCCs in reverse topological order
// O(N + M)
template<typename Graph, typename Process_SCC>
int scc(const Graph &adj, Process_SCC f){
	int n = int(adj.size());
	vector<int> val(n), comp(n, -1), z, cur;
	int timer = 0, ncomps = 0;
	function<void(int)> dfs = [&](int u){
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

// 156485479_4_2
// Biconnected Components / adj[u]: list of [vertex, edgenum]
// O(N + M)
template<typename Graph, typename Process_BCC, typename Process_Bridge = function<void(int, int, int)>>
int bcc(const Graph &adj, Process_BCC f, Process_Bridge g = [](int u, int v, int e){ }){
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
				int si = int(st.size());
				int up = dfs(v, e);
				top = min(top, up);
				if(up == me){
					st.push_back(e);
					f(vector<int>(st.begin() + si, st.end()));
					st.resize(si);
					ncomps ++;
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
// O(N + M)
template<typename Graph, typename Process_Articulation_Point>
void articulation_points(const Graph &adj, Process_Articulation_Point f){
	int n = adj.size();
	vector<bool> visited(n);
	vector<int> tin(n, -1), low(n, -1);
	int timer = 0;
	function<void(int, int)> dfs = [&](int u, int p){
		visited[u] = true;
		tin[u] = low[u] = timer ++;
		int child = 0;
		for(auto v: adj[u]){
			if(v == p) continue;
			if(visited[v]) low[u] = min(low[u], tin[v]);
			else{
				dfs(v, u);
				low[u] = min(low[u], low[v]);
				if(low[v] >= tin[u] && p != -1) f(u);
				++ child;
			}
		}
		if(p == -1 && child > 1) f(u);
	};
	for(int u = 0; u < n; ++ u) if(!visited[u]) dfs(u, -1);
}

// 156485479_4_4_1
// Dinic's Maximum Flow Algorithm
// O(V^2E) ( O(sqrt(V) * E ) for unit network )
template<typename T>
struct flownetwork{
	static constexpr T eps = (T)1e-9;
	int N;
	vector<vector<int>> adj;
	struct edge{
		int from, to;
		T capacity, flow;
	};
	vector<edge> edge;
	int source, sink;
	T flow = 0;
	flownetwork(int N, int source, int sink): N(N), source(source), sink(sink), adj(N){ }
	void clear(){
		for(auto &e: edge) e.flow = 0;
		flow = 0;
	}
	int insert(int from, int to, T fcap, T bcap){
		int ind = edge.size();
		adj[from].push_back(ind);
		edge.push_back({from, to, fcap, 0});
		adj[to].push_back(ind + 1);
		edge.push_back({to, from, bcap, 0});
		return ind;
	}
};
template<typename T>
struct dinic{
	static constexpr T inf = numeric_limits<T>::max();
	flownetwork<T> &g;
	vector<int> ptr, level, q;
	dinic(flownetwork<T> &g): g(g), ptr(g.N), level(g.N), q(g.N){ }
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
					g.edge[ind].flow += F;
					g.edge[ind ^ 1].flow -= F;
					return F;
				}
			}
			-- j;
		}
		return 0;
	}
	T max_flow(){
		while(bfs()){
			for(int i = 0; i < g.N; ++ i) ptr[i] = int(g.adj[i].size()) - 1;
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
		vector<bool> res(g.N);
		for(int i = 0; i < g.N; ++ i) res[i] = (level[i] != -1);
		return {cut, res};
	}
};

// 156485479_4_4_2
// Minimum Cost Maximum Flow Algorithm
// O(N^2 M^2)
template<typename T, typename C>
struct mcmf{
	static constexpr T eps = (T) 1e-9;
	struct edge{
		int from, to;
		T c, f;
		C cost;
	};
	vector<vector<int>> g;
	vector<edge> edges;
	vector<C> d;
	vector<bool> in_queue;
	vector<int> q, pe;
	int n, source, sink;
	T flow;
	C cost;
	mcmf(int n, int source, int sink): n(n), source(source), sink(sink), g(n), d(n), in_queue(n), pe(n){
		assert(0 <= source && source < n && 0 <= sink && sink < n && source != sink);
		flow = cost = 0;
	}
	void clear_flow(){
		for(const edge &e: edges) e.f = 0;
		flow = 0;
	}
	void add(int from, int to, T forward_cap, T backward_cap, C cost){
		assert(0 <= from && from < n && 0 <= to && to < n);
		g[from].push_back((int) edges.size());
		edges.push_back({from, to, forward_cap, 0, cost});
		g[to].push_back((int) edges.size());
		edges.push_back({to, from, backward_cap, 0, -cost});
	}
	bool expath(){
		fill(d.begin(), d.end(), numeric_limits<C>::max());
		q.clear();
		q.push_back(source);
		d[source] = 0;
		in_queue[source] = true;
		int beg = 0;
		bool found = false;
		while(beg < q.size()){
			int i = q[beg ++];
			if(i == sink) found = true;
			in_queue[i] = false;
			for(int id : g[i]){
				const edge &e = edges[id];
				if(e.c - e.f > eps && d[i] + e.cost < d[e.to]){
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
				const edge &e = edges[pe[v]];
				push = min(push, e.c - e.f);
				v = e.from;
			}
			v = sink;
			while(v != source){
				edge &e = edges[pe[v]];
				e.f += push;
				edge &back = edges[pe[v] ^ 1];
				back.f -= push;
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

// 156485479_4_5_1
// LCA
// O(N log N) precomputing, O(1) per query
template<typename T, typename BO = function<T(T, T)>>
struct sparse_table{
	int N;
	BO bin_op;
	vector<vector<T>> val;
	sparse_table(const vector<T> &arr, BO bin_op = [](T x, T y){return min(x, y);}): N(arr.size()), bin_op(bin_op){
		int t = 1, d = 1;
		while(t < N) t *= 2, ++ d;
		val.assign(d, arr);
		for(int i = 0; i < d - 1; ++ i) for(int j = 0; j < N; ++ j){
			val[i + 1][j] = bin_op(val[i][j], val[i][min(N - 1, j + (1 << i))]);
		}
	}
	sparse_table(){ }
	T query(int l, int r){
		int d = 31 - __builtin_clz(r - l);
		return bin_op(val[d][l], val[d][r - (1 << d)]);
	}
	sparse_table &operator=(const sparse_table &otr){
		N = otr.N, bin_op = otr.bin_op;
		val = otr.val;
		return *this;
	}
};
struct LCA{
	vector<int> time;
	vector<long long> depth;
	int root;
	sparse<pair<int, int>> rmq;
	LCA(vector<vector<pair<int, int>>> &adj, int root): root(root), time(adj.size(), -99), depth(adj.size()), rmq(dfs(adj)){}
	vector<pair<int, int>> dfs(vector<vector<pair<int, int>>> &adj){
		vector<tuple<int, int, int, long long>> q(1);
		vector<pair<int, int>> res;
		int T = root;
		while(!q.empty()){
			auto [u, p, d, di] = q.back();
			q.pop_back();
			if(d) res.emplace_back(d, p);
			time[u] = T ++;
			depth[u] = di;
			for(auto &e: adj[u]) if(e.first != p){
				q.emplace_back(e.first, u, d + 1, di + e.second);
			}
		}
		return res;
	}
	int query(int l, int r){
		if(l == r) return l;
		l = time[l], r = time[r];
		return rmq.query(min(l, r), max(l, r)).second;
	}
	long long dist(int l, int r){
		int lca = query(l, r);
		return depth[l] + depth[r] - 2 * depth[lca];
	}
};

// 156485479_4_5_2_1
// Binary Lifting for Unweighted Tree
// O(N log N) preprocessing, O(log N) per lca query
struct binary_lift: vector<vector<int>>{
	int N, root, lg;
	vector<vector<int>> up;
	vector<int> depth;
	binary_lift(int N, int root): N(N), root(root), lg(ceil(log2(N))), depth(N), up(N, vector<int>(lg + 1)){
		this->resize(N);
	}
	void insert(int u, int v){ (*this)[u].push_back(v), (*this)[v].push_back(u); }
	void init(){ dfs(root, root); }
	void dfs(int u, int p){
		up[u][0] = p;
		for(int i = 1; i <= lg; ++ i) up[u][i] = up[up[u][i - 1]][i - 1];
		for(auto &v: (*this)[u]) if(v != p){
			depth[v] = depth[u] + 1;
			dfs(v, u);
		}
	}
	int lca(int u, int v){
		if(depth[u] < depth[v]) std::swap(u, v);
		u = trace_up(u, depth[u] - depth[v]);
		for(int d = lg; d >= 0; -- d) if(up[u][d] != up[v][d]) u = up[u][d], v = up[v][d];
		return u == v ? u : up[u][0];
	}
	int dist(int u, int v){
		return depth[u] + depth[v] - 2 * depth[lca(u, v)];
	}
	int trace_up(int u, int dist){
		if(dist >= depth[u] - depth[root]) return root;
		for(int d = lg; d >= 0; -- d) if(dist & (1 << d)) u = up[u][d];
		return u;
	}
};

// 156485479_4_5_2_2
// Binary Lifting for Weighted Tree Supporting Commutative Monoid Operations
// O(N log N) processing, O(log N) per query
template<typename T, typename BO>
struct binary_lift{
	int N, root, lg;
	BO bin_op;
	const T id;
	vector<T> val;
	vector<vector<pair<int, T>>> adj, up;
	vector<int> depth;
	binary_lift(int N, int root, const vector<T> &val, BO bin_op, T id): N(N), root(root), bin_op(bin_op), id(id), lg(32 - __builtin_clz(N)), depth(N), val(val), adj(N), up(N, vector<pair<int, T>>(lg + 1)){ }
	void insert(int u, int v, T w){
		adj[u].emplace_back(v, w);
		adj[v].emplace_back(u, w);
	}
	void init(){ dfs(root, root, id); }
	void dfs(int u, int p, T w){
		up[u][0] = {p, bin_op(val[u], w)};
		for(int i = 1; i <= lg; ++ i) up[u][i] = {
			up[up[u][i - 1].first][i - 1].first
			, bin_op(up[u][i - 1].second, up[up[u][i - 1].first][i - 1].second)
		};
		for(auto &[v, x]: adj[u]) if(v != p){
			depth[v] = depth[u] + 1;
			dfs(v, u, x);
		}
	}
	pair<int, T> trace_up(int u, int dist){ // Node, Distance (Does not include weight of the node)
		T res = id;
		dist = min(dist, depth[u] - depth[root]);
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
	int dist(int u, int v){
		return depth[u] + depth[v] - 2 * depth[query(u, v).first];
	}
};

// 156485479_4_5_3
// Heavy Light Decomposition
// O(N + M) processing, O(log^2 N) per query
template<typename T, typename LOP, typename QOP, typename AOP, typename INIT = function<T(int, int)>>
struct segment{
	segment *l = 0, *r = 0;
	int low, high;
	LOP lop;                // Lazy op(L, L -> L)
	QOP qop;                // Query op(Q, Q -> Q)
	AOP aop;                // Apply op(Q, L, leftend, rightend -> Q)
	const vector<T> id;     // Lazy id(L), Query id(Q), Disable constant(Q)
	T lset, lazy, val;
	INIT init;
	segment(int low, int high, LOP lop, QOP qop, AOP aop, const vector<T> &id, INIT init): low(low), high(high), lop(lop), qop(qop), aop(aop), id(id), init(init){
		lazy = id[0], lset = id[2], val = init(low, high);
	}
	segment(const vector<T> &arr, int low, int high, LOP lop, QOP qop, AOP aop, const vector<T> &id): low(low), high(high), lop(lop), qop(qop), aop(aop), id(id){
		lazy = id[0], lset = id[2];
		if(high - low > 1){
			int mid = low + (high - low) / 2;
			l = new segment(arr, low, mid, lop, qop, aop, id);
			r = new segment(arr, mid, high, lop, qop, aop, id);
			val = qop(l->val, r->val);
		}
		else val = arr[low];
	}
	void push(){
		if(!l){
			int mid = low + (high - low) / 2;
			l = new segment(low, mid, lop, qop, aop, id);
			r = new segment(mid, high, lop, qop, aop, id);
		}
		if(lset != id[2]){
			l->set(low, high, lset);
			r->set(low, high, lset);
			lset = id[2];
		}
		else if(lazy != id[0]){
			l->update(low, high, lazy);
			r->update(low, high, lazy);
			lazy = id[0];
		}
	}
	void set(int ql, int qr, T x){
		if(qr <= low || high <= ql) return;
		if(ql <= low && high <= qr){
			lset = x;
			lazy = id[0];
			val = aop(id[1], x, low, high);
		}
		else{
			push();
			l->set(ql, qr, x);
			r->set(ql, qr, x);
			val = qop(l->val, r->val);
		}
	}
	void update(int ql, int qr, T x){
		if(qr <= low || high <= ql) return;
		if(ql <= low && high <= qr){
			if(lset != id[2]) lset = lop(lset, x);
			else lazy = lop(lazy, x);
			val = aop(val, x, low, high);
		}
		else{
			push();
			l->update(ql, qr, x);
			r->update(ql, qr, x);
			val = qop(l->val, r->val);
		}
	}
	T query(int ql, int qr){
		if(qr <= low || high <= ql) return id[1];
		if(ql <= low && high <= qr) return val;
		push();
		return qop(l->query(ql, qr), r->query(ql, qr));
	}
	// Below assumes T is an ordered field and node stores positive values
	template<typename IO>
	int plb(T x, IO inv_op){
		if(low + 1 == high) return high;
		push();
		if(l->val < x) return r->plb(inv_op(x, l->val), inv_op);
		else return l->plb(x, inv_op);
	}
	template<typename IO>
	int lower_bound(T x, IO inv_op){ // min i such that query[0, i) >= x
		if(val < x) return high + 1;
		else return plb(x, inv_op);
	}
	template<typename IO>
	int lower_bound(int i, T x, IO inv_op){
		return lower_bound(qop(x, query(low, min(i, high))), inv_op);
	}
	template<typename IO>
	int pub(T x, IO inv_op){
		if(low + 1 == high) return high;
		push();
		if(x < l->val) return l->pub(x, inv_op);
		else return r->pub(inv_op(x, l->val), inv_op);
	}
	template<typename IO>
	int upper_bound(T x, IO inv_op){ // min i such that query[0, i) > val
		if(x < val) return pub(x, inv_op);
		else return high + 1;
	}
	template<typename IO>
	int upper_bound(int i, T x, IO inv_op){
		return upper_bound(qop(x, query(low, min(i, high))), inv_op);
	}
};
template<typename DS, typename BO, typename T, int VALS_IN_EDGES = 1>
struct heavy_light_decomposition{
	int N, root;
	vector<vector<int>> adj;
	vector<int> par, size, depth, next, pos, rpos;
	DS &tr;
	BO bin_op;
	const T id;
	heavy_light_decomposition(int N, int root, DS &tr, BO bin_op, T id): N(N), root(root), adj(N), par(N, -1), size(N, 1), depth(N), next(N), pos(N), tr(tr), bin_op(bin_op), id(id){
		this->root = next[root] = root;
	}
	void insert(int u, int v){
		adj[u].push_back(v);
		adj[v].push_back(u);
	}
	void dfs_sz(int u){
		if(par[u] != -1) adj[u].erase(find(adj[u].begin(), adj[u].end(), par[u]));
		for(auto &v: adj[u]){
			par[v] = u, depth[v] = depth[u] + 1;
			dfs_sz(v);
			size[u] += size[v];
			if(size[v] > size[adj[u][0]]) swap(v, adj[u][0]);
		}
	}
	void dfs_hld(int u){
		static int t = 0;
		pos[u] = t ++;
		rpos.push_back(u);
		for(auto &v: adj[u]){
			next[v] = (v == adj[u][0] ? next[u] : v);
			dfs_hld(v);
		}
	}
	void init(){
		dfs_sz(root), dfs_hld(root);
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
	void updatepath(int u, int v, T val, int is_update = true){
		if(is_update) processpath(u, v, [this, &val](int l, int r){tr.update(l, r, val);});
		else processpath(u, v, [this, &val](int l, int r){tr.set(l, r, val);});
	}
	void updatesubtree(int u, T val, int is_update = true){
		if(is_update) tr.update(pos[u] + VALS_IN_EDGES, pos[u] + size[u], val);
		else tr.set(pos[u] + VALS_IN_EDGES, pos[u] + size[u], val);
	}
	T querypath(int u, int v){
		T res = id;
		processpath(u, v, [this, &res](int l, int r){res = bin_op(res, tr.query(l, r));});
		return res;
	}
	T querysubtree(int u){
		return tr.query(pos[u] + VALS_IN_EDGES, pos[u] + size[u]);
	}
};

// 156485479_4_5_4
// Centroid Decomposition
// O(N log N) processing
struct centroid_decomposition{
	int N, root;
	vector<int> dead, size, par, cpar;
	vector<vector<int>> adj, cchild, dist;
	centroid_decomposition(int N): N(N), adj(N), dead(N), size(N), par(N), cchild(N), cpar(N), dist(N){ }
	void insert(int u, int v){
		adj[u].push_back(v);
		adj[v].push_back(u);
	}
	void dfs_sz(int u){
		size[u] = 1;
		for(auto v: adj[u]) if(!dead[v] && v != par[u]){
			par[v] = u;
			dfs_sz(v);
			size[u] += size[v];
		}
	}
	int centroid(int u){
		par[u] = -1;
		dfs_sz(u);
		int s = size[u];
		while(1){
			int w = 0, msz = 0;
			for(auto v: adj[u]) if(!dead[v] && v != par[u] && msz < size[v]){
				w = v, msz = size[v];
			}
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
		int d = 0;
		for(auto v: adj[u]) if(!dead[v]) dfs_dist(v, u);
		for(auto v: adj[u]) if(!dead[v]) dfs_centroid(v, u);
	}
	void init(){
		dfs_centroid(0, -1);
	}
};

// 156485479_4_5_5
// AHU Algorithm ( Rooted Tree Isomorphism ) / Tree Isomorphism
// O(n)
void radix_sort(vector<pair<int, vector<int>>> &arr){
	int n = int(arr.size()), mxval = 0, mxsz = 1 + accumulate(arr.begin(), arr.end(), 0, [](int x, const pair<int, vector<int>> &y){return max(x, y.second.size());});
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

// 156485479_4_6_1
// Shortest Path Tree On Sparse Graph ( Dijkstra, Bellman Ford, SPFA )
template<typename T = long long, typename BO = plus<T>, typename Compare = less<T>>
struct shortest_path_tree{
	struct edge{
		int from, to;
		T cost;
	};
	int N;
	BO bin_op;
	Compare cmp;
	const T inf, id;
	vector<vector<int>> adj;
	vector<edge> edge;
	vector<T> dist;
	vector<int> parent;
	shortest_path_tree(int N, const T inf = numeric_limits<T>::max() / 8, BO bin_op = plus<T>(), T id = 0, Compare cmp = less<T>()): N(N), inf(inf), bin_op(bin_op), id(id), cmp(cmp), adj(N){ }
	void insert(int u, int v, T w){
		adj[u].push_back(int(edge.size()));
		edge.push_back({u, v, w});
	}
	void init(){
		dist.resize(N), parent.resize(N);
		fill(dist.begin(), dist.end(), inf), fill(parent.begin(), parent.end(), -1);
	}
	void init_dijkstra(int s = 0){
		init();
		dist[s] = id;
		auto qcmp = [&](const pair<T, int> &lhs, const pair<T, int> &rhs){
			return lhs.first == rhs.first ? lhs.second < rhs.second : cmp(rhs.first, lhs.first);
		};
		priority_queue<pair<T, int>, vector<pair<T, int>>, decltype(qcmp)> q(qcmp);
		q.push({id, s});
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
	pair<vector<int>, vector<int>> init_bellman_ford(int s = 0, bool find_any_cycle = false){ // cycle {vertices, edges}
		if(find_any_cycle){
			fill(dist.begin(), dist.end(), id);
			fill(parent.begin(), parent.end(), -1);
		}
		else{
			init();
			dist[s] = id;
		}
		int x;
		for(int i = 0; i < N; ++ i){
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
			for(int i = 0; i < N; ++ i) y = parent[y];
			vector<int> vertices, edges;
			for(int c = y; ; c = edge[parent[c]].from){
				vertices.push_back(c), edges.push_back(parent[c]);
				if(c == y && vertices.size() > 1) break;
			}
			reverse(vertices.begin(), vertices.end()), reverse(edges.begin(), edges.end());
			return {vertices, edges};
		}
	}
	bool init_spfa(int s = 0){ // returns false if cycle
		init();
		vector<int> cnt(N);
		vector<bool> inq(N);
		deque<int> q;
		dist[s] = id;
		q.push_back(s);
		inq[s] = true;
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
						if(cnt[v] > N) return false;
					}
				}
			}
		}
		return true;
	}
	pair<vector<int>, vector<int>> path_from_root(int u){
		vector<int> vertices, edges;
		for(; parent[u] != -1; u = edge[parent[u]].from){
			vertices.push_back(u);
			edges.push_back(parent[u]);
		}
		vertices.push_back(u);
		reverse(vertices.begin(), vertices.end()), reverse(edges.begin(), edges.end());
		return {vertices, edges};
	}
};

// 156485479_4_6_2
// Shortest Path Tree On Dense Graph ( Dijkstra, Floyd Warshall )
template<typename T = long long, typename BO = plus<T>, typename Compare = less<T>>
struct shortest_path_tree_dense{
	int N;
	BO bin_op;
	Compare cmp;
	const T inf, id;
	vector<vector<T>> adj, dist;
	vector<vector<int>> parent, pass;
	shortest_path_tree_dense(int N, const T inf = numeric_limits<T>::max() / 8, BO bin_op = plus<T>(), T id = 0, Compare cmp = less<T>()): N(N), inf(inf), bin_op(bin_op), id(id), cmp(cmp), adj(N, vector<T>(N, inf)){ }
	void insert(int u, int v, T w){
		assert(u != v);
		if(cmp(w, adj[u][v])) adj[u][v] = w;
	}
	void clear(){
		for(int u = 0; u < N; ++ u) fill(adj[u].begin(), adj[u].end(), inf);
	}
	void init(int s){
		dist.resize(N), parent.resize(N), dist[s].resize(N), parent[s].resize(N);
		fill(dist[s].begin(), dist[s].end(), inf), fill(parent[s].begin(), parent[s].end(), -1);
	}
	void init_dijkstra(int s){
		init(s);
		vector<bool> visited(N);
		dist[s][s] = id;
		for(int i = 0; i < N; ++ i){
			int u = -1;
			for(int v = 0; v < N; ++ v) if(!visited[v] && (u == -1 || cmp(dist[s][v], dist[s][u]))) u = v;
			if(dist[s][u] == inf) break;
			visited[u] = true;
			for(int v = 0; v < N; ++ v) if(cmp(bin_op(dist[s][u], adj[u][v]), dist[s][v])){
				dist[s][v] = bin_op(dist[s][u], adj[u][v]);
				parent[s][v] = u;
			}
		}
	}
	vector<int> path_from_root(int s, int u){
		if(dist[s][u] >= inf) return { };
		vector<int> vertices;
		for(; u != s; u = parent[s][u]) vertices.push_back(u);
		vertices.push_back(s);
		reverse(vertices.begin(), vertices.end());
		return vertices;
	}
	void init_all_pair(){
		pass.resize(N, vector<int>(N)), dist = adj;
		for(int u = 0; u < N; ++ u) fill(pass[u].begin(), pass[u].end(), -1), dist[u][u] = id;
	}
	bool init_floyd_warshall(){
		for(int k = 0; k < N; ++ k) for(int i = 0; i < N; ++ i) for(int j = 0; j < N; ++ j){
			if(cmp(dist[i][k], inf) && cmp(dist[k][j], inf) && cmp(bin_op(dist[i][k], dist[k][j]), dist[i][j])){
				dist[i][j] = bin_op(dist[i][k], dist[k][j]);
				pass[i][j] = k;
			}
		}
		for(int u = 0; u < N; ++ u) if(dist[u][u] != id) return false;
		return true;
	}
	vector<int> path_between(int u, int v){
		if(dist[u][v] == inf) return { };
		vector<int> path;
		function<void(int, int)> solve = [&](int u, int v){
			if(pass[u][v] == -1){
				path.push_back(u);
				return;
			}
			solve(u, pass[u][v]), solve(pass[u][v], v);
		};
		solve(u, v);
		path.push_back(v);
		return path;
	}
};

// 156485479_4_7
// Minimum Spanning Forest
// O(M log N)
struct disjoint{
	int N;
	vector<int> parent, size;
	// vector<pair<int, int>> Log_parent, Log_size; // For persistency
	disjoint(int N): N(N), parent(N), size(N, 1){ iota(parent.begin(), parent.end(), 0); }
	void expand(){
		++ N;
		parent.push_back(parent.size());
		size.push_back(1);
	}
	int root(int u){
		// Log_parent.emplace_back(u, parent[u]);
		return parent[u] == u ? u : parent[u] = root(parent[u]);
	}
	bool merge(int u, int v){
		u = root(u), v = root(v);
		if(u == v) return false;
		if(size[u] < size[v]) swap(u, v);
		// Log_parent.emplace_back(v, parent[v]);
		parent[v] = u;
		// Log_size.emplace_back(u, size[u]);
		size[u] += size[v];
		return true;
	}
	bool share(int u, int v){
		return root(parent[u]) == root(parent[v]);
	}
	/*void reverse(int p, int s){
		while(parent.size() != p) reverse_parent();
		while(size.size() != s) reverse_size();
	}
	void reverse_parent(){
		auto [u, p] = Log_parent.back();
		Log_parent.pop_back();
		parent[u] = p;
	}
	void reverse_size(){
		auto [u, s] = Log_size.back();
		Log_size.pop_back();
		size[u] = s;
	}*/
};
template<typename T = long long>
struct minimum_spanning_forest{
	int N;
	vector<vector<pair<int, int>>> adj;
	vector<vector<int>> mstadj;
	vector<int> mstedge;
	vector<tuple<int, int, T>> edge;
	T cost = 0;
	minimum_spanning_forest(int N): N(N), adj(N), mstadj(N){ }
	void insert(int u, int v, T w){
		adj[u].emplace_back(v, edge.size()), adj[v].emplace_back(u, edge.size());
		edge.emplace_back(u, v, w);
	}
	void init_kruskal(){
		int M = int(edge.size());
		vector<int> t(M);
		iota(t.begin(), t.end(), 0);
		sort(t.begin(), t.end(), [&](int i, int j){ return get<2>(edge[i]) < get<2>(edge[j]); });
		disjoint dsu(N);
		for(auto i: t){
			auto [u, v, w] = edge[i];
			if(dsu.merge(u, v)){
				cost += w;
				mstedge.push_back(i);
				mstadj[u].push_back(v), mstadj[v].push_back(u);
			}
		}
	}
	void init_prim(){
		vector<bool> used(N);
		priority_queue<tuple<T, int, int, int>, vector<tuple<T, int, int, int>>, greater<tuple<T, int, int, int>>> q;
		for(int u = 0; u < N; ++ u) if(!used[u]){
			q.emplace(0, u, -1, -1);
			while(!q.empty()){
				auto [w, u, p, i] = q.top();
				q.pop();
				if(used[u]) continue;
				used[u] = true;
				if(p != -1){
					mstedge.push_back(i);
					mstadj[u].push_back(p), mstadj[p].push_back(u);
				}
				cost += w;
				for(auto [v, i]: adj[u]) if(!used[v]) q.emplace(get<2>(edge[i]), v, u, i);
			}
		}
	}
};
// For dense graph
// O(N^2)
template<typename T = long long>
struct minimum_spanning_forest_dense{
	static constexpr T inf = numeric_limits<T>::max();
	int N, edgecnt = 0;
	vector<vector<T>> adj;
	vector<vector<int>> adjL;
	vector<vector<bool>> mstadj;
	T cost = 0;
	minimum_spanning_forest_dense(int N): N(N), adj(N, vector<T>(N, inf)), adjL(N), mstadj(N, vector<bool>(N)){ }
	void insert(int u, int v, T w){
		adj[u][v] = adj[v][u] = w;
		adjL[u].push_back(v), adjL[v].push_back(u);
	}
	void init_prim(){
		vector<bool> used(N), reached(N);
		vector<int> reach;
		vector<tuple<T, int, int>> t(N, {inf, -1, 0});
		for(int u = 0; u < N; ++ u) if(!used[u]){
			function<void(int)> dfs = [&](int u){
				reached[u] = true;
				reach.push_back(u);
				for(auto v: adjL[u]) if(!reached[v]) dfs(v);
			};
			dfs(u);
			get<0>(t[reach[0]]) = 0;
			for(int tt = 0; tt < reach.size(); ++ tt){
				int u = -1;
				for(auto v: reach) if(!used[v] && (u == -1 || get<0>(t[v]) < get<0>(t[u]))) u = v;
				auto [w, p, _] = t[u];
				used[u] = true;
				cost += w;
				if(p != -1){
					mstadj[u][p] = mstadj[p][u] = true;
					++ edgecnt;
				}
				for(auto v: reach) if(adj[u][v] < get<0>(t[v])) t[v] = {adj[u][v], u, v};
			}
			reach.clear();
		}
	}
};

// 156485479_4_8
// Topological Sort / Returns false if there's a cycle
// O(V + E)
template<class Graph>
pair<bool, vector<int>> toposort(const Graph &adj){
	int n = int(adj.size());
	vector<int> indeg(n), res(n);
	for(int u = 0; u < n; ++ u) for(auto v: adj[u]) ++ indeg[v];
	deque<int> q;
	for(int u = 0; u < n; ++ u) if (!indeg[u]) q.push_back(u);
	int cnt = 0;
	while(q.size() > 0){
		int u = q.front();
		q.pop_front();
		res[u] = cnt ++;
		for(auto v: adj[u]) if (!(-- indeg[v])) q.push_back(v);
	}
	return cnt == n;
}
// Lexicographically Smallest Topological Sort / Return false if there's a cycle
// O(V log V + E)
template<class Graph>
pair<bool, vector<int>> toposort(const Graph &adj){
	int n = adj.size();
	vector<int> indeg(n), res(n);
	for(int u = 0; u < n; ++ u) for(auto v: adj[u]) ++ indeg[v];
	priority_queue<int, vector<int>, greater<int>> q;
	for(int u = 0; u < n; ++ u) if (!indeg[u]) q.push(u);
	int cnt = 0;
	while(q.size() > 0){
		int u = q.top();
		q.pop();
		res[u] = cnt ++;
		for(auto v: adj[u]) if (!(-- indeg[v])) q.push(v);
	}
	return cnt == n;
}

// 156485479_4_9
// Two Satisfiability / values hold the result
// O(V + E)
struct two_sat{
	int N;
	vector<vector<int>> adj;
	vector<int> value, val, comp, z;
	two_sat(int N = 0): N(N), adj(N << 1){ }
	int time = 0;
	int add_var(){
		adj.emplace_back();
		adj.emplace_back();
		return ++ N;
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
			either(cur, ~arr[u]);
			either(cur, next);
			either(~arr[u], next);
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
		value.assign(N, -1);
		val.assign(2 * N, 0);
		comp = val;
		for(int u = 0; u < N << 1; ++ u) if(!comp[u]) dfs(u);
		for(int u = 0; u < N; ++ u) if(comp[u << 1] == comp[u << 1 ^ 1]) return false;
		return true;
	}
};

// 156485479_5_1
// Returns the starting position of the lexicographically minimal rotation
// O(n)
int min_rotation(string s){
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
// All Palindromic Substrings ( Manachar's Algorithm )
// O(N)
struct manachar{
	int N;
	vector<int> o, e;
	pair<int, int> build(const string &s){
		N = int(s.size()), o.resize(N), e.resize(N);
		int res = 0, resl, resr;
		int l = 0, r = -1;
		for(int i = 0; i < N; ++ i){
			int k = (i > r) ? 1 : min(o[l + r - i], r - i) + 1;
			while(i - k >= 0 && i + k < N && s[i - k] == s[i + k]) ++ k;
			o[i] = -- k;
			if(res < 2 * k + 1){
				res = 2 * k + 1;
				resl = i - k, resr = i + k + 1;
			}
			if(r < i + k){
				l = i - k;
				r = i + k;
			}
		}
		l = 0; r = -1;
		for(int i = 0; i < N; ++ i){
			int k = (i > r) ? 1 : min(e[l + r - i + 1], r - i + 1) + 1;
			while(i - k >= 0 && i + k - 1 < N && s[i - k] == s[i + k - 1]) ++ k;
			e[i] = -- k;
			if(res < 2 * k){
				res = 2 * k;
				resl = i - k, resr = i + k;
			}
			if(r < i + k - 1){
				l = i - k;
				r = i + k - 1;
			}
		}
		return {resl, resr};
	}
};

// 156485479_5_3
// Suffix Array and Kasai's Algorithm
// O(N log N)
template<typename T, typename BO = function<T(T, T)>>
struct sparse_table{
	int N;
	BO bin_op;
	vector<vector<T>> val;
	sparse_table(const vector<T> &arr, BO bin_op = [](T x, T y){return min(x, y);}): N(arr.size()), bin_op(bin_op){
		int t = 1, d = 1;
		while(t < N) t <<= 1, ++ d;
		val.assign(d, arr);
		for(int i = 0; i < d - 1; ++ i) for(int j = 0; j < N; ++ j){
			val[i + 1][j] = bin_op(val[i][j], val[i][min(N - 1, j + (1 << i))]);
		}
	}
	sparse_table(){ }
	T query(int l, int r){
		int d = 31 - __builtin_clz(r - l);
		return bin_op(val[d][l], val[d][r - (1 << d)]);
	}
	sparse_table &operator=(const sparse_table &otr){
		N = otr.N, bin_op = otr.bin_op;
		val = otr.val;
		return *this;
	}
};
template<typename Str, int lim = 256>
struct suffix_array{
	int N;
	Str s;
	vector<int> p, c, l; // p[i]: starting index of i-th suffix in SA, c[i]: position of suffix of index i in SA
	sparse_table<int, function<int(int, int)>> rmq;
	suffix_array(const Str &s): N(s.size()), c(N), s(s){
		p = sort_cyclic_shifts(s + "$");
		p.erase(p.begin());
		for(int i = 0; i < N; ++ i) c[p[i]] = i;
		l = get_lcp(p);
		rmq = sparse_table<int, function<int(int, int)>>(l);
	}
	vector<int> sort_cyclic_shifts(const Str &s){
		int n = int(s.size());
		vector<int> p(n), c(n), cnt(max(lim, n));
		for(auto x: s) ++ cnt[x];
		for(int i = 1; i < lim; ++ i) cnt[i] += cnt[i - 1];
		for(int i = 0; i < n; ++ i) p[-- cnt[s[i]]] = i;
		int classes = 1;
		for(int i = 1; i < n; ++ i){
			if(s[p[i]] != s[p[i - 1]]) classes ++;
			c[p[i]] = classes - 1;
		}
		vector<int> pn(n), cn(n);
		for(int h = 0; (1 << h) < n; ++ h){
			for(int i = 0; i < n; ++ i){
				pn[i] = p[i] - (1 << h);
				if(pn[i] < 0) pn[i] += n;
			}
			fill(cnt.begin(), cnt.begin() + classes, 0);
			for(auto x: pn) ++ cnt[c[x]];
			for(int i = 1; i < classes; ++ i) cnt[i] += cnt[i - 1];
			for(int i = n - 1; i >= 0; -- i) p[-- cnt[c[pn[i]]]] = pn[i];
			cn[p[0]] = 0, classes = 1;
			for(int i = 1; i < n; ++ i){
				if(c[p[i]] != c[p[i - 1]] || c[(p[i] + (1 << h)) % n] != c[(p[i - 1] + (1 << h)) % n]){
					++ classes;
				}
				cn[p[i]] = classes - 1;
			}
			c.swap(cn);
		}
		return p;
	}
	vector<int> get_lcp(const vector<int> &p){
		int n = int(s.size());
		vector<int> rank(n);
		for(int i = 0; i < n; ++ i) rank[p[i]] = i;
		int k = 0;
		vector<int> l(n - 1);
		for(int i = 0; i < n; ++ i){
			if(rank[i] == n - 1){
				k = 0;
				continue;
			}
			int j = p[rank[i] + 1];
			while(i + k < n && j + k < n && s[i + k] == s[j + k]) ++ k;
			l[rank[i]] = k;
			if(k) -- k;
		}
		return l;
	}
	int lcp(int i, int j){
		return rmq.query(min(c[i], c[j]), max(c[i], c[j]));
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
// Aho Corasic Algorithm
// O(W) preprocessing, O(L) per query
template<int C>
struct aho_corasic{
	struct node{
		int par, link = -1, elink = -1;
		char cpar;
		vector<int> next, go;
		bool isleaf = false;
		int ind;
		node(int par = -1, char pch = '$'): par(par), cpar(pch), next(C, -1), go(C, -1){}
	};
	vector<node> arr;
	function<int(char)> trans;
	aho_corasic(function<int(char)> trans = [](char c){return c < 'Z' ? c - 'A' : c - 'a';}): arr(1), trans(trans){}
	void insert(int ind, const string &s){
		int u = 0;
		for(auto &ch: s){
			int c = trans(ch);
			if(arr[u].next[c] == -1){
				arr[u].next[c] = arr.size();
				arr.emplace_back(u, ch);
			}
			u = arr[u].next[c];
		}
		arr[u].isleaf = true;
		arr[u].ind = ind;
	}
	int get_link(int u){
		if(arr[u].link == -1){
			if(!u || !arr[u].par) arr[u].link = 0;
			else arr[u].link = go(get_link(arr[u].par), arr[u].cpar);
		}
		return arr[u].link;
	}
	int get_elink(int u){
		if(arr[u].elink == -1){
			if(!u || !get_link(u)) arr[u].elink = 0;
			else if(arr[get_link(u)].isleaf) arr[u].elink = get_link(u);
			else arr[u].elink = get_elink(get_link(u));
		}
		return arr[u].elink;
	}
	int go(int u, char ch){
		int c = trans(ch);
		if(arr[u].go[c] == -1){
			if(arr[u].next[c] != -1) arr[u].go[c] = arr[u].next[c];
			else arr[u].go[c] = u == 0 ? 0 : go(get_link(u), ch);
		}
		return arr[u].go[c];
	}
	void print(int u, string s = ""){
		cout << "Node " << u << ": par = " << arr[u].par << ", cpar = " << arr[u].cpar << ", string: " << s << "\n";
		for(int i = 0; i < C; ++ i){
			if(arr[u].next[i] != -1){
				cout << u << " => ";
				print(arr[u].next[i], s + string(1, i + 'a'));
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
template<typename Str, typename UO = function<char(int)>, int lim = 26>
pair<vector<int>, vector<vector<int>>> prefix_automaton(const Str &s, UO trans = [](int c){return c + 'a';}){
	vector<int> p = prefix_function(s);
	int n = int(s.size());
	vector<vector<int>> aut(n, vector<int>(lim));
	for(int i = 0; i < n; ++ i) for(int c = 0; c < lim; ++ c){
		if(i > 0 && trans(c) != s[i]) aut[i][c] = aut[p[i - 1]][c];
		else aut[i][c] = i + (trans(c) == s[i]);
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
			int mid = low + high >> 1;
			query(posi, posi + mid, i) == query(posj, posj + mid, j) ? low = mid : high = mid;
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
	struct node{
		int len = 0, link = -1, firstpos = -1;
		bool isclone = false;
		map<typename Str::value_type, int> next;
		vector<int> invlink;
		int cnt = -1;
	};
	vector<node> state = vector<node>(1);
	int last = 0;
	suffix_automaton(const Str &s){
		state.reserve(s.size());
		for(auto c: s) insert(c);
	}
	void insert(typename Str::value_type c){
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
		for(int u = 0; length --; ){
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
	
// 156485479_5_10
// Palindrome Tree

// 156485479_6_1
// 2D Geometry Classes
template<typename T = long long> struct point{
	T x, y;
	template<typename U> point(const point<U> &otr): x(otr.x), y(otr.y){ }
	template<typename U, typename V> point(const pair<U, V> &p): x(p.first), y(p.second){ }
	template<typename U = T, typename V = T> point(U x = 0, V y = 0): x(x), y(y){ }
	template<typename U> explicit operator point<U>() const{ return point<U>(static_cast<U>(x), static_cast<U>(y)); }
	T operator*(const point &otr) const{ return x * otr.x + y * otr.y; }
	T operator^(const point &otr) const{ return x * otr.y - y * otr.x; }
	point operator+(const point &otr) const{ return point(x + otr.x, y + otr.y); }
	point &operator+=(const point &otr){ return *this = *this + otr; }
	point operator-(const point &otr) const{ return point(x - otr.x, y - otr.y); }
	point &operator-=(const point &otr){ return *this = *this - otr; }
	point operator*(const T &c) const{ return point(x * c, y * c); }
	point &operator*=(const T &c) { return *this = *this * c; }
	point operator/(const T &c) const{ return point(x / c, y / c); }
	point &operator/=(const T &c) { return *this = *this / c; }
	point operator-() const{ return point(-x, -y); }
	bool operator<(const point &otr) const{ return tie(x, y) < tie(otr.x, otr.y); }
	bool operator>(const point &otr) const{ return tie(x, y) > tie(otr.x, otr.y); }
	bool operator<=(const point &otr) const{ return tie(x, y) <= tie(otr.x, otr.y); }
	bool operator>=(const point &otr) const{ return tie(x, y) >= tie(otr.x, otr.y); }
	bool operator==(const point &otr) const{ return tie(x, y) == tie(otr.x, otr.y); }
	bool operator!=(const point &otr) const{ return tie(x, y) != tie(otr.x, otr.y); }
	double norm() const{ return sqrt(x * x + y * y); }
	T sqnorm() const{ return x * x + y * y; }
	double arg() const{ return atan2(y, x); } // [-pi, pi]
	point<double> unit() const{ return point<double>(x, y) / norm(); }
	point perp() const{ return point(-y, x); }
	point<double> normal() const{ return perp().unit(); }
	point<double> rotate(const double &theta) const{ return point<double>(x * cos(theta) - y * sin(theta), x * sin(theta) + y * cos(theta)); }
	point reflect_x() const{ return point(x, -y); }
	point reflect_y() const{ return point(-x, y); }
};
template<typename T> point<T> operator*(const T &c, const point<T> &p){ return point<T>(c * p.x, c * p.y); }
template<typename T> istream &operator>>(istream &in, point<T> &p){ return in >> p.x >> p.y; }
template<typename T> ostream &operator<<(ostream &out, const point<T> &p){ return out << pair<T, T>(p.x, p.y); }
template<typename T> double distance(const point<T> &p, const point<T> &q){ return (p - q).norm(); }
template<typename T> T squared_distance(const point<T> &p, const point<T> &q){ return (p - q).sqnorm(); }
template<typename T, typename U, typename V> T ori(const point<T> &p, const point<U> &q, const point<V> &r){ return (q - p) ^ (r - p); }
template<typename T> T doubled_signed_area(const vector<point<T>> &arr){
	T s = arr.back() ^ arr.front();
	for(int i = 1; i < arr.size(); ++ i) s += arr[i - 1] ^ arr[i];
	return s;
}
template<typename T = long long> struct line{
	point<T> p, d; // p + d*t
	template<typename U = T, typename V = T> line(point<U> p = {0, 0}, point<V> q = {0, 0}, bool Two_Points = true): p(p), d(Two_Points ? q - p : q){ }
	template<typename U> line(point<U> d): p(), d(static_cast<point<T>>(d)){ }
	line(T a, T b, T c): p(a ? -c / a : 0, !a && b ? -c / b : 0), d(-b, a){ }
	template<typename U> explicit operator line<U>() const{ return line<U>(point<U>(p), point<U>(d), false); }
	point<T> q() const{ return p + d; }
	bool degen() const{ return d == point<T>(); }
	tuple<T, T, T> coef(){ return {d.y, -d.x, d.perp() * p}; } // d.y (X - p.x) - d.x (Y - p.y) = 0
};
template<typename T> bool parallel(const line<T> &L, const line<T> &M){ return !(L.d ^ M.d); }
template<typename T> bool on_line(const point<T> &p, const line<T> &L){
	if(L.degen()) return p == L.p;
	return (p - L.p) ^ L.d == 0;
}
template<typename T> bool on_ray(const point<T> &p, const line<T> &L){
	if(L.degen()) return p == L.p;
	auto a = L.p - p, b = L.q() - p;
	return !(a ^ b) && a * L.d <= 0;
}
template<typename T> bool on_segment(const point<T> &p, const line<T> &L){
	if(L.degen()) return p == L.p;
	auto a = L.p - p, b = L.q() - p;
	return !(a ^ b) && a * b <= 0;
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
template<typename T> pair<bool, point<double>> intersect_closed_segments_no_parallel_overlap(const line<T> &L, const line<T> &M) {
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
	if(parallel(L, M)){
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
template<typename It, typename P> void sort_by_angle(It first, It last, const P &origin){
	first = partition(first, last, [&origin](const decltype(*first) &point){ return point == origin; });
	auto pivot = partition(first, last, [&origin](const decltype(*first) &point) { return point > origin; });
	compare_by_angle<P> cmp(origin);
	sort(first, pivot, cmp), sort(pivot, last, cmp);
}

// 156485479_6_2
// Convex Hull and Minkowski Sum
// O(n log n) construction, O(n) if sorted.
template<typename Polygon>
struct convex_hull: pair<Polygon, Polygon>{ // (Lower, Upper) type {0: both, 1: lower, 2: upper}
	int type;
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
	convex_hull operator^(const convex_hull &otr) const{
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
	convex_hull operator+(const convex_hull &otr) const{
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

// 156485479_7_1
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
*/

// 156485479_7_2
// Bump Allocator
static char BUFF[450 << 20];
void *operator new(size_t s){
	static size_t i = sizeof BUFF;
	assert(s < i);
	return (void *)&BUFF[i -= s];
}
void operator delete(void *){}