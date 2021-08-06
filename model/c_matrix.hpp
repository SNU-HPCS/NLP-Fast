#ifndef C_MATRIX_HPP
#define C_MATRIX_HPP

#include <cassert>
#include <fstream>
#include <cmath>
#include <random>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>
#include <string>
#include <mkl.h>

//
// Declaration
//

enum DTYPE {INT16, INT32, INT64, FLOAT32, FLOAT64};
enum ETYPE {INT, FLOAT};
static unsigned int CALC_SIZE[] = {2, 4, 8, 4, 8};
static int TYPE[] = {INT, INT, INT, FLOAT, FLOAT};

// Functions for matrix creation
template <typename T>
T* mat_z(long m, long n);
template <typename T>
T* mat_rand(long m, long n, T min_val = -1, T max_val = 1);

//
// Implementation
//
template <typename T>
T* mat_z(const long m, const long n) {
	auto nelem = m * n;
	if (nelem == 0)
		return nullptr;

	if (nelem / n != m){
		printf("<assert> m=%ld, n=%ld\n", m, n);
		assert(nelem > 0);
	}

	T *data = (T*)mkl_malloc(nelem * sizeof(T), 64);
	memset((char*)data, 0, sizeof(T) * nelem);
	return data;
}

std::default_random_engine generator;
template <typename T>
T* mat_rand(const long m, const long n, const T min_val, const T max_val) {
	// Temporal
	auto nelem = m * n;
	if (nelem == 0)
		return nullptr;

	if (nelem / n != m){
		printf("<assert> m=%ld, n=%ld\n", m, n);
		assert(nelem > 0);
	}

	T *data = (T*)mkl_malloc(nelem * sizeof(T), 64);
	//memset((char*)data, 0, sizeof(T) * nelem);

	if (std::is_integral<T>::value) {
		std::uniform_int_distribution<int> distribution(min_val, max_val - 1);
		for (long i = 0; i < nelem; ++i) {
			data[i] = static_cast<T>(distribution(generator));
		}
	} else {
		// Float
		std::uniform_real_distribution<float> distribution(min_val, max_val);
		for (long i = 0; i < nelem; ++i) {
			data[i] = static_cast<T>(distribution(generator));
		}
	}
	return data;
}

template<typename T>
void dump_array(const T* M, long x, long y) {
	for (long i = 0; i < x; ++i){
		for (long j = 0; j < y; ++j){
			printf("%.2e ", M[i + j * x]);
		}
		printf("\n");
	}
}

template<typename T>
void dump_array_T_cfmt(std::ostream& os, const char* array_name,
                       const float* M, long x, long y);

template<> inline
void dump_array_T_cfmt<float>(std::ostream& os, const char* array_name,
		const float* M, long x, long y){
	std::string tab = "    ";
	std::string dtab = tab + tab;
	std::string ttab = dtab + tab;

	os << "float " << array_name << "[" << y << "][" << x << "] =\n"
	   << "{\n";
	int sep = 4;
	int k = 0;
	for (long i = 0; i < y; ++i){
		os << tab << "{\n";
		k = 0;
		for (long j = 0; j < x; ++j){
			if (k == 0)
				os << dtab;
			os << M[x * i + j] << ',';
			k += 1;
			if (k == sep){
				os << '\n';
				k = 0;
			}
		}
		os << tab << "},\n";
	}
	os << "};\n";
}

template<> inline
void dump_array<int16_t>(const int16_t* M, long x, long y) {
	for (long i = 0; i < x; ++i){
		for (long j = 0; j < y; ++j){
			printf("%d ", M[i + j * x]);
		}
		printf("\n");
	}
}
#endif
