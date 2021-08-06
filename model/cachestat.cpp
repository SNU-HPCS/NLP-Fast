#include "cachestat.hpp"
#include <cstdio>

const char* CacheStat::STR_ENUM_CS_DATA[CacheStat::NUM_CS_DATA] =
		{
				_ENUM_CS_DATA(GEN_STR)
		};


CacheStat operator+(const CacheStat& lhs, const CacheStat& rhs)
{
	CacheStat temp = lhs;
	temp += rhs;
	return temp;
}

CacheStat operator-(const CacheStat& lhs, const CacheStat& rhs)
{
	CacheStat temp = lhs;
	temp -= rhs;
	return temp;
}

void print_cs(const CacheStat& cs){
	for (int i = 0; i < CacheStat::NUM_CS_DATA; ++i)
		printf("%s %ld\n", CacheStat::STR_ENUM_CS_DATA[i], cs.data[i].value);
}
