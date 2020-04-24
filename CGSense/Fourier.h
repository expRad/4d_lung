#pragma once
#include <string>
#include <vector>
#include "C:/dev/FFTW_LIBS/fftw3.h"
#include <iostream>
template<class T>
using mat3d = std::vector< std::vector< std::vector<T> > >;

template<class T>
using mat2d = std::vector< std::vector<T> >;

class Fourier
{
public:
	Fourier(const int& x, const int& y, const int& z) :
		sizeX(x),
		sizeY(y),
		sizeZ(z)
	{
	}
	~Fourier()
	{
		fftwf_destroy_plan(planFWD);
		fftwf_destroy_plan(planBWD);
	}
	void initFT();

	// in place FFT
	void transformFWD(fftwf_complex* a) const;
	void transformBWD(fftwf_complex* a, bool rescale = true) const;

	// out of place FFT
	void transformFWD(fftwf_complex* a, fftwf_complex* b) const;
	void transformBWD(fftwf_complex* a, fftwf_complex* b) const;

	void setSizeX(const int& x) { sizeX = x; }
	void setSizeY(const int& y) { sizeY = y; }
	void setSizeZ(const int& z) { sizeZ = z; }
	void setWisdomFileName(const std::string& s) { wisdomFileName = s; }

	fftwf_plan planFWD;
	fftwf_plan planBWD;

private:
	int sizeX;
	int sizeY;
	int sizeZ;

	std::string wisdomFileName;

	void signAlternate3D(fftwf_complex* a) const;
	inline int indexTransform(const int& i, const int& j, const int& k, const int& ni, const int& nj, const int& nk) const;

};
