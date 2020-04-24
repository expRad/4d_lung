#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "fftw3.h"
#include "Reco.h"

#define _CRT_SECURE_NO_WARNINGS

class IO
{
public:
	IO();
	~IO();

	//template<typename T>
	static void exportVectorBin(const std::vector<float>& v, std::string path);
	static void exportFFTWComplexBin(fftwf_complex* w, const std::string& path, const int& N);
		

	static void importSensitivityMaps(std::vector<fftwf_complex*>& maps, const int N, const int numCh, std::string path);
	static bool importDataBin(const Reco& r, fftwf_complex* v, const std::string& path);	

	static bool importTrajectoryBin(mat2d<double>& traj, const Reco& r);

	static std::vector<float> importDensityCorr(const Reco& r);
	static std::vector<float> importDensityCorr(const std::string& path, const int& N);

};


