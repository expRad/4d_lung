#pragma once
#include <string>
#include <vector>
#include "C:/dev/FFTW_LIBS/fftw3.h"
#include <iostream>
template<class T>
using mat3d = std::vector< std::vector< std::vector<T> > >;

template<class T>
using mat2d = std::vector< std::vector<T> >;
/*
A custom trajectory class should inherit from the Trajectory class.
Further, the function Reco::setTrajectory needs to be overloaded, taking an argument of the custom type.
*/
class Reco;

class Trajectory
{
public:
	Trajectory(Reco* reco)
	{
		r = reco;
	}
	virtual ~Trajectory()
	{

	}
	void calculateKMax();
	void setImportPath(const std::string& s) { importPath = s; }
	mat2d<double> getKMax() { return kmax; }
	virtual void calculateTraj() = 0;
	mat2d<double> kSpaceTraj;

protected:
	Reco * r;
	std::vector< std::vector<double> > kmax;

private:
	std::string importPath;
};

class WaveCaipi : public Trajectory
{
public:
	WaveCaipi(Reco* reco) :
		Trajectory(reco)
	{

	}
	void setNOsc(const int& n) { nOsc = n; }
	void setGradAmp(const double& a) { gradAmp = a; }
	void setAmpCorrY(const double& y) { ampCorrY = y; }
	void setAmpCorrZ(const double& z) { ampCorrZ = z; }
	void setShiftY(const double& y) { shiftY = y; }
	void setShiftZ(const double& z) { shiftZ = z; }
	void setBandwidth(const double& b) { bandwidth = b; }
	void setDwellTime(const double& d) { dwellTime = d; }
	void setPseudoGradAmp(const double& a) { pseudoGradAmp = a; }

	int getNOsc() const { return nOsc; }
	double getGradAmp() const { return gradAmp; }
	double getAmpCorrY() const { return ampCorrY; }
	double getAmpCorrZ() const { return ampCorrZ; }
	double getShiftY() const { return shiftY; }
	double getShiftZ() const { return shiftZ; }
	double getBandwidth() const { return bandwidth; }
	double getDwellTime() const { return dwellTime; }

	virtual void calculateTraj();

private:
	int nOsc;
	double ampCorrY;
	double ampCorrZ;
	double gradAmp;
	double pseudoGradAmp;
	double bandwidth;
	double shiftY;
	double shiftZ;
	double dwellTime;
};