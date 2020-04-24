#pragma once
#include <string>
#include <vector>
#include "C:/dev/FFTW_LIBS/fftw3.h"
#include <iostream>
#include "Fourier.h"
#include "Trajectory.h"
//#include "globalDefs.h"
template<class T>
using mat2d = std::vector< std::vector<T> >;


class Reco
{
public:
	enum class Density
	{
		NONE,
		VORONOI3D,
		IMPORT,
		UNDEFINED
	};
	Reco();
	~Reco();

	class ConvKernel
	{
	public:		
		ConvKernel(Reco* r) :
			sizeX(r->gridSizeX),
			sizeY(r->gridSizeY),
			sizeZ(r->gridSizeZ),
			reco(r),
			kernelFT(NULL)
		{
		}
		~ConvKernel()
		{
			if (kernelFT != NULL)
			{
				fftwf_free(kernelFT);
			}
		}
		
		void calculateWeights();
		void calculateKernelFT();
		void clearWeights();
		void excludeMissingData();		

		float getB() const { return b; }
		float getW() const { return w; }
		int getKernelOrder() const { return kernelOrder; }

		void setB(const float& b1) { b = b1; }
		void setW(const float& w1) { w = w1; }
		void setKernelOrder(const int& a) { kernelOrder = a; }

		mat2d<float> weights;
		mat2d<int> indices;
		fftwf_complex* kernelFT;

	private:
		Reco* reco;
		int sizeX;
		int sizeY;
		int sizeZ;
		float b;
		float w;
		int kernelOrder;

		inline void collectNeighborIndices(int* startPoint, std::vector<int>& convPoints, const mat2d<int>& kernelPoints) const;
		template<typename T>
		inline float convFunction(T* dist);

	};		

	ConvKernel* kernel;
	Trajectory* trajectory;
	Fourier* ft;

	fftwf_complex* data;
	fftwf_complex* dataBuffer;

	std::vector<fftwf_complex*> sMaps;

	std::string dataPath;
	std::vector<std::string> batchFiles;

	std::vector<float> denCorr;

	std::string dataPathRoot;
	std::string dataPathFileName;
	std::string dataPathFileExtension;
	std::vector<bool> emptyPoints;

	std::vector<int> numPerformedCGIterations;
	std::vector<double> cgResiduals;
	std::vector<fftwf_complex*> img;
	fftwf_complex* cgSenseImg;

	std::vector<int> imgRange;
	

	void check();
	void init();
	void finalize();
	bool importParameters(const std::string& path);
	bool createLogFile(const std::string& path);
	bool setDataInputPaths();

	void cgSenseDynamic();

	static inline int indexTransform(const int& i, const int& j, const int& k, const int& ni, const int& nj, const int& nk);
	static std::vector<int> indexBackTransform(const int& ind, const int& nx, const int& ny, const int& nz);
	static inline void indexBackTransformPt(int* ret, const int& ind, const int& nx, const int& ny, const int& nz);
	
	void setTrajectory(WaveCaipi* t);
	void setSampling(const int& x, const int& y, const int& z);
	void setSX(const int& x) { sX = x; }
	void setSY(const int& y) { sY = y; }
	void setSZ(const int& z) { sZ = z; }
	void setNCh(const int& n) { nCh = n; }
	void setGridSize(const int& x, const int& y, const int& z);
	void setGridSizeX(const int& x) { gridSizeX = x; }
	void setGridSizeY(const int& y) { gridSizeY = y; }
	void setGridSizeZ(const int& z) { gridSizeZ = z; }
	void setOffcenter(const int& x, const int& y, const int& z);
	void setOffcenterX(const int& x) { offcenterX = x; }
	void setOffcenterY(const int& y) { offcenterY = y; }
	void setOffcenterZ(const int& z) { offcenterZ = z; }
	void setFovY(const double& d) { fovY = d; }
	void setFovZ(const double& d) { fovZ = d; }
	void setCGThreshold(const double& x) { cgThreshold = x; }
	void setBatch(const int& b) { batch = b; }
	void setDensityCorrection(const Density& d) { density = d; }
	void setDensityCorrectionPath(const std::string& s) { densityCorrPath = s; }
	void setOutputParentDir(const std::string& s) { outputParentDir = s; }
	void setNumThreads(const int& n) { numThreads = n; }

	int getSX() const { return sX; }
	int getSY() const { return sY; }
	int getSZ() const { return sZ; }
	int getNCh() const { return nCh; }
	int getGridSizeX() const { return gridSizeX; }
	int getGridSizeY() const { return gridSizeY; }
	int getGridSizeZ() const { return gridSizeZ; }
	int getOffcenterX() const { return offcenterX; }
	int getOffcenterY() const { return offcenterY; }
	int getOffcenterZ() const { return offcenterZ; }
	int getBatch() const { return batch; }
	double getFovY() const { return fovY; }
	double getFovZ() const { return fovZ; }
	double getCGThreshold() const { return cgThreshold; }
	int getTotalGridSize() const { return totalGridSize; }
	int getNumThreads() const { return numThreads; }
	int getNumThreadsFFTW() const { return numThreadsFFTW; }
	Density getDensityCorrection() const { return density; }
	std::string getDensityCorrectionPath() const { return densityCorrPath; }
	std::string getOutputParentDir() const { return outputParentDir; }

	void allocImg();
	void allocData(fftwf_complex** d);
	void allocSMaps();

	void freeImg();

	void nufftFWD();
	void nufftBWD();
	
private:
	void reduceFOV(fftwf_complex* v);
	void clearFFTWComplex(fftwf_complex* q, const int& gridSize, const int& nThreads = 1);

	void elementProduct(fftwf_complex* result, fftwf_complex* image, fftwf_complex* senseMap, const int& cGridSize, const int& nThreads = 1);
	void mutliplyBySenseMap(fftwf_complex* chImage, const std::vector<double>& senseMap);
	void mutliplyByConjSenseMap(fftwf_complex* chImage, fftwf_complex* senseMap, const int& cGridSize, const int& nThreads = 1);
	void addToSumOfImages(fftwf_complex* sumOfImages, fftwf_complex* img, const int& gridSize);
	std::vector<double> conjScalarProduct(const fftwf_complex* x, const fftwf_complex* y, const int& arrSize, const int& nThreads = 1);
	std::vector<double> complexDivision(std::vector<double> x, std::vector<double> y);
	void addToArrayScaled(fftwf_complex* v1, const std::vector<double>& scale, fftwf_complex* v2, const int& arrSize, const int nThreads = 1);
	void addArraysScaled(fftwf_complex* p, const std::vector<double>& scale, fftwf_complex* r, const int& arrSize, const int nThreads = 1);
	int sX;
	int sY;
	int sZ;
	int nCh;
	int batch;

	double fovY;
	double fovZ;

	int gridSizeX;
	int gridSizeY;
	int gridSizeZ;
	double cgThreshold = -1;
	int totalGridSize;	

	int offcenterX;
	int offcenterY;
	int offcenterZ;

	int numThreads;
	int numThreadsFFTW;

	Density density;

	std::string outputParentDir;
	std::string densityCorrPath;

	friend class Trajectory;
	friend class WaveCaipi;
};



template<typename T>
inline float Reco::ConvKernel::convFunction(T* dist)
{
	const static float dkx = 1.0 / (float)(sizeX - 1);
	const static float dky = 1.0 / (float)(sizeY - 1);
	const static float dkz = 1.0 / (float)(sizeZ - 1);
	const static float wInv = 1.0 / float(w);
	const static float wHalf = w / 2.0;
	const static float FourTimesOneOverW2 = 4.0 / float(w*w);

	const float dx = dist[0];
	const float dy = dist[1];
	const float dz = dist[2];

	if ((dx > wHalf*dkx) || (dy > wHalf * dky) || (dz > wHalf * dkz))
	{
		return 0;
	}
	else
	{
		return ((boost::math::cyl_bessel_i(0, (b)*sqrt(1 - pow(2 * dx / float(w * dkx), 2))) / float(100 * w * dkx)) *
			(boost::math::cyl_bessel_i(0, (b)*sqrt(1 - pow(2 * dy / float(w * dky), 2))) / float(100 * w * dky)) *
			(boost::math::cyl_bessel_i(0, (b)*sqrt(1 - pow(2 * dz / float(w * dkz), 2))) / float(100 * w * dkz)));
	}
}
