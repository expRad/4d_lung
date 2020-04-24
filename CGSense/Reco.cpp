#include "Reco.h"
#include "globalVars.h"
#include <iostream>
#include <fstream>
#define _CRT_SECURE_NO_WARNINGS
#define BOOST_CONFIG_SUPPRESS_OUTDATED_MESSAGE
#include <boost/math/special_functions/bessel.hpp>
#include <omp.h>

Reco::Reco() :
	offcenterX(0),
	offcenterY(0),
	offcenterZ(0),
	batch(0),
	dataPathFileExtension(".bin"),
	numThreads(32),
	numThreadsFFTW(1),
	ft(NULL),
	kernel(NULL),
	trajectory(NULL),
	data(NULL),
	dataBuffer(NULL),
	density(Density::NONE),
	cgSenseImg(NULL)
{
	
}

Reco::~Reco()
{
	if (kernel != NULL)
	{
		delete kernel;
	}

	if (ft != NULL)
	{
		delete ft;
	}

	if (data != NULL)
	{
		fftwf_free(data);
	}

	if (dataBuffer != NULL)
	{
		fftwf_free(dataBuffer);
	}
	for (int ch = 0; ch < sMaps.size(); ch++) // Heap wurde beschädigt
	{
		if (sMaps[ch] != NULL)
		{
			fftwf_free(sMaps[ch]);
		}
	}
	if (cgSenseImg != NULL)
	{
		fftwf_free(cgSenseImg);
	}
}

bool Reco::setDataInputPaths()
{
	for (int i = 0; i < batch; i++)
	{
		batchFiles.push_back(dataPathRoot + dataPathFileName + std::to_string(i) + dataPathFileExtension);
	}

	return true;
}


void Reco::clearFFTWComplex(fftwf_complex* q, const int& size, const int& nThreads)
{
#	pragma omp parallel for num_threads(nThreads) schedule(static) default(shared)
	for (int i = 0; i < size; i++)
	{
		q[i][0] = 0;
		q[i][1] = 0;
	}
}


void Reco::elementProduct(fftwf_complex* result, fftwf_complex* image, fftwf_complex* senseMap, const int& cGridSize, const int& nThreads)
{
#pragma omp parallel for num_threads(nThreads) schedule(static) default(shared)
	for (int i = 0; i < cGridSize; i++)
	{
		result[i][0] = image[i][0] * senseMap[i][0] - image[i][1] * senseMap[i][1];	// re1*re2 - im1*im2
		result[i][1] = image[i][0] * senseMap[i][1] + image[i][1] * senseMap[i][0];	// re1*im2 + im1*re2
	}
}


void Reco::mutliplyBySenseMap(fftwf_complex* chImage, const std::vector<double>& senseMap)
{
	for (int i = 0; i < senseMap.size(); i += 2)
	{
		double reImage = chImage[i / 2][0];
		double imImage = chImage[i / 2][1];
		chImage[i / 2][0] = reImage * senseMap[i] - imImage * senseMap[i + 1];			// re1*re2 - im1*im2
		chImage[i / 2][1] = reImage * senseMap[i + 1] + imImage * senseMap[i];			// re1*im2 + im1*re2
	}
}

void Reco::mutliplyByConjSenseMap(fftwf_complex* chImage, fftwf_complex* senseMap, const int& cGridSize, const int& nThreads)
{
#pragma omp parallel for num_threads(nThreads) schedule(static) default(shared)
	for (int i = 0; i < cGridSize; i++)
	{
		double reImage = chImage[i][0];
		double imImage = chImage[i][1];
		chImage[i][0] = reImage * senseMap[i][0] + imImage * senseMap[i][1];			// re1*re2 - im1*im2
		chImage[i][1] = -reImage * senseMap[i][1] + imImage * senseMap[i][0];			// re1*im2 + im1*re2
	}
}


void Reco::addToSumOfImages(fftwf_complex* sumOfImages, fftwf_complex* img, const int& gridSize)
{
	for (int i = 0; i < gridSize; i++)
	{
#pragma omp atomic
		sumOfImages[i][0] += img[i][0];
#pragma omp atomic
		sumOfImages[i][1] += img[i][1];
	}
}

std::vector<double> Reco::conjScalarProduct(const fftwf_complex* x, const fftwf_complex* y, const int& arrSize, const int& nThreads)
{
	double result0 = 0;
	double result1 = 0;

#pragma omp parallel for num_threads(nThreads) reduction(+:result0,result1)
	for (int i = 0; i < arrSize; i++)
	{
		result0 += x[i][0] * y[i][0] + x[i][1] * y[i][1];
		result1 += x[i][0] * y[i][1] - x[i][1] * y[i][0];
	}
	return { result0,result1 };
}

std::vector<double> Reco::complexDivision(std::vector<double> x, std::vector<double> y)
{
	std::vector<double> result(2);
	result[0] = (x[0] * y[0] + x[1] * y[1]) / (double)(y[0] * y[0] + y[1] * y[1]);
	result[1] = (x[1] * y[0] - x[0] * y[1]) / (double)(y[0] * y[0] + y[1] * y[1]);
	return result;
}

void Reco::addToArrayScaled(fftwf_complex* v1, const std::vector<double>& scale, fftwf_complex* v2, const int& arrSize, const int nThreads)
{
#pragma omp parallel for num_threads(nThreads) default(shared) schedule(static)
	for (int i = 0; i < arrSize; i++)
	{
		v1[i][0] += scale[0] * v2[i][0] - scale[1] * v2[i][1];
		v1[i][1] += scale[0] * v2[i][1] + scale[1] * v2[i][0];
	}
}

void Reco::addArraysScaled(fftwf_complex* p, const std::vector<double>& scale, fftwf_complex* r, const int& arrSize, const int nThreads)
{
#pragma omp parallel for num_threads(nThreads)  default(shared) schedule(static)
	for (int i = 0; i < arrSize; i++)
	{
		double pRe = p[i][0];
		double pIm = p[i][1];
		p[i][0] = r[i][0] + scale[0] * pRe - scale[1] * pIm;
		p[i][1] = r[i][1] + scale[0] * pIm + scale[1] * pRe;
	}
}

void Reco::cgSenseDynamic()
{
	// initial image for CG Sense
	fftwf_complex* a = fftwf_alloc_complex(totalGridSize);
	cgSenseImg = fftwf_alloc_complex(totalGridSize);
	clearFFTWComplex(a, totalGridSize, numThreads);

	// multiply a by the CONJUGATE sensitivity map
	for (int i = 0; i < totalGridSize; i++)
	{
		for (int ch = 0; ch < nCh; ch++)
		{
			a[i][0] += sMaps[ch][i][0] * img[ch][i][0] + sMaps[ch][i][1] * img[ch][i][1];		//re
			a[i][1] += sMaps[ch][i][0] * img[ch][i][1] - sMaps[ch][i][1] * img[ch][i][0];		//im		
		}
	}

	if (cgThreshold == -1)
	{
		std::cout << "\nError: no threshold set for CG SENSE!\n";
		exit(EXIT_FAILURE);
	}

	const int nSamplesPerCh = sX * sY * sZ;
	
	cgResiduals.clear();
	cgResiduals.shrink_to_fit();

	std::cout << "Begin cgSense\n";

	bool doDensityCorrection = !(denCorr.size() == 0);

	fftwf_complex* q = fftwf_alloc_complex(totalGridSize);
	fftwf_complex* p = fftwf_alloc_complex(totalGridSize);
	fftwf_complex* r = fftwf_alloc_complex(totalGridSize);

	for (int i = 0; i < totalGridSize; i++)
	{
		p[i][0] = a[i][0];
		p[i][1] = a[i][1];
		r[i][0] = a[i][0];
		r[i][1] = a[i][1];
		cgSenseImg[i][0] = 0;
		cgSenseImg[i][1] = 0;
		q[i][0] = 0;
		q[i][1] = 0;
	}

	// delete initial image guess
	fftwf_free(a);
	std::vector<double> resVec;

	int iteration = 0;
	const int nMaxIter = 15;
	double currentResidual = 1000;
	while (currentResidual > cgThreshold)
	{
		// clear q
		clearFFTWComplex(q, totalGridSize, numThreads);

		//calculate q = A * p		
#		pragma omp parallel for num_threads(numThreads) schedule(static) default(shared)
		for (int ch = 0; ch < nCh; ch++)
		{
			elementProduct(img[ch], p, sMaps[ch], totalGridSize);
		}

		nufftFWD();
		if (doDensityCorrection)
		{
#			pragma omp parallel for num_threads(numThreads) default(shared) schedule(static)
			for (int ch = 0; ch < nCh; ch++)
			{
				for (int i = 0; i < nSamplesPerCh; i++)
				{
					data[ch*nSamplesPerCh + i][0] *= denCorr[i];
					data[ch*nSamplesPerCh + i][1] *= denCorr[i];
				}
			}
		}
		nufftBWD();

#		pragma omp parallel for num_threads(numThreads) schedule(static) default(shared)
		for (int ch = 0; ch < nCh; ch++)
		{
			mutliplyByConjSenseMap(img[ch], sMaps[ch], totalGridSize);
			addToSumOfImages(q, img[ch], totalGridSize); // update is performed atomically, no need for critical section
		}

		if (!std::isfinite(q[0][0]) || !std::isfinite(q[0][1]))
		{
			std::cout << "\nError in function cgSense: NaN or INF value in image detected!\n";
			exit(EXIT_FAILURE);
		}

		std::vector<double> rr, pq;

#pragma omp parallel sections
		{
#pragma omp section
			{
				rr = conjScalarProduct(r, r, totalGridSize, numThreads / 2);
			}
#pragma omp section
			{
				pq = conjScalarProduct(p, q, totalGridSize, numThreads / 2);
			}
		}

		std::vector<double> rrBypq = complexDivision(rr, pq);
		addToArrayScaled(cgSenseImg, rrBypq, p, totalGridSize, numThreads);

		//r = r_old - r_old*r_old/pq * q
		rrBypq[0] *= -1;
		rrBypq[1] *= -1;
		addToArrayScaled(r, rrBypq, q, totalGridSize, numThreads);

		currentResidual = 0;

		for (int ct = 0; ct < totalGridSize; ct++)
		{
			currentResidual += r[ct][0] * r[ct][0] + r[ct][1] * r[ct][1];
		}
		currentResidual = sqrt(currentResidual);
		cgResiduals.push_back(currentResidual);

		std::cout << "\n\tIteration:\t" << iteration << "\t" << (currentResidual) << "\t" << cgThreshold << "\n";

		std::vector<double> rrNew = conjScalarProduct(r, r, totalGridSize);
		std::vector<double> scale = complexDivision(rrNew, rr);

		addArraysScaled(p, scale, r, totalGridSize, numThreads); //first argument is overwritten

		iteration++;
		if (iteration > nMaxIter) break;

	} // end iteration loop

	numPerformedCGIterations.push_back(iteration);

	fftwf_free(p);
	fftwf_free(r);
	fftwf_free(q);

	std::cout << "\nEnd cgSense\n";

}

// overload this function for custom Trajectory classes
void Reco::setTrajectory(WaveCaipi* t)
{
	t->calculateKMax();
	t->calculateTraj();

	trajectory = dynamic_cast<Trajectory*>(t);

	if (trajectory == NULL)
	{
		std::cout << "\nError: Trajectory conversion failed in setTrajectory!\n";
		exit(EXIT_FAILURE);
	}
}



bool Reco::createLogFile(const std::string& path)
{	
	return true;
}


void Reco::check()
{
	bool success = true;
	
	if (batchFiles.size() != 0)
	{
		std::cout << "Checking all files...";
		for (int i = 0; i < batchFiles.size(); i++)
		{
			std::string currentPath = batchFiles[i];
			std::FILE* f = std::fopen(currentPath.c_str(), "rb");
			if (f == NULL)
			{
				std::cout << "\nError: Could not open binary file " << currentPath << "\n";
				exit(EXIT_FAILURE);
			}
			std::fclose(f);
		}
		std::cout << " Done!\n";
	}


	if (!success)
	{
		exit(EXIT_FAILURE);
	}
}

void Reco::init()
{
	//kernel = new ConvKernel(gridSizeX, gridSizeY, gridSizeZ);
	kernel = new ConvKernel(this);
	kernel->setB(14.9086);
	kernel->setW(4.25);
	kernel->setKernelOrder(5);

	ft = new Fourier(gridSizeX, gridSizeY, gridSizeZ);

	sMaps.resize(nCh);
	for (int ch = 0; ch < nCh; ch++)
	{
		sMaps[ch] = NULL;
	}

	totalGridSize = gridSizeX * gridSizeY * gridSizeZ;

	if (!createLogFile(outputParentDir + "log.txt"))
	{
		std::cout << "\nWarning: Failed to create log-File. Could not open path " << outputParentDir + "log.txt" << "\n";
	}

	imgRange = { gridSizeX, gridSizeY, gridSizeZ };
}

void Reco::finalize()
{
	std::string path = outputParentDir + "results\\performedIterations.txt";
	std::ofstream os;
	os.open(path.c_str());
	for (int i = 0; i < numPerformedCGIterations.size(); i++)
	{
		os << numPerformedCGIterations[i] << "\n";
	}
	os.close();
}


void Reco::setSampling(const int& x, const int& y, const int& z)
{
	sX = x;
	sY = y;
	sZ = z;
}

void Reco::setGridSize(const int& x, const int& y, const int& z)
{
	gridSizeX = x;
	gridSizeY = y;
	gridSizeZ = z;
}

void Reco::setOffcenter(const int & x, const int & y, const int & z)
{
	offcenterX = x;
	offcenterY = y;
	offcenterZ = z;
}

void Reco::allocImg()
{
	img.resize(nCh);
	for (int i = 0; i < nCh; i++)
	{
		img[i] = fftwf_alloc_complex(totalGridSize);
	}
}

void Reco::allocData(fftwf_complex** d)
{
	*d = fftwf_alloc_complex(sX*sY*sZ*nCh);
}

void Reco::allocSMaps()
{
	for (int i = 0; i < nCh; i++)
	{
		sMaps[i] = fftwf_alloc_complex(gridSizeX*gridSizeY*gridSizeZ);
	}
}

void Reco::freeImg()
{
	for (int i = 0; i < nCh; i++)
	{
		fftwf_free(img[i]);
	}
	img.clear();
	img.shrink_to_fit();
}

void Reco::nufftBWD()
{
#	pragma omp parallel for num_threads(numThreads) schedule(dynamic) default(shared)
	for (int ch = 0; ch < nCh; ch++)
	{
		if (data == NULL)
		{
			std::cout << "\nError in nufftBWD function: data is NULL pointer!\n";
			exit(EXIT_FAILURE);
		}
		const int nSamplesPerCh = sX * sY * sZ;
		int firstEl = nSamplesPerCh * ch;

		const int gridSize = gridSizeX * gridSizeY * gridSizeZ;
		const double rescale = 1e-3;

		for (int i = 0; i < gridSize; i++)
		{
			img[ch][i][0] = 0;
			img[ch][i][1] = 0;
		}

		int index;
		float kernelVal;
		for (int i = 0; i < nSamplesPerCh; i++)
		{
			const fftwf_complex d = { data[firstEl + i][0], data[firstEl + i][1] };
			//std::cout << kernel->indices[i].size() << "\n";
			const int loopBound = kernel->indices[i].size();
			for (int ii = 0; ii < loopBound; ii++)
			{
				index = kernel->indices[i][ii];				// denotes cartesian grid point
				kernelVal = kernel->weights[i][ii];			// denotes corresponding kernel weight
				img[ch][index][0] += d[0] * kernelVal; // re
				img[ch][index][1] += d[1] * kernelVal; // im		
			}
		}

		ft->transformBWD(&img[ch][0]);
		
		// divide by fourier transform of gridding kernel (i.e., multiply by the inverse)
		for (int i = 0; i < gridSize; i++)
		{
			double gridRe = img[ch][i][0];
			double gridIm = img[ch][i][1];
			img[ch][i][0] = (gridRe * kernel->kernelFT[i][0] - gridIm * kernel->kernelFT[i][1]) * rescale;
			img[ch][i][1] = (gridIm * kernel->kernelFT[i][0] + gridRe * kernel->kernelFT[i][1]) * rescale;
		}

		reduceFOV(&img[ch][0]);
	}
}

void Reco::nufftFWD()
{
#	pragma omp parallel for num_threads(numThreads) schedule(dynamic) default(shared)
	for (int ch = 0; ch < nCh; ch++)
	{
		const int nSamplesPerCh = sX * sY * sZ;
		int firstEl = nSamplesPerCh * ch;

		for (int i = 0; i < gridSizeX*gridSizeY*gridSizeZ; i++)
		{
			double gridRe = img[ch][i][0];
			double gridIm = img[ch][i][1];
			img[ch][i][0] = (gridRe * kernel->kernelFT[i][0] - gridIm * kernel->kernelFT[i][1]);
			img[ch][i][1] = (gridIm * kernel->kernelFT[i][0] + gridRe * kernel->kernelFT[i][1]);
		}

		ft->transformFWD(&img[ch][0]);

		for (int i = 0; i < nSamplesPerCh; i++)
		{
			data[firstEl + i][0] = 0;
			data[firstEl + i][1] = 0;
			for (int ii = 0; ii < kernel->indices[i].size(); ii++)
			{
				int index = kernel->indices[i][ii];			// denotes cartesian grid point
				double kernelVal = kernel->weights[i][ii];	// denotes corresponding kernel weight
				data[firstEl + i][0] += img[ch][index][0] * kernelVal; // re
				data[firstEl + i][1] += img[ch][index][1] * kernelVal; // im		
			}
		}

	}
}

bool Reco::importParameters(const std::string& path)
{	
	return true;
}

void Reco::ConvKernel::calculateKernelFT()
{

	kernelFT = fftwf_alloc_complex(sizeX*sizeY*sizeZ);
	const double dkx = 1.0 / (double)(sizeX - 1);
	const double dky = 1.0 / (double)(sizeY - 1);
	const double dkz = 1.0 / (double)(sizeZ - 1);

	int kSpaceCenter[] = { int(sizeX / 2), int(sizeY / 2), int(sizeZ / 2) };

//#	pragma omp parallel for num_threads(reco->numThreads) schedule(static) default(shared)
	for (int i = 0; i < sizeX; i++)
	{
		for (int j = 0; j < sizeY; j++)
		{
			for (int k = 0; k < sizeZ; k++)
			{
				double distance[] = {
					sqrt(dkx*dkx*(i - kSpaceCenter[0])*(i - kSpaceCenter[0])),
					sqrt(dky * dky*(j - kSpaceCenter[1])*(j - kSpaceCenter[1])),
					sqrt(dkz * dkz*(k - kSpaceCenter[2])*(k - kSpaceCenter[2]))
				};
				
				kernelFT[k + sizeZ * j + sizeY * sizeZ*i][0] = convFunction(distance);
				kernelFT[k + sizeZ * j + sizeY * sizeZ*i][1] = 0;
			}
		}
	}

	reco->ft->transformBWD(kernelFT, false);
		
#	pragma omp parallel for num_threads(reco->numThreads) schedule(static) default(shared)
	for (int i = 0; i < sizeX*sizeY*sizeZ; i++)
	{
		double re = kernelFT[i][0];
		double im = 0;
		kernelFT[i][0] = re / (double)(re*re + im * im);
		kernelFT[i][1] = 0;
	}


}

inline int Reco::indexTransform(const int& i, const int& j, const int& k, const int& ni, const int& nj, const int& nk)
{
	return k + nk * j + nk * nj*i;
}

std::vector<int> Reco::indexBackTransform(const int& ind, const int& nx, const int& ny, const int& nz)
{
	int i = (int)floor(ind / (double)(ny*nz));
	int j = (int)floor((ind - ny * nz * i) / (double)(nz));
	int k = ind - nz * j - ny * nz*i;
	return { i,j,k };


}

inline void Reco::indexBackTransformPt(int* ret, const int& ind, const int& nx, const int& ny, const int& nz)
{
	ret[0] = (int)(ind / (double)(ny*nz));
	ret[1] = (int)((ind - ny * nz  * ret[0]) / (double)(nz));
	ret[2] = ind - nz * ret[1] - ny * nz*ret[0];
}


void Reco::reduceFOV(fftwf_complex* v)
{
	const int sizeX = gridSizeX;
	const int sizeY = gridSizeY;
	const int sizeZ = gridSizeZ;
	const int sX1 = imgRange[0];
	const int sY1 = imgRange[1];
	const int sZ1 = imgRange[2];

	if ((sizeX == sX1) &&
		(sizeY == sY1) &&
		(sizeZ == sZ1)
		)
	{
		return;
	}

	const int nSamples = sX1 * sY1 * sZ1;
	const int nSamplesZP = sizeX * sizeY * sizeZ;
	const int firstNonZeroX = (sizeX - sX1) / 2 + offcenterX;
	const int lastNonZeroX = sX1 + firstNonZeroX;
	const int firstNonZeroY = (sizeY - sY1) / 2 + offcenterY;
	const int lastNonZeroY = sY1 + firstNonZeroY;
	const int firstNonZeroZ = (sizeZ - sZ1) / 2 + offcenterZ;
	const int lastNonZeroZ = sZ1 + firstNonZeroZ;

	for (int x = 0; x < sizeX; x++)
	{
		for (int y = 0; y < sizeY; y++)
		{
			for (int z = 0; z < sizeZ; z++)
			{
				if (
					(x < firstNonZeroX) || (x > lastNonZeroX) ||
					(y < firstNonZeroY) || (y > lastNonZeroY) ||
					(z < firstNonZeroZ) || (z > lastNonZeroZ)
					)
				{
					int indZP = indexTransform(x, y, z, sizeX, sizeY, sizeZ);	// index in row major convention
					v[indZP][0] = 0;
					v[indZP][1] = 0;
				}
			}
		}
	}
}

void Reco::ConvKernel::clearWeights()
{
	weights.clear();
	weights.shrink_to_fit();
	indices.clear();
	indices.shrink_to_fit();
}

void Reco::ConvKernel::excludeMissingData()
{
	const int nSamplesPerCh = reco->sX * reco->sY * reco->sZ;

	for (int i = 0; i < nSamplesPerCh; i++)
	{
		if ((reco->data[i][0] == 0) && (reco->data[i][1] == 0))
		{
			weights[i].clear();
			weights[i].shrink_to_fit();
			indices[i].clear();
			indices[i].shrink_to_fit();
		}
	}

}

void Reco::ConvKernel::calculateWeights()
{
	weights.clear();
	weights.shrink_to_fit();
	indices.clear();
	indices.shrink_to_fit();

	const int nSamples = reco->getSX() * reco->getSY() * reco->getSZ();

	weights.resize(nSamples);
	indices.resize(nSamples);

	const int estimatedNumOfKernelPts = pow(2 * kernelOrder  + 1, 3);

	float dkx = 1.0 / (float)(sizeX - 1);
	float dky = 1.0 / (float)(sizeY - 1);
	float dkz = 1.0 / (float)(sizeZ - 1);

	double zeroDist[] = { 0., 0., 0. };
	const double kernelThreshold = convFunction(zeroDist) * 1e-7;

	std::vector< std::vector<int> > preCalcKernelPoints;
	for (int i1 = -kernelOrder; i1 <= kernelOrder; i1++)
	{
		for (int i2 = -kernelOrder; i2 <= kernelOrder; i2++)
		{
			for (int i3 = -kernelOrder; i3 <= kernelOrder; i3++)
			{
				preCalcKernelPoints.push_back({ i1,i2,i3 });
			}
		}
	}
	std::vector<int> pts;
	pts.reserve(estimatedNumOfKernelPts);

#pragma omp critical
	{
		std::cout << "Calculating kernel weights for each grid point\n";
	}

	const double invYZ = 1.0 / double(sizeY * sizeZ);
	const double invZ = 1.0 / double(sizeZ);
	const int yz = sizeY * sizeZ;



#	pragma omp parallel for num_threads(reco->getNumThreads()) schedule(static) default(shared) firstprivate(pts)
	for (int i = 0; i < nSamples; i++)
	{
		pts.clear();
		
		double kernelVal;

		int nearestGridPoint[] = {
			(int)((sizeX - 1) * (reco->trajectory->kSpaceTraj[i][0]) + 0.5),
			(int)((sizeY - 1) * (reco->trajectory->kSpaceTraj[i][1]) + 0.5),
			(int)((sizeZ - 1) * (reco->trajectory->kSpaceTraj[i][2]) + 0.5)
		};

		collectNeighborIndices(nearestGridPoint, pts, preCalcKernelPoints);

		int ind[3];

		for (int j = 0; j < pts.size(); j++)
		{
			ind[0] = (int)(pts[j] * invYZ);			// (int) works as floor-function
			ind[1] = (int)((pts[j] - yz * ind[0]) * invZ);
			ind[2] = pts[j] - sizeZ * ind[1] - yz * ind[0];

			double dist[] = {
				sqrt((reco->trajectory->kSpaceTraj[i][0] - ind[0] * dkx)*(reco->trajectory->kSpaceTraj[i][0] - ind[0] * dkx)),
				sqrt((reco->trajectory->kSpaceTraj[i][1] - ind[1] * dky)*(reco->trajectory->kSpaceTraj[i][1] - ind[1] * dky)),
				sqrt((reco->trajectory->kSpaceTraj[i][2] - ind[2] * dkz)*(reco->trajectory->kSpaceTraj[i][2] - ind[2] * dkz))
			};

			kernelVal = convFunction(dist);
			if (kernelVal > kernelThreshold)
			{
				indices[i].push_back(pts[j]);
				weights[i].push_back(kernelVal);
			}
		}

	}

#	pragma omp critical
	{
		std::cout << "Finished calculating kernel weights\n";
	}
}

inline void Reco::ConvKernel::collectNeighborIndices(int* startPoint, std::vector<int>& convPoints, const mat2d<int>& kernelPoints) const
{
	for (int i = 0; i < kernelPoints.size(); i++)
	{
		if ((startPoint[0] + kernelPoints[i][0] < 0) || (startPoint[0] + kernelPoints[i][0] > sizeX - 1) ||
			(startPoint[1] + kernelPoints[i][1] < 0) || (startPoint[1] + kernelPoints[i][1] > sizeY - 1) ||
			(startPoint[2] + kernelPoints[i][2] < 0) || (startPoint[2] + kernelPoints[i][2] > sizeZ - 1)
			)
		{
			continue;
		}
		else
		{

			convPoints.push_back((startPoint[2] + kernelPoints[i][2]) + sizeZ * (startPoint[1] + kernelPoints[i][1]) + sizeY * sizeZ*(startPoint[0] + kernelPoints[i][0]));
		}
	}
}

