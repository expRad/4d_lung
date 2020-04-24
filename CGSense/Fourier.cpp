#include "Fourier.h"


void Fourier::initFT()
{
	if (fftwf_import_wisdom_from_filename(wisdomFileName.c_str()) == 0)
	{
		//std::cout << "Note: No FFTW-Wisdom data found. Need to compute new FFT plans\n";
	}

	fftwf_complex* tmpFFTWComplex = fftwf_alloc_complex(sizeX*sizeY*sizeZ);
	planFWD = fftwf_plan_dft_3d(sizeX, sizeY, sizeZ, tmpFFTWComplex, tmpFFTWComplex, FFTW_FORWARD, FFTW_MEASURE);
	planBWD = fftwf_plan_dft_3d(sizeX, sizeY, sizeZ, tmpFFTWComplex, tmpFFTWComplex, FFTW_BACKWARD, FFTW_MEASURE);
	fftwf_export_wisdom_to_filename(wisdomFileName.c_str());
	fftwf_free(tmpFFTWComplex);

}

void Fourier::transformFWD(fftwf_complex* a) const
{
	signAlternate3D(a);
	fftwf_execute_dft(planFWD, a, a);
	signAlternate3D(a);
}

void Fourier::transformBWD(fftwf_complex* a, bool rescale) const
{
	signAlternate3D(a);
	fftwf_execute_dft(planBWD, a, a);
	signAlternate3D(a);

	if (rescale)
	{
		const double rescaleFactor = 1.0 / sqrt(sizeX*sizeY*sizeZ);
		for (int i = 0; i < sizeX*sizeY*sizeZ; i++)
		{
			a[i][0] *= rescaleFactor;
			a[i][1] *= rescaleFactor;
		}
	}

}


void Fourier::signAlternate3D(fftwf_complex* a) const
{
	//	#pragma omp parallel for num_threads(nThreads) default(shared) schedule(static)
	for (int i = 0; i < sizeX; i++)
	{
		const int iModResult = i % 2;
		for (int j = 0; j < sizeY; j++)
		{
			const int jModResult = j % 2;
			for (int k = 0; k < sizeZ; k += 2)
			{
				if (jModResult == iModResult)
				{
					int ind = indexTransform(i, j, k, sizeX, sizeY, sizeZ);
					a[ind][0] *= -1;
					a[ind][1] *= -1;
				}
				else
				{
					int ind = indexTransform(i, j, k + 1, sizeX, sizeY, sizeZ);
					a[ind][0] *= -1;
					a[ind][1] *= -1;
				}
			}
		}
	}

}

inline int Fourier::indexTransform(const int& i, const int& j, const int& k, const int& ni, const int& nj, const int& nk) const
{
	return k + nk * j + nk * nj*i;
}