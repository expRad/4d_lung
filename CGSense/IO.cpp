#include "IO.h"
#define _CRT_SECURE_NO_WARNINGS

IO::IO()
{
}

IO::~IO()
{
}

void IO::exportVectorBin(const std::vector<float>& v, std::string path)
{
	std::FILE* f = std::fopen(path.c_str(), "wb");
	if (f == NULL)
	{
		std::cout << "\nError: Could not open binary file " << path << "\n";
		exit(EXIT_FAILURE);
	}
	std::fwrite(&v[0], sizeof(std::vector<float>::value_type), v.size(), f);
	std::fclose(f);
}


void IO::importSensitivityMaps(std::vector<fftwf_complex*>& maps, const int N, const int numCh, std::string path)
{
	std::cout << "Importing sensitivity maps\n";
	for (int i = 0; i < numCh; i++)
	{
		std::string filename = path + "map" + std::to_string(i) + ".bin";
		std::FILE* f = std::fopen(filename.c_str(), "rb");
		if (f == NULL)
		{
			std::cout << "\nError: Could not open binary file " << path << "\n";
			exit(EXIT_FAILURE);
		}

		size_t num = std::fread(&maps[i][0], sizeof(float), 2 * N, f);

		if (num != 2 * N)
		{
			std::cout << "\nError: Number of imported data points is different from expected number!\nFilename: " << filename << "\n";
			std::cout << num << "\t" << 2 * N << "\n";
			exit(EXIT_FAILURE);
		}
		
		std::cout << "\t" << (int)(100 * (i + 1) / (double)(numCh)) << "%\r";
	}
	std::cout << "Done.\n";

}



bool IO::importDataBin(const Reco& r, fftwf_complex* v, const std::string& path)
{
	bool success = true;
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (f == NULL)
	{
		std::cout << "\nError: Could not open binary file " << path << "\n";
		exit(EXIT_FAILURE);
	}
	const int nSamplesPerCh = r.getSX() * r.getSY() * r.getSZ();
	const int N =nSamplesPerCh* r.getNCh();

	float* buf = new float[2 * N];
	size_t num = std::fread(&buf[0], sizeof(float), 2 * N, f);
	if (num != 2 * N)
	{
		std::cout << "\nError: Number of imported data points is different from expected number!\nDoes expected precision match file precision?\n";
		std::cout << "Path: " << path << "\n";
		std::cout << 2 * N << "  " << num << "\n";
		exit(EXIT_FAILURE);
	}

	int ch = 0;
	const double nSamplesPerChInv = 1.0 / double(nSamplesPerCh);
	const double invSxSy = 1.0 / double(r.getSX()*r.getSY());
	const int sYsZ = r.getSY() * r.getSZ();
	const int sXsY = r.getSX() * r.getSY();
	const double invSx = 1.0 / double(r.getSX());

#	pragma omp parallel for num_threads(r.getNumThreads()) schedule(static) firstprivate(ch)
	for (int k = 0; k <N; k++)
	{
		ch = k * nSamplesPerChInv;
		int i3 = (int)(((k) % nSamplesPerCh) * invSxSy);
		int i2 = (int)((((k) % nSamplesPerCh) - sXsY * i3) * invSx);
		int i1 = ((k) % nSamplesPerCh) - r.getSX() * i2 - sXsY * i3;
		v[ch*nSamplesPerCh + i3 + r.getSZ() * i2 + sYsZ * i1][0] = buf[2 * k];
		v[ch*nSamplesPerCh + i3 + r.getSZ() * i2 + sYsZ * i1][1] = buf[2 * k + 1];
	}

	delete[] buf;

	return success;
}

std::vector<float> IO::importDensityCorr(const Reco & r)
{
#	pragma omp critical
	{
		std::cout << "Importing density correction matrix\n";
	}
	const int N = r.getSX() * r.getSY() * r.getSZ();
	std::vector<float> d(N);
	std::FILE* f = std::fopen(r.getDensityCorrectionPath().c_str(), "rb");
	if (f == NULL)
	{
		std::cout << "\nError: Could not open binary file " << r.getDensityCorrectionPath() << "\n";
		exit(EXIT_FAILURE);
	}
	size_t num = std::fread(&d[0], sizeof(float), N, f);
	if (num != d.size())
	{
		std::cout << "\nError: Number of imported data points is different from expected number!\nDoes expected precision match file precision?\n";
		std::cout << d.size() << "  " << num << "\n";
		exit(EXIT_FAILURE);
	}
	std::fclose(f);
#	pragma omp critical
	{
		std::cout << "Done importing density correction matrix\n";
	}
	return d;
}

std::vector<float> IO::importDensityCorr(const std::string & path, const int & N)
{
#	pragma omp critical
	{
		std::cout << "Importing density correction matrix\n";
	}
	std::vector<float> d(N);
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (f == NULL)
	{
		std::cout << "\nError: Could not open binary file " << path << "\n";
		exit(EXIT_FAILURE);
	}
	size_t num = std::fread(&d[0], sizeof(float), N, f);
	if (num != d.size())
	{
		std::cout << "\nError: Number of imported data points is different from expected number!\nDoes expected precision match file precision?\n";
		std::cout << d.size() << "  " << num << "\n";
		exit(EXIT_FAILURE);
	}
	std::fclose(f);
#	pragma omp critical
	{
		std::cout << "Done importing density correction matrix\n";
	}
	return d;

}

void IO::exportFFTWComplexBin(fftwf_complex* w, const std::string& path, const int& N)
{
	std::vector<float> v(2 * N);
	for (int i = 0; i < 2 * N; i += 2)
	{
		v[i] = w[i / 2][0];
		v[i + 1] = w[i / 2][1];
	}

	std::FILE* f = std::fopen(path.c_str(), "wb");
	if (f == NULL)
	{
		std::cout << "\nError: Could not open binary file " << path << "\n\tTrying to rename...";
		std::string path2 = path + "_1.bin";
		f = std::fopen(path2.c_str(), "wb");
		if (f == NULL)
		{
			std::cout << " Failed!\n";
			exit(EXIT_FAILURE);
		}
		else
		{
			std::cout << " Succeeded!\n";
		}

	}
	std::fwrite(&v[0], sizeof(std::vector<float>::value_type), v.size(), f);
	std::fclose(f);
}