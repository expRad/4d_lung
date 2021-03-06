#include "stdafx.h"
#include "globalDefs.h"
#include "globalVars.h"
#include "Fourier.h"
#include "Trajectory.h"
#include "Reco.h"
#include "functions.h"
#include "IO.h"

/*
###########################
		To Do
Wisdom Filename
Functionality for single data set
###########################
*/

/*
###########################
The following libraries are required for this project
	- Boost	https://www.boost.org/
	- FFTW3	http://www.fftw.org/
	- CGAL	https://www.cgal.org/
###########################
*/

int main(int argc, char* argv[])
{
	// clock
	auto startGlobal = std::chrono::system_clock::now();

	// auto stream-flush
	std::cout << std::unitbuf;

	// Reconstruction object
	Reco reco;

	// allow nested omp-parallelism
	omp_set_nested(1);

	// reconstruction matrix size
	reco.setGridSize(512, 512, 512);

	reco.setSampling(448, 224, 112);
	reco.setNCh(30);

	reco.setFovY(0.500);
	reco.setFovZ(0.00223 * reco.getSZ());

	// set trajectory parameters
	WaveCaipi* wc1 = new WaveCaipi(&reco);
	wc1->setNOsc(4);
	wc1->setDwellTime(6.3e-6);		// s
	wc1->setGradAmp(0.006);			// T/m	
	wc1->setBandwidth(350);			// Hz/px
	wc1->setAmpCorrY(0.998);
	wc1->setAmpCorrZ(1.01);
	wc1->setShiftY(5.6 * 1e-6);		// s
	wc1->setShiftZ(6.5 * 1e-6);		// s

	reco.setTrajectory(wc1);

	reco.setDensityCorrection(Reco::Density::NONE); // no density correction required for wave-CAIPI

	//########################################
	// data paths
	////######################################

	// set the paths of the files to reconstruct
	reco.setBatchFiles({
	"X:\\Path\\To\\state0.bin",
	"X:\\Path\\To\\state7.bin"
	});

	// set the path to the output folder
	reco.setOutputParentDir("X:\\Path\\To\\outputDir\\");




	reco.setCGThreshold(2.08355e-6 * 0.3);	

	const int nSamples = reco.getNCh() * reco.getSX() * reco.getSY() * reco.getSZ();
	const int nSamplesPerCh = reco.getSX() * reco.getSY() * reco.getSZ();
	const int cGridSize = reco.getGridSizeX()*reco.getGridSizeY()*reco.getGridSizeZ(); // total grid points
	
	// initialize reconstruction and check for conflicts
	reco.init();
	reco.check();

	const bool multipleData = !(reco.getBatch() == 0);
	const int nRecos = reco.getBatch();
	bool verb = !multipleData;

	// set max. number of allowed threads
	reco.setNumThreads(16);
	omp_set_num_threads(reco.getNumThreads());
	std::cout << "Using up to " << omp_get_max_threads() << " threads\n";

	//initialize parallel FFT
	if (fftwf_init_threads() == 0)
	{
		std::cout << "\nWarning: Could not initialize parallel FFTW!\n";
	}

	fftwf_plan_with_nthreads(reco.getNumThreadsFFTW());

	reco.allocData(&reco.data);

	if (multipleData) reco.allocData(&reco.dataBuffer);

	bool finishedDataImport = false;


	
	//std::cout << "Enter parallel section\n";
	// import data and calculate trajectory, FT of convolution kernel and kernel weights
#	pragma omp parallel sections
	{
#		pragma omp section
		{
			auto startKernelTime = std::chrono::system_clock::now();
			if (multipleData)
			{
				//const std::string currentPath = reco.getBatchFileNum(0);
				std::future<bool> fut = std::async(std::launch::async, &IO::importDataBin, std::ref(reco), reco.data, static_cast<const std::string&>(reco.getBatchFileNum(0)));
				fut.wait();
				//IO::importDataBin(reco, reco.data, reco.batchFiles[0]);
			}
			//else IO::importDataBin(reco, reco.data, reco.dataPath);
			auto endKernelTime = std::chrono::system_clock::now();
			auto elapsedKernelTime = std::chrono::duration_cast<std::chrono::milliseconds>(endKernelTime - startKernelTime);
#			pragma omp critical
			{
				std::cout << "Data import time: " << elapsedKernelTime.count() << " ms\n";
			}
			finishedDataImport = true;
		}

#		pragma omp section
		{
#			pragma omp critical
			{
				std::cout << "Preparing FFT algorithm...\n";
			}

			reco.ft->initFT();

#			pragma omp critical
			{
				std::cout << "Finished preparing FFT algorithm\n";
			}

			reco.kernel->calculateKernelFT();

			// allocate memory for sensitivty maps
			reco.allocSMaps();
		}
#		pragma omp section
		{
			// calculate kernel weights
			auto startKernelTime = std::chrono::system_clock::now();
			reco.kernel->calculateWeights();				
			auto endKernelTime = std::chrono::system_clock::now();
			auto elapsedKernelTime = std::chrono::duration_cast<std::chrono::milliseconds>(endKernelTime - startKernelTime);
#			pragma omp critical
			{
				std::cout << "Kernel Calculation time: " << elapsedKernelTime.count() << " ms\n";
			}

			// Density correction	
			if (reco.getDensityCorrection() == Reco::Density::VORONOI3D)
			{
				reco.denCorr = estimateDensityV3D(reco.trajectory->kSpaceTraj, reco.getSX(), reco.getSY(), reco.getSZ(), false, 1.0);
			}
			else if (reco.getDensityCorrection() == Reco::Density::IMPORT)
			{
				reco.denCorr = IO::importDensityCorr(reco.getDensityCorrectionPath(), nSamplesPerCh);
			}				
			else
			{
				reco.denCorr.clear();
				reco.denCorr.shrink_to_fit();
			}			

			if (!finishedDataImport)
			{
#				pragma omp critical
				{
					std::cout << "Waiting for data import to finish...\n";
				}

			}
		}
	}//end parallel sections

	


	for (int dataNum = 0; dataNum < nRecos; dataNum++)
	{
		if (nRecos>1) std::cout << "Processing data set " << dataNum + 1 << " / " << nRecos << " ...\n";	

		// buffered data loading
		std::future<bool> futureDataImport;
		if (multipleData)
		{
			if (dataNum != 0)
			{
				std::swap(reco.data, reco.dataBuffer);
			}
			if (dataNum < nRecos - 1)
			{
				futureDataImport = std::async(std::launch::async, &IO::importDataBin, std::ref(reco), reco.dataBuffer,  static_cast<const std::string&>(reco.getBatchFileNum(dataNum + 1)));
			}
		}

		if (dataNum > 0)
		{
			reco.kernel->clearWeights();
			reco.kernel->calculateWeights();
		}
		
		reco.kernel->excludeMissingData(); // exclude zeros
			
		performDensityCorrection(reco, reco.data, reco.denCorr, nSamplesPerCh, reco.getNCh(), reco.getNumThreads());
			
		// allocate data for the individual channel-images
		reco.allocImg();

		//#####################################################
		// calculate single coil images
		//#####################################################
		if (verb) std::cout << "Calculating single-coil images...\n";
		reco.nufftBWD();			
		if (verb) std::cout << "\nDone.\n";

		// root-sum-of-squares image
		std::vector<float> imgAvg = rssq2DVecImg(reco.img, cGridSize);
		if (verb) std::cout << "Exporting rssq image... ";
		IO::exportVectorBin(imgAvg, reco.getOutputParentDir() + "resultRssq.bin");
		if (verb) std::cout << "Done.\n";

		// import sensitivity maps
		if (dataNum == 0)
		{				
			IO::importSensitivityMaps(reco.sMaps, cGridSize, reco.getNCh(), reco.getOutputParentDir() + "sensitivityMaps\\");
		}
			
		// deallocate memory that is no longer used
		imgAvg.clear();
		imgAvg.shrink_to_fit();
		
		//#######################################################
		// CG Sense 
		//#######################################################
		auto beginCG = std::chrono::system_clock::now();
		reco.cgSenseDynamic();		
		auto endCG = std::chrono::system_clock::now();
		auto elapsedCG = std::chrono::duration_cast<std::chrono::seconds>(endCG - beginCG);
		std::cout << "CG-Sense execution time: " << elapsedCG.count() << " s\n";
		std::cout << "Exporting Image... ";

		reco.finalize();
	
		IO::exportFFTWComplexBin(reco.cgSenseImg, reco.getOutputParentDir() + "results\\b_state" + std::to_string(dataNum) + ".bin", cGridSize);		

		std::cout << "Done.\n";

		// wait for data import to finish (data for the next iteration)
		if (dataNum < nRecos - 1)
		{
			// this will throw an exception if no future was created!
			futureDataImport.wait();
			if ((!futureDataImport.get()) && multipleData)				
			{
				std::cout << "\nError: Failed to asynchronously import data!\n";
				exit(EXIT_FAILURE);
			}
		}

	} // end loop over batch

	
	auto endGlobal = std::chrono::system_clock::now();
	auto elapsedGlobal = std::chrono::duration_cast<std::chrono::seconds>(endGlobal - startGlobal);
	std::cout << "Total execution time " << elapsedGlobal.count() << " s\n";

	std::cout << "Finished.\n";
	return 0;
}