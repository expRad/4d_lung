#include "Trajectory.h"
#include "Reco.h"
#include "globalVars.h"
#include <iostream>
#include <fstream>
#define _CRT_SECURE_NO_WARNINGS
#include <omp.h>

void Trajectory::calculateKMax()
{
	kmax = {
		{ 0.0, 1.0 },
	{ r->sY / (double)(r->fovY * 2.0*gammaBar), -r->sY / (double)(r->fovY * 2.0*gammaBar) },
	{ r->sZ / (double)(r->fovZ * 2.0*gammaBar), -r->sZ / (double)(r->fovZ * 2.0*gammaBar) }
	};
}

void WaveCaipi::calculateTraj()
{
#	pragma omp critical
	{
		std::cout << "Calculating wave-CAIPI trajectory...\n";
	}

	const double tADC = dwellTime * r->getSX();
	const double gradRasterTime = 10 * pow(10, -6);	// 10 us
	const int os = 10000;
	const double gradRasterTimeOs = gradRasterTime / (double)(os);

	const int gradSamples = (int)((tADC) / (double)(gradRasterTime));
	const int gradSamplesZ = (int)((tADC + 0.5*tADC / (double)(nOsc)) / (double)(gradRasterTime));

	const int nSamples = r->sX * r->sY * r->sZ;
	const double kSpaceOscAmpY = gradAmp * ampCorrY * tADC * pow(sin(nOsc*PI / (double)(2 * nOsc)), 2) / (double)(nOsc*PI);
	const double kSpaceOscAmpZ = gradAmp * ampCorrZ * tADC * pow(sin(nOsc*PI / (double)(2 * nOsc)), 2) / (double)(nOsc*PI);

	mat2d<double> traj(nSamples, std::vector<double>(3));

	std::vector<double> basePointsY(r->sY);
	std::vector<double> basePointsZ(r->sZ);
	mat2d<double> helix(r->sX, std::vector<double>(3));

	std::vector<double> gradWaveY(gradSamples);
	std::vector<double> gradWaveZ(gradSamplesZ);
	std::vector<double> kWaveY(gradSamples*os);
	std::vector<double> kWaveZ(gradSamplesZ*os);

	std::vector<double> gradWaveYos;
	std::vector<double> gradWaveZos;

	gradWaveYos.reserve(os*gradSamples);
	gradWaveZos.reserve(os*gradSamplesZ);
	for (int i = 0; i < gradSamples; i++)
	{
		gradWaveY[i] = -gradAmp * ampCorrY * sin(2.0 * PI * nOsc * double(i) / double(gradSamples));
	}
	for (int i = 0; i < gradSamplesZ; i++)
	{
		gradWaveZ[i] = gradAmp * ampCorrZ *sin(2.0 * PI * (nOsc + 0.5) * double(i) / double(gradSamplesZ));
	}
	for (int i = 0; i < gradSamples; i++)
	{
		for (int j = 0; j < os; j++)
		{
			gradWaveYos.push_back(gradWaveY[i]);
		}
	}
	for (int i = 0; i < gradSamplesZ; i++)
	{
		for (int j = 0; j < os; j++)
		{
			gradWaveZos.push_back(gradWaveZ[i]);
		}
	}

	kWaveY[0] = 0;
	kWaveZ[0] = 0;

	for (int i = 1; i < os*gradSamples; i++)
	{
		kWaveY[i] = kWaveY[i - 1] + gradRasterTimeOs * 0.5 * (gradWaveYos[i - 1] + gradWaveYos[i]);
	}
	for (int i = 1; i < os*gradSamplesZ; i++)
	{
		kWaveZ[i] = kWaveZ[i - 1] + gradRasterTimeOs * 0.5 * (gradWaveZos[i - 1] + gradWaveZos[i]);
	}

	double tY = 0;
	double tZ = tADC / double(4.0 * nOsc);
	for (int i = 0; i < r->sX; i++)
	{
		tY = (i + 1) * dwellTime - shiftY;
		tZ = tADC / double(4.0 * nOsc) + (i + 1) * dwellTime - shiftZ;
		
		int elY = std::round(tY / double(tADC) * kWaveY.size());
		int elZ = std::round(tZ / double(tADC + 0.5*tADC / double(nOsc)) * kWaveZ.size());
		if (elY > kWaveY.size() - 1) elY = kWaveY.size() - 1;
		if (elZ > kWaveZ.size() - 1) elZ = kWaveZ.size() - 1;

		helix[i][0] = i / (double)(r->sX - 1.0);

		if (tY < 0) helix[i][1] = 0.0;
		else helix[i][1] = kWaveY[elY];
		helix[i][2] = kWaveZ[elZ];
	}
	std::vector<double> ttY, ttZ;
	for (int i = 0; i < r->sX; i++)
	{
		ttY.push_back(helix[i][1]);
		ttZ.push_back(helix[i][2]);
	}

	for (int i = 0; i < r->sY; i++)
	{
		basePointsY[i] = kmax[1][0] - i / (double)(r->sY - 1.0) * (kmax[1][0] - kmax[1][1]);
	}
	for (int i = 0; i < r->sZ; i++)
	{
		basePointsZ[i] = kmax[2][1] - i / (double)(r->sZ - 1.0) * (kmax[2][1] - kmax[2][0]);
	}
	
	for (int i = 0; i < nSamples; i++)
	{
		traj[i][0] = helix[i%r->sX][0];
		traj[i][1] = basePointsY[(i / r->sX) % r->sY] + helix[i%r->sX][1];
		traj[i][2] = basePointsZ[i / (r->sX*r->sY)] + helix[i%r->sX][2];
	}

	double maxX = 0;
	double minX = 0;
	double maxY = 0;
	double minY = 0;
	double maxZ = 0;
	double minZ = 0;

	for (int i = 0; i < nSamples; i++)
	{
		if (traj[i][1] > maxY) maxY = traj[i][1];
		if (traj[i][1] < minY) minY = traj[i][1];
		if (traj[i][2] > maxZ) maxZ = traj[i][2];
		if (traj[i][2] < minZ) minZ = traj[i][2];
	}

	// add some zeros in k-space periphery
	const double nBorderPts = 42.0;
	const double freeSpaceFactorX = (1.0*nBorderPts) / double(r->getGridSizeX());
	const double freeSpaceFactorY = (1.0*nBorderPts - 0.405*kSpaceOscAmpY / double((fabs(kmax[1][0]) + fabs(kmax[1][1]))) * r->getGridSizeY()) / double(r->getGridSizeY());
	const double freeSpaceFactorZ = (1.0*nBorderPts - 0.405*kSpaceOscAmpZ / double((fabs(kmax[2][0]) + fabs(kmax[2][1]))) * r->getGridSizeZ()) / double(r->getGridSizeZ());

	for (int i = 0; i < nSamples; i++)
	{
		traj[i][1] -= minY;
		traj[i][2] -= minZ;

		traj[i][1] = traj[i][1] / (double)(maxY - minY);
		traj[i][2] = traj[i][2] / (double)(maxZ - minZ);

		traj[i][0] *= (1.0 - 2 * freeSpaceFactorX);
		traj[i][1] *= (1.0 - 2 * freeSpaceFactorY);
		traj[i][2] *= (1.0 - 2 * freeSpaceFactorZ);
		traj[i][0] += freeSpaceFactorX;
		traj[i][1] += freeSpaceFactorY;
		traj[i][2] += freeSpaceFactorZ;
	}


	mat2d<double> data2(nSamples, std::vector<double>(3));
	for (int k = 0; k < nSamples; k++)
	{
		int i3 = (int)((k % nSamples) / (r->sX*r->sY));
		int i2 = (int)(((k % nSamples) - r->sX * r->sY*i3) / r->sX);
		int i1 = (k % nSamples) - r->sX * i2 - r->sX * r->sY*i3;
		data2[i3 + r->sZ * i2 + r->sY * r->sZ*i1] = traj[k];
	}
	traj = data2;


#pragma omp critical
	{
		std::cout << "Finished calculating trajectory\n";
	}
	kSpaceTraj = traj;

}


