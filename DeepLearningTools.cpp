#include <vector>
#include <iostream>
#include <time.h>
#include <math.h>
#include "DeepLearningTools.h"


FullConnection::FullConnection(const int &isize, const int &osize)
	:weight(osize,std::vector<double>(isize)),
	bias(osize),logit(osize),sigma(osize), lastLayerOutput(isize)
{
	inputRows = 1;
	inputCols = isize;
	outputRows = 1;
	outputCols = osize;

	srand((unsigned)time(NULL));
	for (size_t i = 0; i < osize; i++)
	{
		for (size_t j = 0; j < isize; j++)
		{
			weight[i][j] = getRandom();
		}
	}
	for (size_t i = 0; i < osize; i++)
	{
		bias[i] = getRandom();
	}
	std::cout << "A full connect layer has been initialized, input size = " << isize
		<< " output size = " << osize << std::endl;
}

std::vector<double> FullConnection::calcOutput(const std::vector<double> &input)
{
	lastLayerOutput = input;
	for (size_t outIndex = 0; outIndex < outputCols; outIndex++)
	{
		logit[outIndex] = 0;
		for (size_t inIndex = 0; inIndex < inputCols; inIndex++)
		{
			logit[outIndex] += weight[outIndex][inIndex] * lastLayerOutput[inIndex];
		}
		logit[outIndex] = sigmoid(logit[outIndex] + bias[outIndex]);
	}
	//std::cout << "The" << inputSize	<< " * " << outputSize << 
	//	" fc layer forward propagation done." << std::endl;
	return logit;
}

void FullConnection::changeWeight(
	const std::vector<double> &nextLayerSigma,
	const std::vector<std::vector<double> > & nextLayerWeight)
{
	if (nextLayerWeight.size() != 0)
	{
		for (size_t outIndex = 0; outIndex < outputCols; outIndex++)
		{
			sigma[outIndex] = 0;
			for (size_t nextLayerOutIndex = 0; nextLayerOutIndex < nextLayerWeight.size(); nextLayerOutIndex++)
			{
				sigma[outIndex] += nextLayerWeight[nextLayerOutIndex][outIndex] * nextLayerSigma[nextLayerOutIndex];
			}
			sigma[outIndex] *= logit[outIndex] * (1 - logit[outIndex]);
		}
	}
	else
	{
		for (size_t outIndex = 0; outIndex < outputCols; outIndex++)
		{
			sigma[outIndex] = -logit[outIndex] * (1 - logit[outIndex])*nextLayerSigma[outIndex];
		}
	}
	for (size_t outIndex = 0; outIndex < outputCols; outIndex++)
	{
		for (size_t inIndex = 0; inIndex < inputCols; inIndex++)
		{
			weight[outIndex][inIndex] -= lr * sigma[outIndex] * lastLayerOutput[inIndex];
		}
		bias[outIndex] -= lr * sigma[outIndex];
	}
	//std::cout << "The" << inputSize << " * " << outputSize <<
	//	" fc layer back propagation done." << std::endl;
}

SoftMax::SoftMax(const int &isize, const int &osize)
	:weight(osize, std::vector<double>(isize + 1)),
	bias(osize), logit(osize), sigma(osize), lastLayerOutput(isize)
{
	inputRows = 1;
	inputCols = isize;
	outputRows = 1;
	outputCols = osize;

	srand((unsigned)time(NULL));
	for (size_t i = 0; i < osize; i++)
	{
		for (size_t j = 0; j < isize + 1; j++)
		{
			weight[i][j] = 1;
		}
	}
	for (size_t i = 0; i < osize; i++)
	{
		bias[i] = 1;
	}
	std::cout << "A softmax layer has been initialized, input size = " << isize
		<< " output size = " << osize << std::endl;
}

std::vector<double> SoftMax::calcOutput(const std::vector<double> &input)
{
	lastLayerOutput = input;
	lastLayerOutput.resize(lastLayerOutput.size() + 1, 1);
	for (size_t outIndex = 0; outIndex < outputCols; outIndex++)
		logit[outIndex] = 0;
	
	for (size_t outIndex = 0; outIndex < outputCols; outIndex++)
	{
		for (size_t inIndex = 0; inIndex < inputCols + 1; inIndex++)
		{
			logit[outIndex] += weight[outIndex][inIndex] * lastLayerOutput[inIndex];
		}
	}
	//std::cout << "The" << inputSize	<< " * " << outputSize << 
	//	" fc layer forward propagation done." << std::endl;
	double sum = 0;
	for (size_t outIndex = 0; outIndex < outputCols; outIndex++)
	{
		logit[outIndex] = exp(logit[outIndex]);
		sum += logit[outIndex];
	}
	for (size_t outIndex = 0; outIndex < outputCols; outIndex++)
	{
		logit[outIndex] /= sum;
	}
	return logit;
}
void SoftMax::changeWeight(
	const std::vector<double> &nextLayerSigma,
	const std::vector<std::vector<double> > & nextLayerWeight)
{
	for (size_t outIndex = 0; outIndex < outputCols; outIndex++)
	{
		for (size_t inIndex = 0; inIndex < inputCols + 1; inIndex++)
		{
			//std::cout << weight[inIndex][outIndex] << " " << lr * lastLayerOutput[inIndex] * nextLayerSigma[outIndex] << " ";
			weight[outIndex][inIndex] += lr * lastLayerOutput[inIndex] * nextLayerSigma[outIndex];
			//std::cout << weight[inIndex][outIndex] << std::endl;
		}
	}
	for (size_t i = 0; i < sigma.size(); i++)
	{
		sigma[i] = -nextLayerSigma[i];
	}
	
}