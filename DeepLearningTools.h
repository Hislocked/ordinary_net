#pragma once
#include <vector>
#include <iostream>
#include <math.h>

#define lr 0.001 //learn rate

inline double getRandom()    // get random from -1 to 1
{
	return (((double)rand() / RAND_MAX) * 2.0 - 1);
}
inline double sigmoid(double x)
{
	double ans = 1 / (1 + exp(-x));
	return ans;
}

struct NetLayer //virtual base struct
{
	virtual std::vector<double> calcOutput(const std::vector<double> &input) = 0;
	virtual void changeWeight(
		const std::vector<double> &nextLayerLoss,
		const std::vector<std::vector<double> > & nextLayerSigma) = 0;
	int inputRows = 1, inputCols = 1, outputRows = 1, outputCols = 1;
};

struct FullConnection : public NetLayer
{
	FullConnection(const int &isize, const int &osize);

	//get layer logits
	std::vector<double> calcOutput(const std::vector<double> &input);

	//do layer bp
	void changeWeight(
		const std::vector<double> &nextLayerLoss, 
		const std::vector<std::vector<double> > & nextLayerSigma);

	int inputRows, inputCols, outputRows, outputCols;
	std::vector<std::vector<double> > weight;
	std::vector<double> bias, logit, sigma, lastLayerOutput;
};

struct SoftMax : public NetLayer
{
	SoftMax(const int &isize, const int &osize);

	//get layer logits
	std::vector<double> calcOutput(const std::vector<double> &input);

	//do layer bp
	void changeWeight(
		const std::vector<double> &nextLayerLoss,
		const std::vector<std::vector<double> > & nextLayerSigma);

	int inputRows, inputCols, outputRows, outputCols;
	std::vector<std::vector<double> > weight;
	std::vector<double> bias, logit, sigma, lastLayerOutput;
};

//struct Convolution : public NetLayer
//{
//	Convolution(const int &isize, const int &osize);
//
//	//get layer logits
//	std::vector<double> calcOutput(const std::vector<double> &input);
//
//	//do layer bp
//	void changeWeight(
//		const std::vector<double> &nextLayerLoss,
//		const std::vector<std::vector<double> > & nextLayerSigma);
//
//	int inputRows, inputCols, outputRows, outputCols;
//	std::vector<std::vector<double> > weight;
//	std::vector<double> bias, logit, sigma, lastLayerOutput;
//};