#pragma once
#include <vector>
#include <iostream>
#include <memory>
#include "DeepLearningTools.h"

struct Net
{
	Net(int, std::vector<std::string>,std::vector<std::vector<int> >);
	std::vector<std::vector<double> > forwardPropagation(const std::vector<std::vector<double> > &inputImage);
	void backPropagation(std::vector<double>);
	bool saveWeight(int steps);
	bool loadWeight(std::string);

	std::vector<NetLayer*> netStruct;
};