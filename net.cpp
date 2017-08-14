#include <vector>
#include <iostream>
#include <time.h>
#include <fstream>
#include <string>
#include <memory>
#include "DeepLearningTools.h"
#include "net.h"

Net::Net(int layers, std::vector<std::string> nS, std::vector<std::vector<int> > nSNum) : netStruct(layers)
{
	if (nS.size() == 0)
	{
		netStruct[0] = new SoftMax(28 * 28, 10);
	}
	else
	{
		for (size_t i = 0; i < nS.size(); i++)
		{
			if (nS[i] == "s")
			{
				netStruct[i] = new SoftMax(nSNum[i][0]* nSNum[i][1], nSNum[i][2]* nSNum[i][3]);
			}
			if (nS[i] == "f")
			{
				netStruct[i] = new FullConnection(nSNum[i][0] * nSNum[i][1], nSNum[i][2] * nSNum[i][3]);
			}
		}
	}
}

std::vector<std::vector<double> > Net::forwardPropagation(const std::vector<std::vector<double> > &input)
{
	std::vector<double> intputOneRow(input[0]);
	for (size_t i = 0; i < netStruct.size(); i++)
	{
		FullConnection* fcP = dynamic_cast<FullConnection*>(netStruct[i]);
		if (fcP != NULL)
		{
			if (intputOneRow.size() != fcP->inputCols)
			{
				std::cerr << i << "th layer:input and layer doesn't match.\n" <<
					"input size:1 *" << intputOneRow.size() << "\n" <<
					"layer size:" << fcP->inputRows << "*" << fcP->inputCols << std::endl;
			}
			intputOneRow = fcP->calcOutput(intputOneRow);
		}
		else
		{
			SoftMax* smP = dynamic_cast<SoftMax*>(netStruct[i]);
			if (smP != NULL)
			{
				if (intputOneRow.size() != smP->inputCols)
				{
					std::cerr << i << "th layer:input and layer doesn't match.\n" <<
						"input size:1 *" << intputOneRow.size() << "\n" <<
						"layer size:" << smP->inputRows << "*" << smP->inputCols << std::endl;
				}
				intputOneRow = smP->calcOutput(intputOneRow);
			}
			else
			{
				std::cerr << "Unkown layer!" << std::endl;
			}
			
		}
	}
	std::vector<std::vector<double> > ret(1,intputOneRow);
	return ret;
}

void Net::backPropagation(std::vector<double> error)
{
	std::vector<std::vector<double> > nextLayerWeight;
	for (int i = netStruct.size() - 1; i >= 0; i--)
	{
		FullConnection* fcP = dynamic_cast<FullConnection*>(netStruct[i]);
		if (fcP != NULL)
		{
			//std::cout << i << "th layer error size:" << error.size() << std::endl;
			//if (nextLayerWeight.size() == 0)
			//{
			//	std::cout << i << "th layer next layer weight size: 0" << std::endl;
			//}
			//else
			//{
			//	std::cout << i << "th layer next layer weight size: " << 
			//		nextLayerWeight.size() << "*" << nextLayerWeight[0].size() << std::endl;
			//}
			fcP->changeWeight(error, nextLayerWeight);
			error = fcP->sigma;
			nextLayerWeight = fcP->weight;
			
		}
		else
		{
			SoftMax* smP = dynamic_cast<SoftMax*>(netStruct[i]);
			if (smP != NULL)
			{
				smP->changeWeight(error, nextLayerWeight);
				error = smP->sigma;
				nextLayerWeight = smP->weight;
			}
			else
			{
				std::cerr << "Unkown layer!" << std::endl;
			}
		}
	}
}

bool Net::saveWeight(int steps)
{
	std::string sSteps(std::to_string(steps)),sTime(std::to_string(time(NULL)));
	std::ofstream saveFile("saved/" + sSteps + " " + sTime + ".save", std::ostream::out);
	if (saveFile.is_open())
	{
		std::vector<std::vector<double> > weight;
		std::vector<double> bias;
		saveFile << "This is a TupuNet weights file." << std::endl;
		saveFile << steps << std::endl;
		saveFile << netStruct.size() << std::endl;
		for (size_t i = 0; i < netStruct.size(); i++)
		{
			FullConnection* fcP = dynamic_cast<FullConnection*>(netStruct[i]);
			if (fcP != NULL)
			{
				saveFile 
					<< "f " << fcP->inputRows
					<< " " << fcP->inputCols
					<< " " << fcP->outputRows
					<< " " << fcP->outputCols << std::endl;
			}
			else
			{
				SoftMax* smP = dynamic_cast<SoftMax*>(netStruct[i]);
				if (smP != NULL)
				{
					saveFile 
						<< "s " << smP->inputRows
						<< " " << smP->inputCols
						<< " " << smP->outputRows
						<< " " << smP->outputCols << std::endl;
				}
				else
				{
					std::cerr << "Unkown layer!" << std::endl;
					saveFile.close();
					return true;
				}

			}
		}
		for (size_t i = 0; i < netStruct.size(); i++)
		{
			FullConnection* fcP = dynamic_cast<FullConnection*>(netStruct[i]);
			if (fcP != NULL)
			{
				weight = fcP->weight;
				for (size_t m = 0; m < weight.size(); m++)
				{
					for (size_t n = 0; n < weight[m].size(); n++)
					{
						saveFile << weight[m][n] << " ";
					}
				}
				saveFile << std::endl;
				bias = fcP->bias;
				for (size_t i = 0; i < bias.size(); i++)
				{
					saveFile << bias[i];
				}
				saveFile << std::endl;
			}
			else
			{
				SoftMax* smP = dynamic_cast<SoftMax*>(netStruct[i]);
				if (smP != NULL)
				{
					weight = smP->weight;
					for (size_t m = 0; m < weight.size(); m++)
					{
						for (size_t n = 0; n < weight[m].size(); n++)
						{
							saveFile << weight[m][n] << " ";
						}
					}
					saveFile << std::endl;
					bias = smP->bias;
					for (size_t i = 0; i < bias.size(); i++)
					{
						saveFile << bias[i];
					}
					saveFile << std::endl;
				}
				else
				{
					std::cerr << "Unkown layer!" << std::endl;
					saveFile.close();
					return true;
				}
			}
		}
		std::cout << "Net weights have been saved!" << std::endl;
		return true;
	}
	else
	{
		std::cout << "Save weights failed!" << std::endl;
		return false;
	}
	saveFile.close();
}

bool Net::loadWeight(std::string fileName) {
	std::string sTmp;
	int irTmp, icTmp, orTmp, ocTmp;
	std::ifstream readFile(fileName, std::ifstream::in);
	if (readFile.is_open())
	{
		std::string oneLine;
		int steps,num;
		std::getline(readFile, oneLine);
		if (oneLine != "This is a TupuNet weights file.")
		{
			std::cout << "This file is not a TupuNet weight file." << std::endl;
			readFile.close();
			return false;
		}
		std::getline(readFile, oneLine);
		steps = std::stoi(oneLine);
		std::cout << "The weights are now been trained at " << steps << "steps" << std::endl;
		std::getline(readFile, oneLine);
		int layers = std::stoi(oneLine);
		for (size_t i = 0; i < netStruct.size(); i++)
		{
			delete netStruct[i];
		}
		netStruct.clear();
		for (size_t i = 0; i < layers; i++)
		{
			//std::getline(readFile, oneLine);
			readFile >> sTmp >> irTmp >> icTmp >> orTmp >> ocTmp;
			readFile.ignore();
			if (sTmp == "s")
			{
				SoftMax * sP = new SoftMax(irTmp * icTmp, orTmp * ocTmp);
				netStruct.push_back(sP);
			}
			else if (sTmp == "f")
			{
				FullConnection * fP = new FullConnection(irTmp * icTmp, orTmp * ocTmp);
				netStruct.push_back(fP);
			}
			else
			{
				std::cerr << "Unkown layer!" << std::endl;
				readFile.close();
				return false;
			}
			
		}
		for (size_t i = 0; i < netStruct.size(); i++)
		{
			FullConnection* fcP = dynamic_cast<FullConnection*>(netStruct[i]);
			if (fcP != NULL)
			{
				int weightNum = fcP->inputRows * fcP->inputCols * fcP->outputRows * fcP->outputCols;
				int tmp, m = 0, n = 0;
				for (size_t j = 0; j < weightNum; j++)
				{
					readFile >> fcP->weight[n][m++];
					if (m == fcP->inputCols)
					{
						m = 0;
						n++;
					}
				}
				for (size_t j = 0; j < fcP->bias.size(); j++)
				{
					readFile >> fcP->bias[j];
				}
				std::cout << "Read file done!" << std::endl;
			}
			else
			{
				SoftMax* smP = dynamic_cast<SoftMax*>(netStruct[i]);
				if (smP != NULL)
				{
					int weightNum = smP->inputRows * (smP->inputCols + 1) * smP->outputRows * smP->outputCols;
					int tmp, m = 0, n = 0;
					for (size_t j = 0; j < weightNum; j++)
					{
						readFile >> smP->weight[n][m++];
						if (m == smP->inputCols + 1)
						{
							m = 0;
							n++;
						}
					}
					for (size_t j = 0; j < smP->bias.size(); j++)
					{
						readFile >> smP->bias[j];
					}
					std::cout << "Read file done!" << std::endl;
				}
				else
				{
					std::cerr << "Unkown layer!" << std::endl;
					readFile.close();
					return false;
				}
			}
		}
	}
	else
	{
		std::cout << "Open file failed!" << std::endl;
		return false;
	}
	readFile.close();
	return true;
}
