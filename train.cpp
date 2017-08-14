#include <iostream>
#include <vector>
#include <math.h>
#include <string>
#include <cstring>
#include "net.h"
#include "loadDataset.h"
using namespace std;

int layers = 1, irTmp, icTmp, orTmp, ocTmp, stepAll;
vector<vector<int>> nSNum;
vector<string> nS;
string saveFile,sTmp;

vector<vector<double>>  getLoss(vector<vector<double>> logits, double labels) {
	for (size_t i = 0; i < logits[0].size(); i++)
	{
		logits[0][i] = ((i == (int)labels) ? 1 : 0) - logits[0][i];
	}
	return logits;
}

int top_1(vector<vector<double>> logits) {
	int maxIndex = 0;
	double maxNum = 0;

	for (size_t i = 0; i < logits[0].size(); i++)
	{
		if (logits[0][i] > maxNum)
		{
			maxNum = logits[0][i];
			maxIndex = i;
		}
	}
	return maxIndex;
}

void help(){
	cout << "This program is not a complete version now." << endl;
	cout << "You can use some functionality." << endl;
	cout << "Please follow the instructions below:" << endl;
}

void instructions() {
	cout << "Do you want to continue training from a save file?" << endl;
	cout << "Please enter file name with path or nothing but ENTER." << endl;
	getline(cin, saveFile);
	if (saveFile.size() == 0)
	{
		cout << "Let's start a new deep net." << endl;
		cout << "Input 0 for quick start. You will get a 1*784 1*10 softmax layer and start training 100000 steps quickly." << endl;
		cout << "----------------------------------------------" << endl;
		cout << "Deep net layers : "; getline(cin, sTmp);
		layers = stoi(sTmp);
		if (layers == 0) {
			layers++;
			stepAll = 100000;
			return;
		}
		cout << "Now please give me the struct of the layers." << endl;
		cout << "Sorry that there are only two layer can be used now." << endl;
		cout << "Note that f for full connection,s for softmax." << endl;
		cout << "Input the layer size of input/output rows and cols at the same time." << endl;
		cout << "FOR EXAMPLE: f 1 784 1 10" << endl;
		cout << "That means a full connection layer with input size 1*784, output size 1*10." << endl;
		cout << "Now please input your network structure." << endl;
		for (size_t i = 0; i < layers; i++)
		{
			cout << "Layer" << i << " : "; cin >> 
				sTmp >> irTmp >> icTmp >> orTmp >> ocTmp; 
			nS.push_back(sTmp);
			nSNum.push_back({ irTmp, icTmp, orTmp, ocTmp });
		}
	}
	cout << "How many steps you want to train?" << endl;
	cin >> stepAll;
	cout << "Now start training." << endl;
}

int main(int argc, char *argv[]) {
	if (argc > 1)	help();
	instructions();
	vector<double> labels;
	read_Mnist_Label("dataset/train-labels.idx1-ubyte", labels);
	vector<vector<double> > images;
	read_Mnist_Images("dataset/train-images.idx3-ubyte", images);
	for (size_t i = 0; i < images.size(); i++)
	{
		for (size_t j = 0; j < images[i].size(); j++)
		{
			images[i][j] /= 255;
		}
	}
	if (layers == 0)	layers++;
	Net netForMnist(layers, nS, nSNum);
	if (saveFile.size() != 0)
	{
		cout << "Restoring save file." << endl;
		if (!netForMnist.loadWeight(saveFile))
			return 0;
	}
	double lossSum = 0;
	int stepCount = 0;
	for (size_t step = 1; step <= stepAll; step++)
	{
		int sampleIndex = (int)rand()%images.size();
		vector<vector<double>> input(1, images[sampleIndex]);
		vector<vector<double>> logits = netForMnist.forwardPropagation(input);
		vector<vector<double>> loss = getLoss(logits, labels[sampleIndex]);
		netForMnist.backPropagation(loss[0]);
		lossSum += log(logits[0][labels[sampleIndex]]);
		stepCount++;
		if (step % 1000 == 0)
		{
			//double lossSum = 0;
			//for (size_t i = 0; i < loss[0].size(); i++)
			//{
			//	lossSum += 0.5*loss[0][i] * loss[0][i];
			//}
			cout << "step " << step << "-loss(avg):" << lossSum/ stepCount << endl;
			lossSum = 0;
			stepCount = 0;
		}
		if (step % 100000 == 0)
		{
			netForMnist.saveWeight(step);
			cout << "Start evaluation." << endl;
			vector<double> testLabels;
			read_Mnist_Label("dataset/t10k-labels.idx1-ubyte", testLabels);
			vector<vector<double> > testImages;
			read_Mnist_Images("dataset/t10k-images.idx3-ubyte", testImages);
			for (size_t i = 0; i < testImages.size(); i++)
			{
				for (size_t j = 0; j < testImages[i].size(); j++)
				{
					testImages[i][j] /= 255;
				}
			}
			double sum = 0, correct = 0;
			for (size_t i = 1; i <= testLabels.size(); i++)
			{
				vector<vector<double>> testInput(1, testImages[i - 1]);
				vector<vector<double>> testLogits = netForMnist.forwardPropagation(testInput);
				int prediction = top_1(testLogits);
				//for (size_t i = 0; i < testLogits[0].size(); i++)
				//{
				//	cout << testLogits[0][i] << " ";
				//}
				//cout << endl;
				//cout << prediction << (int)testLabels[i - 1] << " ";
				if (prediction == (int)testLabels[i-1])
				{
					correct++;
				}
				sum++;
				if (i % 1000 == 0)
				{
					cout << i << " precision: " << correct / sum << endl;
				}
			}
		}
	}
}