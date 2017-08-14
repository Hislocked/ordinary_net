#include <iostream>
#include <vector>
#include "net.h"
#include "loadDataset.h"
using namespace std;

int layers = 1;
vector<vector<int>> nSNum;
vector<string> nS;
string saveFile;

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

void help() {
	cout << "This program is not a complete version now." << endl;
	cout << "You can use some functionality." << endl;
	cout << "Please follow the instructions below:" << endl;
}

void instructions() {
	cout << "Please enter save file name with path:" << endl;
	getline(cin, saveFile);
	if (saveFile.size() == 0) {
		cout << "If you dont have a save file,please use program train to get one." << endl;
	}
}

int main(int argc, char *argv[]) {
	if (argc > 1)	help();
	instructions();
	vector<double>labels;
	read_Mnist_Label("dataset/t10k-labels.idx1-ubyte", labels);
	vector<vector<double> > images;
	read_Mnist_Images("dataset/t10k-images.idx3-ubyte", images);
	for (size_t i = 0; i < images.size(); i++)
	{
		for (size_t j = 0; j < images[i].size(); j++)
		{
			images[i][j] /= 255;
		}
	}
	Net netForMnist(layers, nS, nSNum);
	if (!netForMnist.loadWeight(saveFile))
		return 0;
	double sum = 0, correct = 0;

	for (size_t i = 1; i <= images.size(); i++)
	{
		vector<vector<double>> input(1, images[i - 1]);
		//std::cout << std::endl;
		//for (size_t i = 1; i <= input[0].size(); i++)
		//{
		//	printf("%1.2f ", input[0][i - 1]);
		//	if (i % 28 == 0)
		//		std::cout << std::endl;
		//}
		vector<vector<double>> logits = netForMnist.forwardPropagation(input);
		//for (size_t i = 0; i < logits[0].size(); i++)
		//{
		//	cout << logits[0][i] << " ";
		//}
		//cout << endl;
		int prediction = top_1(logits);
		if (prediction == labels[i - 1])
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