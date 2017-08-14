#pragma once
#include <iostream>  
#include <fstream>  
#include <string>  
#include <vector>  

int ReverseInt(int i);

void read_Mnist_Label(std::string filename, std::vector<double>&labels);

void read_Mnist_Images(std::string filename, std::vector<std::vector<double> >&images);