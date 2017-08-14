# ordinary_net
Build my own deep learning tools.

This program now achieve the following functions of the deep neural network:
1. dynamically create network structures based on instructions
2. automatically save the .save file every 100k steps
3. automatic test every 100k steps while training
4. supports reading save files for testing
5. supports reading save files and continues training

The network structure currently supported are only full connection layer and softmax layer.
The activation function uses sigmoid.
Train and test sets are MNIST.

To run:

——File compilation:

		Training: g++ train.cpp net.cpp loadDataset.cpp DeepLearningTools.cpp -std=c++11 train -o etc.
		Testing: g++ evaluation.cpp net.cpp loadDataset.cpp DeepLearningTools.cpp -std=c++11 -o evaluation
		
——Program operation:

		The two programs have wizards. Follow the wizard.
		There are two training files in the saved folder:
				1. only one softmax layer network, the accuracy rate of 92.23%.
				2. a neural network consisting of 2 layers of +1 softmax is fully connected, and the accuracy is 95.03%.
				
——Test environment:

		Ubuntu1604(64)
		G++ version 4.9.4
