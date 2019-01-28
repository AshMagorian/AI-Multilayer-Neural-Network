#include <iostream>
#include <cmath>
#include <vector>

#include "Neuron.h"

int main()
{
	float X, Y, learningRate, wCurrent35, wCurrent45;
	float inputSignal[6][6], inputWeight[6][6];
	int iterations = 100000;
	std::vector<float> theta;

	learningRate = 0.1f;

	inputWeight[1][3] = 0.5f; inputWeight[2][3] = 0.4f; // Sets the weights as arbitrary values
	inputWeight[1][4] = 0.9f; inputWeight[2][4] = 1.0f;
	inputWeight[3][5] = -1.2f; inputWeight[4][5] = 1.1f;

	theta.resize(6);
	theta[3] = 0.8f; theta[4] = -0.1f; theta[5] = 0.3f; // Sets the threshhold as arbitrary values

	Neuron* neuron3 = new Neuron(theta[3], 2); // Create the neuron objects (neurons 1 and 2 are input layer neurons)
	Neuron* neuron4 = new Neuron(theta[4], 2);
	Neuron* neuron5 = new Neuron(theta[5], 2);

	neuron3->SetWeight(inputWeight[1][3], 0); 	neuron3->SetWeight(inputWeight[2][3], 1);
	neuron4->SetWeight(inputWeight[1][4], 0); 	neuron4->SetWeight(inputWeight[2][4], 1);
	neuron5->SetWeight(inputWeight[3][5], 0); 	neuron5->SetWeight(inputWeight[4][5], 1);

	std::vector<float> tx1;
	std::vector<float> tx2;
	std::vector<float> tYd5;
	tx1.resize(4);
	tx2.resize(4);
	tYd5.resize(4);
	tx1.at(0) = 1.0f; tx2.at(0) = 1.0f; tYd5.at(0) = 0.0f; // Setting up training data
	tx1.at(1) = 0.0f; tx2.at(1) = 1.0f; tYd5.at(1) = 1.0f;
	tx1.at(2) = 1.0f; tx2.at(2) = 0.0f; tYd5.at(2) = 1.0f;
	tx1.at(3) = 0.0f; tx2.at(3) = 0.0f; tYd5.at(3) = 0.0f;
	std::cout << "Training Data\n\n";
	std::cout << "Input1 = 1, Input2 = 1, Output = 0" << std::endl;
	std::cout << "Input1 = 0, Input2 = 1, Output = 1" << std::endl;
	std::cout << "Input1 = 1, Input2 = 0, Output = 1" << std::endl;
	std::cout << "Input1 = 0, Input2 = 0, Output = 0" << std::endl;

	std::cout << "\nCalculating final weights using a multilayer neuron system of 5 neurons\n\n";
	std::cout << "Structure: \n\n";

	std::cout << "1-3" << std::endl;
	std::cout << " X >5" << std::endl;
	std::cout << "2-4 \n" << std::endl;

	int p = 0;
	while (p <= iterations - 4)
	{
		float EpocSumError = 0.0f;
		int i = p % 4; //index for the training data 0,1,2,3,0,1,2,3,...
		
		//std::cout << "\nIteration No " << p + 1 << "\n" << i + 1 << "/4\n" ; // print iteration number

		neuron3->SetInput(tx1.at(i), 0); //Feeding forward the input values through neuron 3
		neuron3->SetInput(tx2.at(i), 1); 
		neuron3->GetTotalInputX();  
		neuron3->SigmoidFunc();

		neuron4->SetInput(tx1.at(i), 0);//Feeding forward the input values through neuron 4
		neuron4->SetInput(tx2.at(i), 1);
		neuron4->GetTotalInputX();
		neuron4->SigmoidFunc(); 

		neuron5->SetInput(neuron3->GetOutputY(), 0); //Feeding forward the output of neurons 3 and 4 through neuron 5
		neuron5->SetInput(neuron4->GetOutputY(), 1); 
		neuron5->GetTotalInputX(); 
		neuron5->SigmoidFunc(); 

		neuron5->CalculateErrorGradient(neuron5->GetError(tYd5.at(i))); // Calculating error gradient of neuron 5
		wCurrent35 = neuron5->GetWeight(0); wCurrent45 = neuron5->GetWeight(1); // save the current values before updating them
		neuron5->UpdateWeight(learningRate, 0); 
		neuron5->UpdateWeight(learningRate, 1);
		neuron5->UpdateTheta(learningRate); 

		neuron3->CalculateErrorGradient(wCurrent35 * neuron5->GetErrorGradient()); // Calculating error gradient of neuron 3
		neuron3->UpdateWeight(learningRate, 0); 
		neuron3->UpdateWeight(learningRate, 1); 
		neuron3->UpdateTheta(learningRate);

		neuron4->CalculateErrorGradient(wCurrent45 * neuron5->GetErrorGradient()); // Calculating error gradient of neuron 4
		neuron4->UpdateWeight(learningRate, 0); 
		neuron4->UpdateWeight(learningRate, 1); 
		neuron4->UpdateTheta(learningRate);

		neuron3->GetTotalInputX();
		neuron3->GetOutputY();
		neuron4->GetTotalInputX();
		neuron4->GetOutputY();
		neuron5->SetInput(neuron3->GetOutputY(), 0);
		neuron5->SetInput(neuron4->GetOutputY(), 1);
		neuron5->GetTotalInputX();
		neuron5->GetOutputY();

		//std::cout << "tyd5 = " << tYd5.at(i) << "\t Y5 = " << neuron5->GetOutputY() << "\n";

		EpocSumError += pow(tYd5.at(i) - neuron5->GetOutputY(), 2);

		//std::cout << "epoc = " << EpocSumError << "\n";

		p++;

		if (EpocSumError < 0.01)
		{
			break;
		}
	}

	std::cout << "Results found after " << p + 1 << " iterations!\n\n";

	std::cout << "Final wight connecting neurons 3 and 5 = " << neuron5->GetWeight(0) << std::endl;
	std::cout << "Final wight connecting neurons 4 and 5 = " << neuron5->GetWeight(1) << std::endl;
	std::cout << "\nFinal wight connecting neurons 1 and 4 = " << neuron4->GetWeight(0) << std::endl;
	std::cout << "Final wight connecting neurons 2 and 4 = " << neuron4->GetWeight(1) << std::endl;
	std::cout << "\nFinal wight connecting neurons 1 and 3 = " << neuron3->GetWeight(0) << std::endl;
	std::cout << "Final wight connecting neurons 2 and 3 = " << neuron3->GetWeight(1) << std::endl;

	std::cin.get();

	delete neuron3;
	delete neuron4;
	delete neuron5;

	return 0;
}