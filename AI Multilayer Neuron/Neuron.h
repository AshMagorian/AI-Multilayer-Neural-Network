#include <cmath>
#include <vector>

class Neuron
{
public:

	Neuron();
	Neuron(float _theta, int _connectionTotal);
	~Neuron();

	float GetWeight(int _i);
	float GetInput(int _i);

	void SetAllInputs(float _input[], float _weight[], int _size);
	void SetTheta(float _theta);
	void UpdateTheta(float _learningRate);

	void SetInput(float _input, int _i);
	void SetWeight(float _weight, int _i);
	void UpdateWeight(float _learningRate, int _i);

	void AddInput(float _input, float _weight);

	float GetTotalInputX();
	float GetOutputY();

	float StepFunc();
	float SignFunc();
	float SigmoidFunc();
	float LinearFunc();

	float GetError(float _desiredOutput);

	float CalculateErrorGradient(float _error);
	float GetErrorGradient();

private:
	std::vector<float> m_weights;
	std::vector<float> m_inputs;
	float m_totalInputX = 0;
	float m_outputY;
	float m_theta;
	float m_margin = 0.0000001f;
	int m_connectionTotal = 0;
	int m_currentConnections = 0;

	float m_error;
	float m_errorGradient;
};