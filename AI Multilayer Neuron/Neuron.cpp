#include "Neuron.h"
#include <iostream>

Neuron::Neuron()
{

}
Neuron::Neuron(float _theta, int _connectionTotal) : m_theta(_theta), m_connectionTotal(_connectionTotal)
{
	m_weights.resize(m_connectionTotal);
	m_inputs.resize(m_connectionTotal);
}
Neuron::~Neuron()
{

}

float Neuron::GetWeight(int _i)
{
	if (_i >= 0 && _i < m_connectionTotal)
	{
		return m_weights.at(_i);
	}
}

float Neuron::GetInput(int _i)
{
	if (_i >= 0 && _i < m_connectionTotal)
	{
		return m_inputs.at(_i);
	}
	return 0;
}

void Neuron::SetAllInputs(float _input[], float _weight[], int _size)
{
	if (_size == m_connectionTotal)
	{
		for (int i = 0; i < _size; i++)
		{
			m_inputs.at(i) = _input[i];
			m_weights.at(i) = _weight[i];
		}
		m_currentConnections = m_connectionTotal;
	}
}

void Neuron::SetTheta(float _theta)
{
	m_theta = _theta;
}

void Neuron::UpdateTheta(float _learningRate)
{
	m_theta += (_learningRate * (-1) * m_errorGradient);
}

void Neuron::SetInput(float _input, int _i)
{
	m_inputs.at(_i) = _input;
}

void Neuron::SetWeight(float _weight, int _i)
{
	m_weights.at(_i) = _weight;
}

void Neuron::UpdateWeight(float _learningRate, int _i)
{
	m_weights.at(_i) += (_learningRate * m_inputs.at(_i) * m_errorGradient);
}

void Neuron::AddInput(float _input, float _weight)
{
	m_connectionTotal++;
	m_weights.resize(m_connectionTotal);
	m_inputs.resize(m_connectionTotal);
	m_weights.at(m_connectionTotal - 1) = _weight;
	m_inputs.at(m_connectionTotal - 1) = _input;
}

float Neuron::GetTotalInputX()
{
	float tmp = 0.0f;
	for (int i = 0; i < m_connectionTotal; i++)
	{
		tmp += m_inputs.at(i) * m_weights.at(i);
	}
	m_totalInputX = tmp;
	return tmp;
}

float Neuron::GetOutputY()
{
	return m_outputY;
}

float Neuron::StepFunc()
{
	m_totalInputX = GetTotalInputX();
	float temp = m_totalInputX - m_theta;
	if (temp > 0.0f)
	{
		return 1.0f;
	}
	else if (abs(temp) <= m_margin)
	{
		m_outputY = 1.0f;
		return 1.0f;
	}
	else
	{
		m_outputY = 0.0f;
		return 0.0f;
	}
}

float Neuron::SignFunc()
{
	m_totalInputX = GetTotalInputX();
	float temp = m_totalInputX - m_theta;
	if (temp > 0.0f)
	{
		m_outputY = 1.0f;
		return 1.0f;
	}
	else if (abs(temp) <= m_margin)
	{
		m_outputY = 1.0f;
		return 1.0f;
	}
	else
	{
		m_outputY = -1.0f;
		return -1.0f;
	}
}

float Neuron::SigmoidFunc()
{
	float tmp;
	m_totalInputX = GetTotalInputX();

	tmp = 1 / (1 + exp(-(m_totalInputX - m_theta)));
	m_outputY = tmp;
	return tmp;
}

float Neuron::LinearFunc()
{
	float tmp;
	m_totalInputX = GetTotalInputX();
	tmp = m_totalInputX - m_theta;
	m_outputY = tmp;
	return tmp;
}

float Neuron::GetError(float _desiredOutput)
{
	m_error = _desiredOutput - SigmoidFunc();
	return m_error;
}

float Neuron::CalculateErrorGradient(float _error)
{
	float tmp = 0.0f;
	tmp = SigmoidFunc() * (1 - SigmoidFunc()) * _error;
	m_errorGradient = tmp;
	return tmp;
}

float Neuron::GetErrorGradient()
{
	return m_errorGradient;
}