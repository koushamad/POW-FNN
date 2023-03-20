package lib

import "github.com/gonum/matrix/mat64"

type NeuralNetwork struct {
	Layers []*Layer
}

func NewNeuralNetwork(inputSize int, layerSizes []int) *NeuralNetwork {
	nn := &NeuralNetwork{}
	prevSize := inputSize

	for _, size := range layerSizes {
		nn.Layers = append(nn.Layers, NewLayer(prevSize, size))
		prevSize = size
	}

	return nn
}

func (nn *NeuralNetwork) Train(X, Y *mat64.Dense, learningRate float64) {
	// Feedforward
	activations := nn.Feedforward(X)

	// Backpropagation
	errors := []*mat64.Dense{}
	delta := &mat64.Dense{}
	delta.Sub(Y, activations[len(activations)-1])
	errors = append(errors, delta)

	for i := len(nn.Layers) - 1; i > 0; i-- {
		layer := nn.Layers[i]
		delta = &mat64.Dense{}
		delta.Mul(errors[0], layer.Weights.T())
		errors = append([]*mat64.Dense{delta}, errors...)
	}

	// Update weights
	for i, layer := range nn.Layers {
		layer.UpdateWeights(activations[i], errors[i], learningRate)
	}
}

func (nn *NeuralNetwork) Feedforward(X *mat64.Dense) []*mat64.Dense {
	activations := []*mat64.Dense{X}

	for _, layer := range nn.Layers {
		activations = append(activations, layer.Feedforward(activations[len(activations)-1]))
	}

	return activations
}

func (nn *NeuralNetwork) CalculateError(X, Y *mat64.Dense) float64 {
	activations := nn.Feedforward(X)
	predictions := activations[len(activations)-1]

	var error float64
	rows, _ := predictions.Dims()
	for i := 0; i < rows; i++ {
		error += 0.5 * (predictions.At(i, 0) - Y.At(i, 0)) * (predictions.At(i, 0) - Y.At(i, 0))
	}
	return error / float64(rows)
}
