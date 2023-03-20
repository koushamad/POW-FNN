package lib

import "github.com/gonum/matrix/mat64"

type Layer struct {
	Weights *mat64.Dense
	Biases  *mat64.Dense
}

func NewLayer(inputSize, outputSize int) *Layer {
	return &Layer{
		Weights: mat64.NewDense(outputSize, inputSize, randomArray(outputSize*inputSize, -1, 1)),
		Biases:  mat64.NewDense(outputSize, 1, randomArray(outputSize, -1, 1)),
	}
}

func (l *Layer) Feedforward(input *mat64.Dense) *mat64.Dense {
	z := &mat64.Dense{}
	z.Mul(l.Weights, input.T())
	z.Add(z, l.Biases)
	a := &mat64.Dense{}
	a.Apply(sigmoid, z)
	a.Clone(a.T())

	return a
}

func (l *Layer) UpdateWeights(input, delta *mat64.Dense, learningRate float64) {
	gradient := &mat64.Dense{}
	gradient.Mul(delta, input)
	gradient.Scale(learningRate, gradient)
	l.Weights.Add(l.Weights, gradient)

	l.Biases.Add(l.Biases, delta)
}
