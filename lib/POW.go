package lib

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"math"
	"math/rand"
	"time"
)

type InputData struct {
	X mat64.Dense
	Y mat64.Dense
}

func PoWFNN(data InputData, targetError float64) {
	rand.Seed(time.Now().UnixNano())

	// Define the neural network
	nn := NewNeuralNetwork(2, []int{2, 1})

	// Train the neural network
	var error float64
	for {
		nn.Train(&data.X, &data.Y, 0.1)
		error = nn.CalculateError(&data.X, &data.Y)
		fmt.Printf("Error: %f\n", error)

		if error < targetError {
			break
		}
	}

	fmt.Println("PoW completed!")
}

func sigmoid(_, _ int, v float64) float64 {
	return 1.0 / (1.0 + math.Exp(-v))
}

func randomArray(size int, low, high float64) []float64 {
	arr := make([]float64, size)
	for i := range arr {
		arr[i] = rand.Float64()*(high-low) + low
	}
	return arr
}
