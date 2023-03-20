package main

import "github.com/gonum/matrix/mat64"
import "github.com/koushamad/POW-FNN/lib"

func main() {
	// Example input data
	data := lib.InputData{
		X: *mat64.NewDense(4, 2, []float64{0, 0, 0, 1, 1, 0, 1, 1}),
		Y: *mat64.NewDense(4, 1, []float64{0, 1, 1, 0}),
	}

	// PoW with FNN
	targetError := 0.01
	lib.PoWFNN(data, targetError)
}
