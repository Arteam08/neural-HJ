# Neural Network Solver for Hamilton-Jacobi Equations

This project implements a neural network-based approach to solve Hamilton-Jacobi equations. The implementation includes both traditional numerical solvers and deep learning models to compare and analyze solutions.

## Project Structure

- `Data_generation/`: Scripts for generating training and testing data
- `Models/`: Neural network model implementations
- `solvers/`: Traditional numerical solvers for Hamilton-Jacobi equations
- `Tests/`: Test cases and validation scripts
- `Helper/`: Utility functions and helper modules

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd hj-scheme
```

2. Install the package in development mode:
```bash
pip install -e .
```

## Usage

The main components of the project can be used as follows:

1. Generate training data using the Data_generation module
2. Train the neural network models using the provided training scripts
3. Compare results with traditional numerical solvers
4. Visualize and analyze the solutions

## Requirements

The project requires Python 3.7+ and the following main dependencies:
- PyTorch
- NumPy
- Matplotlib
- SciPy

For a complete list of dependencies, refer to `setup.py`.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 