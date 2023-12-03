##Neural Network for Heart Disease Prediction



This repository contains a simple implementation of a neural network for heart disease prediction using Python. The neural network is implemented from scratch, and the code provides a basic understanding of forward and backward propagation.

Getting Started
To get started with the project, follow these steps:

Prerequisites
Make sure you have the following installed:

Python (>=3.6)
Jupyter Notebook (optional, for running the provided code)
Installation
Clone the repository to your local machine:

bash
Copy code
git clone https://github.com/your-username/heart-disease-prediction.git
Usage
Open the Jupyter Notebook file (heart_disease_prediction.ipynb) using Jupyter Notebook.

Run the cells one by one to see the implementation and results.

Adjust hyperparameters, such as learning rate (LR) and the number of iterations (ITERATIONS), as needed.

python
Copy code
# Hyperparameters
LR = 0.1
ITERATIONS = 1000

# Initialize and train the neural network
nn = NN(LR=LR, X_train=x_train_scale, y_train=y_train, X_test=x_test_scale, y_test=y_test)
nn.train(iteration=ITERATIONS)
Explore the results, including the training and test error plots.
Code Structure
heart_disease_prediction.ipynb: Jupyter Notebook file containing the code.
heart.csv: Dataset file (make sure it's in the same directory as the notebook).
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
The code is a basic implementation for educational purposes and may not be suitable for production use.
Feel free to explore, modify, and use this code as a starting point for your own projects. If you encounter any issues or have suggestions, please open an issue.

Happy coding!
Saeid Khoobdel
