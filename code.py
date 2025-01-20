import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = pd.read_csv("linearX.csv").values
Y = pd.read_csv("linearY.csv").values
X = (X - np.mean(X)) / np.std(X)
Y = (Y - np.mean(Y)) / np.std(Y)
X = np.c_[np.ones(X.shape[0]), X]


def gradient_descent(x, y, alp, iteration, convergence_criteria=1e-6):
    m = len(y)
    theta1 = np.zeros(x.shape[1])
    cost_history1 = []
    for i in range(iteration):
        predictions = x.dot(theta1)
        errors = predictions - y.flatten()
        cost = (1 / (2 * m)) * np.sum(errors ** 2)
        cost_history1.append(cost)
        gradient = (1 / m) * x.T.dot(errors)
        theta1 -= alp * gradient
        if i > 0 and abs(cost_history1[-2] - cost_history1[-1]) < convergence_criteria:
            break
    return theta1, cost_history1


alpha = 0.5
iterations = 1000
theta, cost_history = gradient_descent(X, Y, alpha, iterations)
print("Theta (parameters):", theta)
print("Final cost:", cost_history[-1])
iterations = len(cost_history[:50])
plt.plot(range(iterations), cost_history[:iterations])
plt.title("Cost Function vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()
plt.scatter(X[:, 1], Y, color='blue', label='Dataset')
plt.plot(X[:, 1], X.dot(theta), color='red', label='Regression Line')
plt.title("Dataset and Regression Line")
plt.xlabel("Predictor Variable")
plt.ylabel("Response Variable")
plt.legend()
plt.show()
learning_rates = [0.005, 0.5, 5]
for lr in learning_rates:
    _, cost_hist = gradient_descent(X, Y, lr, 50)
    plt.plot(range(len(cost_hist)), cost_hist, label=f"lr={lr}")
plt.title("Cost Function vs Iterations for Different Learning Rates")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.legend()
plt.show()


def stochastic_gradient_descent(x, y, alp, iteration):
    m = len(y)
    theta2 = np.zeros(x.shape[1])
    cost_history2 = []
    for i in range(iteration):
        for j in range(m):
            rand_idx = np.random.randint(0, m)
            x_sample = x[rand_idx, :].reshape(1, -1)
            y_sample = y[rand_idx]

            prediction = x_sample.dot(theta2)
            error = prediction - y_sample
            theta2 -= alp * error * x_sample.flatten()
        cost = (1 / (2 * m)) * np.sum((X.dot(theta2) - y.flatten()) ** 2)
        cost_history2.append(cost)
    return theta, cost_history


def mini_batch_gradient_descent(x, y, alp, iteration, batch_size):
    m = len(y)
    theta3 = np.zeros(x.shape[1])
    cost_history3 = []
    for i in range(iteration):
        for j in range(0, m, batch_size):
            x_batch = x[j:j + batch_size, :]
            y_batch = y[j:j + batch_size]
            predictions = x_batch.dot(theta3)
            errors = predictions - y_batch.flatten()
            theta3 -= alp * (1 / batch_size) * x_batch.T.dot(errors)
        cost = (1 / (2 * m)) * np.sum((x.dot(theta3) - y.flatten()) ** 2)
        cost_history3.append(cost)
    return theta3, cost_history3


theta_sgd, cost_sgd = stochastic_gradient_descent(X, Y, alp=0.05, iteration=50)
theta_mb, cost_mb = mini_batch_gradient_descent(X, Y, alp=0.05, iteration=50, batch_size=16)
plt.plot(range(len(cost_history)), cost_history, label="Batch Gradient Descent")
plt.plot(range(len(cost_sgd)), cost_sgd, label="Stochastic Gradient Descent")
plt.plot(range(len(cost_mb)), cost_mb, label="Mini-Batch Gradient Descent")
plt.title("Cost Function vs Iterations for Different Methods")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.legend()
plt.show()
