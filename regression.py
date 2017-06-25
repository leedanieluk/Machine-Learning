import random
import matplotlib.pyplot as plt

def linearRegression(x_data, y_data, max_reps, l_rate, min_delta_error):
    par1 = random.uniform(-2, 2)
    par2 = random.uniform(-2, 2)
    old_cost = 0
    cost_data = []
    # print("Init Par 1: " + str(par1))
    # print("Init Par 2: " + str(par2))

    for rep in range(max_reps):
        grad1 = 0
        grad2 = 0
        cost = 0

        for i in range(len(x_data)):
            hypothesis = par1 + par2 * x_data[i]
            cost = cost + ((hypothesis - y_data[i]) ** 2) / 2

        cost = cost / len(x_data)
        cost_data.extend([cost])

        if abs(cost - old_cost) < min_delta_error:
            # print("Rep " + str(rep) + ") " + "Cost: " + str(cost))
            # print("Rep " + str(rep) + ") Par 1: " + str(par1))
            # print("Rep " + str(rep) + ") Par 2: " + str(par2))
            return par1, par2, cost_data
            break

        old_cost = cost
        
        # print("Rep " + str(rep) + ") " + "Cost: " + str(cost))

        for i in range(len(x_data)):
            hypothesis = par1 + par2 * x_data[i]
            error = hypothesis - y_data[i]
            grad1 = grad1 + error
            grad2 = grad2 + error * x_data[i]

        grad1 = grad1 / len(x_data)
        grad2 = grad2 / len(x_data)

        par1 = par1 - l_rate * grad1
        par2 = par2 - l_rate * grad2
    
    # print("Rep " + str(rep) + ") Par 1: " + str(par1))
    # print("Rep " + str(rep) + ") Par 2: " + str(par2))
    return par1, par2, cost_data

datasize = 100
x_data = []
y_data = []
counter = 0

for data in range(datasize):
    x_data.extend([data + 1])
    y_data.extend([random.uniform(-5, 5)])

max_reps = 2000
l_rate = 0.0001
min_delta_error = 0.01

par1, par2, cost_data = linearRegression(x_data, y_data, max_reps, l_rate, min_delta_error)

cost_x = []
for counter in range(len(cost_data)):
    cost_x.extend([counter])

plt.figure("Cost")
plt.plot(cost_x, cost_data)
plt.ylabel("Cost J")
plt.xlabel("Iteration")
plt.title("Cost Function over Iteration")

plt.figure("Linear Regression Prediction")
plt.plot(x_data, y_data, 'ro')

x_init = min(x_data) - 1
y_init = par1 + par2 * x_init
x_end = max(x_data) + 1
y_end = par1 + par2 * x_end

plt.plot([x_init, x_end], [y_init, y_end])
plt.axis([0, datasize + 1, -10, 10])
plt.ylabel("y")
plt.xlabel("x")
plt.title("Data and Linear Regression Prediction")
plt.show()

