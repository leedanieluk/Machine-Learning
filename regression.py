# regression (x_data, y_data, nth_order) -> (m, b)
# plot (x_start, x_end, x_steps) -> (graph)
import random
import matplotlib.pyplot as plt

def linearRegression(x_data, y_data, reps, l_rate):
    par1 = random.uniform(0, 1)
    par2 = random.uniform(0, 1)
    print("Init Par1: " + str(par1))
    print("Init Par2: " + str(par2))

    for rep in range(reps):
        grad1 = 0
        grad2 = 0
        cost = 0

        for i in range(len(x_data)):
            hypothesis = par1 + par2 * x_data[i]
            cost = cost + ((hypothesis - y_data[i]) ** 2) / 2

        cost = cost / len(x_data)
        
        print("Cost: " + str(cost))

        for i in range(len(x_data)):
            hypothesis = par1 + par2 * x_data[i]
            error = hypothesis - y_data[i]
            grad1 = grad1 + error
            grad2 = grad2 + error * x_data[i]

        grad1 = grad1 / len(x_data)
        grad2 = grad2 / len(x_data)

        par1 = par1 - l_rate * grad1
        par2 = par2 - l_rate * grad2

    print("Final Par1: " + str(par1))
    print("Final Par2: " + str(par2))
    return par1, par2

datasize = 20
x_data = []
y_data = []
counter = 0

for data in range(datasize):
    counter = counter + 1
    x_data.extend([counter])
    y_data.extend([counter + random.uniform(-30, 30)])

reps = 10
l_rate = 0.01

par1, par2 = linearRegression(x_data, y_data, reps, l_rate)

plt.plot(x_data, y_data, 'ro')

x_init = min(x_data) - 1
y_init = par1 + par2 * x_init
x_end = max(x_data) + 1
y_end = par1 + par2 * x_end

plt.plot([x_init, x_end], [y_init, y_end])
plt.show()
