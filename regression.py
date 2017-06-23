# regression (x_data, y_data, nth_order) -> (m, b)
# plot (x_start, x_end, x_steps) -> (graph)
import random
import matplotlib.pyplot as plt

def linearRegression(x_data, y_data, reps, l_rate):
    par1 = 0
    par2 = 0
    print("Init Par1: " + str(par1))
    print("Init Par2: " + str(par2))

    for rep in range(reps):
        grad1 = 0
        grad2 = 0
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

x_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
y_data = [1, 4, 2, 6, 2, 5, 3, 20, 1, 2, 3, 1, 5]
reps = 2000
l_rate = 0.001

par1, par2 = linearRegression(x_data, y_data, reps, l_rate)

plt.plot(x_data, y_data, 'ro')

x_init = min(x_data) - 1
y_init = par1 + par2 * x_init
x_end = max(x_data) + 1
y_end = par1 + par2 * x_end

plt.plot([x_init, x_end], [y_init, y_end])
plt.show()
