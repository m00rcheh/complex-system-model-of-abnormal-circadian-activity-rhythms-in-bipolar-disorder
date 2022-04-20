import numpy as np
import matplotlib.pyplot as plt


def recurrent_map(parameters, init_x):
    x = []
    for itr in range(parameters["num_iteration"]):
        init_x = parameters["B"] * np.tanh(parameters["w1"] * init_x) - \
                 parameters["A"] * np.tanh(parameters["w2"] * init_x)
        if itr >= (parameters["num_iteration"] - parameters["num_attractor_sample"]):
            x.append(init_x)
    return np.array(x).flatten()


def bifurcation(rmap, parameters, bifurcation_parameter, parameter_range, method):
    np.random.seed(283)
    x_init = np.random.rand(1)

    if method == "forward":
        parameters["B"] = 5.821
        bifurcation_parameter_values = np.linspace(parameter_range[0], parameter_range[1], 8000).reshape(-1, 1)
        bifurcation_matrix = np.zeros((2 * parameters["num_attractor_sample"], bifurcation_parameter_values.shape[0]))
        for count, value in enumerate(bifurcation_parameter_values):
            parameters.update({bifurcation_parameter: value})
            bifurcation_matrix[:param["num_attractor_sample"], count] = rmap(param, x_init)
            bifurcation_matrix[param["num_attractor_sample"]:, count] = rmap(param, -x_init)
            x_init = bifurcation_matrix[-1, count]
    if method == "reverse":
        parameters["A"] = 12.47
        bifurcation_parameter_values = np.linspace(parameter_range[1], parameter_range[0], 8000).reshape(-1, 1)
        bifurcation_matrix = np.zeros((2 * parameters["num_attractor_sample"], bifurcation_parameter_values.shape[0]))
        for count, value in enumerate(bifurcation_parameter_values):
            parameters.update({bifurcation_parameter: value})
            bifurcation_matrix[:param["num_attractor_sample"], count] = rmap(param, x_init)
            bifurcation_matrix[param["num_attractor_sample"]:, count] = rmap(param, -x_init)
            x_init = bifurcation_matrix[-1, count]
        new_data = np.zeros(bifurcation_matrix.shape)
        for i in range(bifurcation_matrix.shape[1]):
            new_data[:, i] = bifurcation_matrix[:, bifurcation_matrix.shape[1] - 1 - i]
        bifurcation_matrix = new_data
    return bifurcation_matrix


param = {"w1": 1.487, "w2": 0.2223, "B": 5.821, "num_iteration": 4000, "num_attractor_sample": 100, "A": 12.47}
Bifurcation_matrix_A = bifurcation(recurrent_map, param, "A", [0, 40], "forward")
Bifurcation_matrix_B = bifurcation(recurrent_map, param, "B", [0, 20], "reverse")

np.savetxt('Bifurcation_matrix_A.csv', Bifurcation_matrix_A, delimiter=',')
np.savetxt('Bifurcation_matrix_B.csv', Bifurcation_matrix_B, delimiter=',')

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(np.linspace(0, 40, 8000).reshape(-1, 1), Bifurcation_matrix_A.T, '.', color="#006685", markersize=0.01)
plt.xlim((0, 40))
plt.ylim((-6.0, 6.0))
plt.xlabel("A")
plt.ylabel("$X_{inf}$")

plt.subplot(2, 1, 2)
plt.plot(np.linspace(0, 20, 8000).reshape(-1, 1), Bifurcation_matrix_B.T, '.', color="#BF003F", markersize=0.01)
plt.xlim((0, 20))
plt.ylim((-10.0, 10.0))
plt.xlabel("B")
plt.ylabel("$X_{inf}$")

plt.show()
