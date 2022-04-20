import numpy as np
import matplotlib.pyplot as plt


def recurrent_map(parameters, init_x):
    x = []
    for itr in range(parameters["num_iteration"]):
        init_x = parameters["B"] * np.tanh(parameters["w1"] * init_x) - \
                 parameters["A"] * np.tanh(parameters["w2"] * init_x)
        if itr >= (parameters["num_iteration"] - parameters["num_days"] - 5):  # collect five additional samples
            x.append(init_x)
    return np.array(x).flatten()


def rulkov_map(parameters, init_x, stim, num_itr, control):
    x = np.zeros((num_itr, 2))  # the first column is x and the second one is y
    x[0:2, 0:2] = init_x
    for itr in range(1, num_itr-1):
        g = parameters["g"] + control[itr]
        if x[itr, 1] <= parameters["yth"]:
            e = 0
        else:
            e = 1
        u = parameters["beta"] + stim[itr]
        if x[itr, 0] <= 0:
            f = parameters["alpha"] / (1 - x[itr, 0]) + u
        if (x[itr, 0] > 0) and x[itr, 0] < parameters["alpha"] + u and x[itr - 1, 0] <= 0:
            f = parameters["alpha"] + u
        if (x[itr, 0] > parameters["alpha"] + u) or (x[itr - 1, 0] > 0):
            f = -1
        x[itr + 1, 0] = (1 - e) * f + e * parameters["xp"]
        if (x[itr, 0] > parameters["alpha"] + u) or (x[itr - 1, 0] > 0):
            x[itr + 1, 1] = parameters["ys"]
        else:
            x[itr + 1, 1] = (1 - parameters["mu"]) * x[itr, 1] - g * x[itr, 1] * (1 - x[itr, 1]) ** 2
    return x


def square_pulse_generator(err, num_simulation_days, simulation_mode):
    err = err[5:]
    if simulation_mode == "constant":
        k = [1.1 - np.tanh(np.sum(err) / err.shape[0]) / 10, np.tanh(np.sum(err) / err.shape[0])]
        start_time = np.repeat(16 - 5 * k[1], num_simulation_days).reshape(-1, 1)
        end_time = np.repeat(24 + 5 * k[1], num_simulation_days).reshape(-1, 1)
        pulse = k[0] * np.ones((num_simulation_days * 24, 1))
    else:
        start_time = 16 - 5 * np.tanh(err.reshape(-1, 1))
        end_time = 24 + 5 * np.tanh(err.reshape(-1, 1))
        pulse = np.repeat(1.1 - np.tanh(err.reshape(-1, 1) / 10), 24).reshape(-1, 1) * \
                np.ones((num_simulation_days * 24, 1))

    for day_index in range(0, num_simulation_days):
        pulse[(day_index * 24 + start_time[day_index, 0]).astype(int):
              (day_index * 24 + end_time[day_index, 0]).astype(int)] = 0
    return pulse[4:]


def error_to_stimuli(err, stim_amp, cff):
    stim_inp = np.zeros((240 * err.shape[0], 1))
    cnt = 0
    for stim_time in np.arange(0, stim_inp.shape[0], 240):
        stim_inp[int(stim_time + 10 * cff * err[cnt, 0])] = stim_amp
        cnt += 1
    return stim_inp


def error_to_control_parameter(err, cff):
    control_parameter = np.zeros((240 * err.shape[0], 1))
    cnt = 0
    for stim_time in np.arange(0, control_parameter.shape[0], 240):
        control_parameter[int(stim_time): 240 + int(stim_time)] = \
            cff * err[cnt, 0]
        cnt += 1
    return control_parameter


def rulkov_pulse(sig, err, cff):
    cnt = 0
    sig = sig.reshape(-1, 1)
    scale = np.zeros((240 * err.shape[0], 1))
    for stim_time in np.arange(0, sig.shape[0], 240):
        scale[int(stim_time): 240 + int(stim_time)] = 1.1 - 10 * cff * err[cnt, 0]
        cnt += 1
    return np.multiply(scale, sig)


def pulse_generation(map_param, rulkov_param):
    time_series = recurrent_map(map_param, np.random.rand(1)).reshape(-1, 1)
    coeff = (map_param["A"] /map_param["A_normal"] - 1) * np.sign(map_param["A"] /map_param["A_normal"] - 1) -\
            (map_param["B"] /map_param["B_normal"] - 1) * np.sign(map_param["B"] /map_param["B_normal"] - 1)
    Period4_error = np.abs(time_series - np.roll(time_series, 4))
    delta = np.tanh(Period4_error[5:]).reshape(-1, 1)
    gamma = np.tanh(0.1 * Period4_error[5:]).reshape(-1, 1)
    inp_signal = error_to_stimuli(delta, param_Rulkov_model["Ap"], coeff)
    variable_g = error_to_control_parameter(delta, coeff)
    ts = rulkov_map(rulkov_param, np.array(([-0.8, 0.0])), inp_signal, 240 * map_param["num_days"], variable_g)
    out = rulkov_pulse(ts[:, 1], gamma, coeff)
    return out


np.random.seed(283)
map_param_normal = {"w1": 1.487, "w2": 0.2223, "B_normal": 5.821, "A_normal": 14.47, "A": 14.87, "B": 5.82,
             "num_iteration": 4000, "num_days": 80}
map_param_depression = {"w1": 1.487, "w2": 0.2223, "B_normal": 5.821, "A_normal": 14.47, "A": 16.47, "B": 5.82,
             "num_iteration": 4000, "num_days": 80}
map_param_mania = {"w1": 1.487, "w2": 0.2223, "B_normal": 5.821, "A_normal": 14.47, "A": 12.47, "B": 7.65,
             "num_iteration": 4000, "num_days": 80}
param_Rulkov_model = {"xp": -0.8, "yth": 0.01, "alpha": 3.2, "beta": -2.5780, "ys": 1.3, "mu": 0.002,
                      "g": 0.3, "Ap": 3.0}
#
t = np.arange(0, 240 * map_param_normal["num_days"]) / 240
ts_normal = pulse_generation(map_param_normal, param_Rulkov_model)
ts_depression = pulse_generation(map_param_depression, param_Rulkov_model)
ts_mania = pulse_generation(map_param_mania, param_Rulkov_model)

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(t, ts_depression, "#006685")
plt.xlim(0, map_param_normal["num_days"])
plt.ylim(0, 2.7)
plt.xticks(np.arange(0, map_param_normal["num_days"], 10))
plt.title("Depressive")
plt.ylabel("Simulated activity")

plt.subplot(3, 1, 2)
plt.plot(t, ts_normal, "#FFE48D")
plt.xlim(0, map_param_normal["num_days"])
plt.ylim(0, 2.7)
plt.xticks(np.arange(0, map_param_normal["num_days"], 10))
plt.title("Normal")
plt.ylabel("Simulated activity")

plt.subplot(3, 1, 3)
plt.plot(t, ts_mania, "#BF003F")
plt.xlim(0, map_param_normal["num_days"])
plt.ylim(0, 2.7)
plt.xticks(np.arange(0, map_param_normal["num_days"], 10))
plt.title("Manic")
plt.ylabel("Simulated activity")
plt.xlabel("Time (day)")

plt.tight_layout()
plt.show()

