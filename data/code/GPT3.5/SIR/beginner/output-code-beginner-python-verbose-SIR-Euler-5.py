import numpy as np
import matplotlib.pyplot as plt


def sir_model_euler(infectious_rate, recovery_rate, initial_population, initial_infected, time_step, total_time):
    time = np.arange(0, total_time, time_step)
    s = np.zeros_like(time)
    i = np.zeros_like(time)
    r = np.zeros_like(time)
    s[0] = initial_population - initial_infected
    i[0] = initial_infected
    r[0] = 0

    for t in range(1, len(time)):
        ds_dt = -infectious_rate * s[t - 1] * i[t - 1] / initial_population
        di_dt = (infectious_rate * s[t - 1] * i[t - 1] / initial_population) - recovery_rate * i[t - 1]
        dr_dt = recovery_rate * i[t - 1]

        s[t] = s[t - 1] + ds_dt * time_step
        i[t] = i[t - 1] + di_dt * time_step
        r[t] = r[t - 1] + dr_dt * time_step

    return time, s, i, r


# Example usage
infectious_rate = 0.3
recovery_rate = 0.1
initial_population = 1000
initial_infected = 10
time_step = 0.1
total_time = 100

time, s, i, r = sir_model_euler(infectious_rate, recovery_rate, initial_population, initial_infected, time_step, total_time)

plt.plot(time, s, label='Susceptible')
plt.plot(time, i, label='Infected')
plt.plot(time, r, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using Euler Method')
plt.legend()
plt.grid(True)
plt.show()
