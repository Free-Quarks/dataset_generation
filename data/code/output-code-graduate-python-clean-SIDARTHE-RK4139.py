import numpy as np
import matplotlib.pyplot as plt

# Function to implement the SIDARTHE model using RK4

def sidarthe_model(R0, t_inc, t_inf, t_ser, t_hos, t_mor, t_imm, N, I0, S0, A0, D0, R0, T):
    def derivs(y, t, R0, t_inc, t_inf, t_ser, t_hos, t_mor, t_imm):
        S, I, D, A, R = y
        dSdt = -R0 * S * (I + A) / N
        dIdt = R0 * S * (I + A) / N - I / t_inf
        dDdt = I * (1 / t_inf) * (1 / t_ser) - D / t_hos
        dAdt = I * (1 / t_inf) * (1 - (1 / t_ser)) - A / t_imm
        dRdt = R / t_imm
        return dSdt, dIdt, dDdt, dAdt, dRdt

    # Initial conditions
    y0 = S0, I0, D0, A0, R0
    t = np.linspace(0, T, T)

    # Solve the ODE using RK4
    y = np.zeros((T, 5))
    y[0] = y0
    for i in range(1, T):
        t_i = t[i - 1]
        dt = t[i] - t[i - 1]
        dy1 = derivs(y[i - 1], t_i, R0, t_inc, t_inf, t_ser, t_hos, t_mor, t_imm)
        dy2 = derivs(y[i - 1] + dt / 2 * dy1, t_i + dt / 2, R0, t_inc, t_inf, t_ser, t_hos, t_mor, t_imm)
        dy3 = derivs(y[i - 1] + dt / 2 * dy2, t_i + dt / 2, R0, t_inc, t_inf, t_ser, t_hos, t_mor, t_imm)
        dy4 = derivs(y[i - 1] + dt * dy3, t_i + dt, R0, t_inc, t_inf, t_ser, t_hos, t_mor, t_imm)
        y[i] = y[i - 1] + dt / 6 * (dy1 + 2 * dy2 + 2 * dy3 + dy4)

    # Plotting the results
    plt.plot(t, y[:, 0], label='Susceptible')
    plt.plot(t, y[:, 1], label='Infected')
    plt.plot(t, y[:, 2], label='Deaths')
    plt.plot(t, y[:, 3], label='Asymptomatic')
    plt.plot(t, y[:, 4], label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIDARTHE Model')
    plt.legend()
    plt.show()


# Example usage
R0 = 2.5
t_inc = 5
# Time from infection to recovery
t_inf = 14
# Time from infection to severe symptoms
t_ser = 9
# Time from severe symptoms to hospitalization
t_hos = 2
# Time from hospitalization to death or recovery
t_mor = 3
# Time from recovery to immunity
t_imm = 30
N = 1000000
I0 = 100
S0 = N - I0
A0 = 0
D0 = 0
R0 = 0
T = 200

sidarthe_model(R0, t_inc, t_inf, t_ser, t_hos, t_mor, t_imm, N, I0, S0, A0, D0, R0, T)
