import numpy as np
import matplotlib.pyplot as plt

def SIR_RK4(beta, gamma, S0, I0, R0, t_end, dt):
    def SIR_model(SIR, beta, gamma):
        S, I, R = SIR
        dS_dt = -beta * S * I / (S + I + R)
        dI_dt = beta * S * I / (S + I + R) - gamma * I
        dR_dt = gamma * I
        return [dS_dt, dI_dt, dR_dt]

    population = S0 + I0 + R0
    initial_conditions = [S0/population, I0/population, R0/population]
    t = np.arange(0, t_end+dt, dt)
    SIR = np.zeros((len(t), 3))
    SIR[0] = initial_conditions

    for i in range(len(t)-1):
        k1 = SIR_model(SIR[i], beta, gamma)
        k2 = SIR_model(SIR[i] + 0.5*dt*k1, beta, gamma)
        k3 = SIR_model(SIR[i] + 0.5*dt*k2, beta, gamma)
        k4 = SIR_model(SIR[i] + dt*k3, beta, gamma)
        SIR[i+1] = SIR[i] + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    return t, SIR[:,0]*population, SIR[:,1]*population, SIR[:,2]*population

beta = 0.3
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
t_end = 100
dt = 0.1

plt.figure(figsize=(12,6))

# Run simulation
t, S, I, R = SIR_RK4(beta, gamma, S0, I0, R0, t_end, dt)

# Plot results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK4')
plt.legend()
plt.grid(True)
plt.show()
