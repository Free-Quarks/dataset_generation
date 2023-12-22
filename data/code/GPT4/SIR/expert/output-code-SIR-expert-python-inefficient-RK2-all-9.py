import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model_rk2(y, t, beta, gamma):
  S, I, R = y
  dSdt = -beta * S * I
  dIdt = beta * S * I - gamma * I
  dRdt = gamma * I
  return [dSdt, dIdt, dRdt]

def run_sir_model_rk2(S0, I0, R0, beta, gamma, T, dt):
  N = int(T/dt) + 1
  t = np.linspace(0, T, N)
  y = np.empty((N, 3))
  y[0] = [S0, I0, R0]

  for i in range(N-1):
    k1 = np.asarray(sir_model_rk2(y[i], t[i], beta, gamma))
    k2 = np.asarray(sir_model_rk2(y[i] + dt*k1, t[i] + dt, beta, gamma))
    y[i+1] = y[i] + dt * 0.5 * (k1 + k2)

  plt.figure(figsize=(10,5))
  plt.plot(t, y[:,0], label='Susceptible')
  plt.plot(t, y[:,1], label='Infected')
  plt.plot(t, y[:,2], label='Recovered')
  plt.xlabel('Time')
  plt.ylabel('Population')
  plt.legend()
  plt.show()

run_sir_model_rk2(S0=0.9, I0=0.1, R0=0.0, beta=0.35, gamma=0.1, T=70, dt=0.1)
