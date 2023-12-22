import numpy as np
import matplotlib.pyplot as plt
import json

def SIR_model(S, I, R, beta, gamma):
  """
  SIR model differential equations.

  Args:
    S: number of susceptible
    I: number of infected
    R: number of recovered
    beta: contact rate
    gamma: mean recovery rate

  Returns:
    dSdt : differential equation of S
    dIdt : differential equation of I
    dRdt : differential equation of R
  """
  dSdt = -beta * S * I
  dIdt = beta * S * I - gamma * I
  dRdt = gamma * I
  return [dSdt, dIdt, dRdt]

def RK4(S, I, R, beta, gamma, dt):
  """
  Runge-Kutta 4th order solution for SIR model.

  Args:
    S: number of susceptible
    I: number of infected
    R: number of recovered
    beta: contact rate
    gamma: mean recovery rate
    dt: time step

  Returns:
    S, I, R: new susceptibles, infected and recovered
  """
  for i in range(1, len(S)):
    k1 = dt * np.array(SIR_model(S[i-1], I[i-1], R[i-1], beta, gamma))
    k2 = dt * np.array(SIR_model(S[i-1] + 0.5*k1[0], I[i-1] + 0.5*k1[1], R[i-1] + 0.5*k1[2], beta, gamma))
    k3 = dt * np.array(SIR_model(S[i-1] + 0.5*k2[0], I[i-1] + 0.5*k2[1], R[i-1] + 0.5*k2[2], beta, gamma))
    k4 = dt * np.array(SIR_model(S[i-1] + k3[0], I[i-1] + k3[1], R[i-1] + k3[2], beta, gamma))
    S[i] = S[i-1] + (1/6)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    I[i] = I[i-1] + (1/6)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    R[i] = R[i-1] + (1/6)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
  return S, I, R

# Test the function
S = np.zeros(1000)
I = np.zeros(1000)
R = np.zeros(1000)
I[0] = 1
beta = 0.2
gamma = 0.1
dt = 0.1
S, I, R = RK4(S, I, R, beta, gamma, dt)

plt.figure(figsize=(8, 6))
plt.plot(S, 'b', label='Susceptibles')
plt.plot(I, 'r', label='Infected')
plt.plot(R, 'g', label='Recovered')
plt.legend(loc='best')
plt.xlabel('Time')
plt.ylabel('Number')
plt.grid()
plt.show()
