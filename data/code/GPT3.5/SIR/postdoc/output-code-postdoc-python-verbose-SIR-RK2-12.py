import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(N, I0, R0, beta, gamma, tmax, dt):
    S0 = N - I0 - R0
    S = [S0]
    I = [I0]
    R = [R0]
    t = [0]
    
    while t[-1] < tmax:
        S_star = S[-1] - beta*S[-1]*I[-1]*dt
        I_star = I[-1] + beta*S[-1]*I[-1]*dt - gamma*I[-1]*dt
        R_star = R[-1] + gamma*I[-1]*dt
        
        S.append(S[-1] - beta*S_star*I_star*dt)
        I.append(I[-1] + beta*S_star*I_star*dt - gamma*I_star*dt)
        R.append(R[-1] + gamma*I_star*dt)
        
        t.append(t[-1] + dt)
    
    return np.array(t), np.array(S), np.array(I), np.array(R)

N = 1000
I0 = 1
R0 = 0
beta = 0.3
gamma = 0.1
tmax = 100
dt = 0.1

# Run simulation
t, S, I, R = SIR_RK2(N, I0, R0, beta, gamma, tmax, dt)

# Plot results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK2')
plt.legend()
plt.show()

