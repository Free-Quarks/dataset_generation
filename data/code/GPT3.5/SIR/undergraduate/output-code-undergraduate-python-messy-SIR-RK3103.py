import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(beta, gamma, N, I0, T):
    def f(t, y):
        S, I, R = y
        dS_dt = -beta*S*I/N
        dI_dt = beta*S*I/N - gamma*I
        dR_dt = gamma*I
        return [dS_dt, dI_dt, dR_dt]
    
    def RK3_step(f, t, y, h):
        k1 = f(t, y)
        k2 = f(t + h/2, [y_i + h/2*k_i for y_i, k_i in zip(y, k1)])
        k3 = f(t + h, [y_i - h*k1_i + 2*h*k2_i for y_i, k1_i, k2_i in zip(y, k1, k2)])
        y_new = [y_i + h/6*(k1_i + 4*k2_i + k3_i) for y_i, k1_i, k2_i, k3_i in zip(y, k1, k2, k3)]
        return y_new
    
    t = np.linspace(0, T, int(T)+1)
    h = t[1] - t[0]
    
    S = N - I0
    I = I0
    R = 0
    
    y = [S, I, R]
    
    sol = [y]
    for i in range(len(t)-1):
        y = RK3_step(f, t[i], y, h)
        sol.append(y)
    
    sol = np.array(sol)
    
    return t, sol[:,0], sol[:,1], sol[:,2]


beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
T = 100


fig, ax = plt.subplots()

# Run the model
t, S, I, R = SIR_RK3(beta, gamma, N, I0, T)

# Plot the results
ax.plot(t, S, label='Susceptible')
ax.plot(t, I, label='Infected')
ax.plot(t, R, label='Recovered')
ax.set_xlabel('Time')
ax.set_ylabel('Population')
ax.set_title('SIR Model using RK3')
ax.legend()
plt.show()
