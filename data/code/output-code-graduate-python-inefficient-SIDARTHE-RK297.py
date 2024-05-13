import numpy as np
import matplotlib.pyplot as plt

def sidarthe_model(params, y0, t):
    alpha, beta, rho, sigma, gamma, delta = params
    S_0, I_0, D_0, A_0, R_0, T_0, H_0, E_0 = y0
    N = sum(y0)
    
    S = S_0
    I = I_0
    D = D_0
    A = A_0
    R = R_0
    T = T_0
    H = H_0
    E = E_0
    
    S_arr = [S_0]
    I_arr = [I_0]
    D_arr = [D_0]
    A_arr = [A_0]
    R_arr = [R_0]
    T_arr = [T_0]
    H_arr = [H_0]
    E_arr = [E_0]
    
    dt = t[1] - t[0]
    
    for i in range(1, len(t)):
        S_next = S - alpha*S*I/N - rho*S*A/N
        I_next = I + alpha*S*I/N - sigma*I - beta*I
        D_next = D + sigma*I - gamma*D
        A_next = A + rho*S*A/N - delta*A
        R_next = R + gamma*D
        T_next = T + beta*I
        H_next = H + delta*A
        E_next = N - S_next - I_next - D_next - A_next - R_next - T_next - H_next
        
        S = S_next
        I = I_next
        D = D_next
        A = A_next
        R = R_next
        T = T_next
        H = H_next
        E = E_next
        
        S_arr.append(S_next)
        I_arr.append(I_next)
        D_arr.append(D_next)
        A_arr.append(A_next)
        R_arr.append(R_next)
        T_arr.append(T_next)
        H_arr.append(H_next)
        E_arr.append(E_next)
        
    return S_arr, I_arr, D_arr, A_arr, R_arr, T_arr, H_arr, E_arr


params = (0.2, 0.1, 0.03, 0.05, 0.1, 0.15)
y0 = (1000, 1, 0, 0, 0, 0, 0, 0)
t = np.linspace(0, 100, 1000)

S, I, D, A, R, T, H, E = sidarthe_model(params, y0, t)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, D, label='Deceased')
plt.plot(t, A, label='Asymptomatic')
plt.plot(t, R, label='Recovered')
plt.plot(t, T, label='Total Cases')
plt.plot(t, H, label='Hospitalized')
plt.plot(t, E, label='Exposed')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SIDARTHE Model')
plt.show()
