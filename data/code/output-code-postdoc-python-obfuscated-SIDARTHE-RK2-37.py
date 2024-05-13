import numpy as np
import matplotlib.pyplot as plt

def sidarthe_model(beta, alpha, gamma, delta, epsilon, theta, N, I0, R0, D0, T): 
    def F(t, Y):
        S, I, D, A, R, T = Y
        N = S + I + D + A + R
        return [-beta*S*I/N, (beta*S*I)/N - (alpha + gamma)*I, delta*I - epsilon*A - theta*I, epsilon*A, gamma*I, delta*I - theta*I]

    Y0 = [N-I0-R0-D0, I0, D0, 0, R0, T]
    t = np.linspace(0, T, T+1)
    sol = odeint(F, Y0, t)
    return sol


beta = 0.8
alpha = 0.07
gamma = 0.07
epsilon = 0.001
theta = 0.07
delta = 0.14
N = 1000000
I0 = 10
R0 = 0
D0 = 0
T = 100

solution = sidarthe_model(beta, alpha, gamma, delta, epsilon, theta, N, I0, R0, D0, T)

plt.plot(solution[:, 0], label='S')
plt.plot(solution[:, 1], label='I')
plt.plot(solution[:, 2], label='D')
plt.plot(solution[:, 3], label='A')
plt.plot(solution[:, 4], label='R')
plt.plot(solution[:, 5], label='T')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIDARTHE Model')
plt.legend()
plt.show()
