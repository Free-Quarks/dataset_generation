import numpy as np
import matplotlib.pyplot as plt


# Function to implement the SIDARTHE model using RK2
# Parameters:
#   beta: average number of contacts per person per day
#   gamma: inverse of the average duration of infection
#   sigma: inverse of the average duration of the latency period
#   mu: case fatality rate
#   N: total population size
#   T: number of time steps
#   initial_conditions: dictionary with initial values for each compartment
# Returns:
#   arrays containing the values of each compartment at each time step

def simulate_sidarthe_rk2(beta, gamma, sigma, mu, N, T, initial_conditions):
    # Initialize arrays
    S = np.zeros(T)
    I = np.zeros(T)
    D = np.zeros(T)
    A = np.zeros(T)
    R = np.zeros(T)
    T = np.zeros(T)
    H = np.zeros(T)
    E = np.zeros(T)
    
    # Set initial conditions
    S[0] = initial_conditions['S']
    I[0] = initial_conditions['I']
    D[0] = initial_conditions['D']
    A[0] = initial_conditions['A']
    R[0] = initial_conditions['R']
    T[0] = initial_conditions['T']
    H[0] = initial_conditions['H']
    E[0] = initial_conditions['E']
    
    # Iterate over time steps
    for t in range(1, T):
        dt = 1
        S[t] = S[t-1] - beta*S[t-1]*I[t-1]*dt/N
        E[t] = E[t-1] + beta*S[t-1]*I[t-1]*dt/N - sigma*E[t-1]*dt
        I[t] = I[t-1] + sigma*E[t-1]*dt - (gamma+mu)*I[t-1]*dt
        D[t] = D[t-1] + mu*I[t-1]*dt
        A[t] = A[t-1] + gamma*I[t-1]*dt
        R[t] = R[t-1] + (1-mu)*I[t-1]*dt
        T[t] = T[t-1] + sigma*E[t-1]*dt
        H[t] = H[t-1] + mu*I[t-1]*dt
    
    return S, I, D, A, R, T, H, E


# Example usage
N = 1000000
T = 100
beta = 0.2
gamma = 0.1
sigma = 0.05
mu = 0.03
initial_conditions = {'S': N-1, 'I': 1, 'D': 0, 'A': 0, 'R': 0, 'T': 0, 'H': 0, 'E': 0}

S, I, D, A, R, T, H, E = simulate_sidarthe_rk2(beta, gamma, sigma, mu, N, T, initial_conditions)

time = np.arange(T)

plt.plot(time, S, label='Susceptible')
plt.plot(time, I, label='Infected')
plt.plot(time, D, label='Deceased')
plt.plot(time, A, label='Asymptomatic')
plt.plot(time, R, label='Recovered')
plt.plot(time, T, label='Tested')
plt.plot(time, H, label='Hospitalized')
plt.plot(time, E, label='Exposed')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Population')
plt.show()
