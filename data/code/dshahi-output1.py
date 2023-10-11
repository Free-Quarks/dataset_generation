import numpy as np
import matplotlib.pyplot as plt


def sidarthe_model(N, E0, I0, A0, S0, R0, T0, H0, E2_0, H2_0, days, beta, sigma, gamma, alpha, delta, theta):
    # Initial conditions
    E = [E0]
    I = [I0]
    A = [A0]
    S = [S0]
    R = [R0]
    T = [T0]
    H = [H0]
    E2 = [E2_0]
    H2 = [H2_0]
    
    # Parameters
    dt = 1
    t = np.arange(days)
    
    for day in range(days-1):
        # Compute new infections
        new_infections = beta * (I[day] + A[day] + T[day]) * S[day] / N
        
        # Compute new exposed
        new_exposed = sigma * new_infections
        
        # Compute new asymptomatic
        new_asymptomatic = gamma * A[day]
        
        # Compute new infected
        new_infected = alpha * I[day]
        
        # Compute new recovered
        new_recovered = (1 - delta) * R[day]
        
        # Compute new tested
        new_tested = theta * T[day]
        
        # Compute new hospitalized
        new_hospitalized = (1 - delta) * (1 - theta) * H[day]
        
        # Compute new severe
        new_severe = delta * (I[day] + A[day] + T[day])
        
        # Compute new critical
        new_critical = delta * H[day]
        
        # Compute new exposed 2
        new_exposed2 = sigma * new_asymptomatic
        
        # Compute new hospitalized 2
        new_hospitalized2 = (1 - delta) * new_severe
        
        # Update variables
        S.append(S[day] - new_infections)
        E.append(E[day] + new_exposed - new_asymptomatic)
        I.append(I[day] + new_infected - new_recovered - new_critical)
        A.append(A[day] + new_asymptomatic - new_exposed2)
        R.append(R[day] + new_recovered)
        T.append(T[day] + new_tested)
        H.append(H[day] + new_hospitalized - new_hospitalized2)
        E2.append(E2[day] + new_exposed2)
        H2.append(H2[day] + new_hospitalized2)
        
    return t, S, E, I, A, R, T, H, E2, H2


N = 1000000
E0 = 1
I0 = 1
A0 = 1
S0 = N - E0 - I0 - A0
R0 = 0
T0 = 0
H0 = 0
E2_0 = 0
H2_0 = 0

# Parameters
days = 100
beta = 0.2
sigma = 0.1
gamma = 0.1
alpha = 0.2
delta = 0.05
theta = 0.2

# Run model
t, S, E, I, A, R, T, H, E2, H2 = sidarthe_model(N, E0, I0, A0, S0, R0, T0, H0, E2_0, H2_0, days, beta, sigma, gamma, alpha, delta, theta)

# Plot results
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, A, label='Asymptomatic')
plt.plot(t, R, label='Recovered')
plt.plot(t, T, label='Tested')
plt.plot(t, H, label='Hospitalized')
plt.plot(t, E2, label='Exposed 2')
plt.plot(t, H2, label='Hospitalized 2')

plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SIDARTHE Model')
plt.legend()
plt.grid()
plt.show()
