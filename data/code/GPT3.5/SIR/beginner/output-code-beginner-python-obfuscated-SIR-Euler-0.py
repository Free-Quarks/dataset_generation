import matplotlib.pyplot as plt

# Function to implement the SIR model
def sir_model(beta, gamma, s0, i0, r0, N, t):
    # Define arrays to store values
    S = [s0]
    I = [i0]
    R = [r0]

    # Define step size
    dt = t[1] - t[0]

    # Iterate over time steps
    for idx in range(1, len(t)):
        # Calculate new values
        dsdt = -beta * S[idx-1] * I[idx-1] / N
        didt = beta * S[idx-1] * I[idx-1] / N - gamma * I[idx-1]
        drdt = gamma * I[idx-1]

        # Update values
        S.append(S[idx-1] + dsdt * dt)
        I.append(I[idx-1] + didt * dt)
        R.append(R[idx-1] + drdt * dt)

    return S, I, R

# Example usage
time = range(0, 101)

# Define parameters
beta = 0.2
gamma = 0.1
s0 = 0.99
i0 = 0.01
r0 = 0
N = s0 + i0 + r0

# Call function
S, I, R = sir_model(beta, gamma, s0, i0, r0, N, time)

# Plot results
plt.plot(time, S, label='Susceptible')
plt.plot(time, I, label='Infected')
plt.plot(time, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
