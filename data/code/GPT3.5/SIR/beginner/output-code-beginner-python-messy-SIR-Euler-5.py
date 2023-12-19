import matplotlib.pyplot as plt

def SIR_model(S, I, R, beta, gamma, num_days):
    N = S + I + R
    S_list = [S]
    I_list = [I]
    R_list = [R]
    for _ in range(num_days):
        dS = -beta * S * I / N
        dI = beta * S * I / N - gamma * I
        dR = gamma * I
        S += dS
        I += dI
        R += dR
        S_list.append(S)
        I_list.append(I)
        R_list.append(R)
    return S_list, I_list, R_list

# Example usage
S, I, R = 990, 10, 0
beta, gamma = 0.2, 0.1
num_days = 100
S_list, I_list, R_list = SIR_model(S, I, R, beta, gamma, num_days)

days = range(num_days + 1)

plt.plot(days, S_list, label='Susceptible')
plt.plot(days, I_list, label='Infected')
plt.plot(days, R_list, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Number of individuals')
plt.title('SIR Model')
plt.legend()
plt.show()
