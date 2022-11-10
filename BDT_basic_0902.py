import numpy as np
import pandas as pd
import os    # import and read files
import os.path
from scipy.optimize import minimize

def temp_forward(sp, N):
    """return a tree of forward rates with the given spot rates"""
    temp_f = np.zeros( (N+1, N+1) )
    temp_f[0,0] = sp[0]
    for i in range(1, N+1 ):
        M = i + 1
        temp_f[0, i] = u * temp_f[0, i-1]
        for j in range(1, M ):
            temp_f[j, i] = d * temp_f[j - 1, i - 1]
    
    return temp_f

def discount_zero(sp):
    """return the prices of bonds with the given spot rates"""
    dis_zero = []
    for idx, i in enumerate(sp):
        temp = prin*(1/(1+i)**(idx+1))
        dis_zero.append(temp)
    
    return dis_zero

def forward_discount(fd_rates, N):
    """calculating present value of bonds with the given forward rates and N"""
    p_forward = np.zeros( (N+1, N+1) )
    for i in range(0, N+1):
        for j in range(0, N+1):
            if fd_rates[j, i] == 0:
                continue
            else:
                p_forward[j, i] = prin/(1+fd_rates[j, i])
    return p_forward

def rate_gap(adj, fd_rates, N):
    ab_forward = np.zeros( (N+1, N+1) )
    adj_rates = np.zeros( (N+1, N+1) )
    # only last comluns should be adjusted
    for i in range(0, N+1):
        for j in range(0, N+1):
            if i == N:
                adj_rates[j, i] = fd_rates[j, i] + adj
            else:
                adj_rates[j, i] = fd_rates[j, i]

    # calculate a new prices of bonds based on adjusted forward rates
    p_forward = forward_discount(adj_rates, N)
    for i in range(0, N+1):
        ab_forward[i, N] = p_forward[i, N]

    # average each nods' value and discount
    for i in range(N-1, -1, -1):
        for j in range(N-1, -1, -1):
            if j > i:
                continue
            else:
                ab_forward[j, i] = (ab_forward[j, i+1] + ab_forward[j+1, i+1])*0.5/(1+adj_rates[j, i])
                
    target = ab_forward[0,0]
    base = discount_zero(spot)[N]
    # print("price: \n", ab_forward)
    # print("adj_rates: \n", adj_rates)
    # print("gap: ", abs(base - target))
    return abs(base - target)


if __name__ ==  '__main__':
    # Initialise parameters
    spot = [0.05, 0.0551, 0.0604] # given spot rates, as this input has been changed, the output also be changed
    N = len(spot)-1 # the number of steps
    prin = 10000 # principle payment at the mature date
    sigma = 0.1
    u = np.exp(sigma)
    d = 1/u
    adj = 0.001
    temp_f = temp_forward(spot, N)

    final_rates = np.zeros( (N+1, N+1) ) # finally adjusted forward rates
    # before adjusted, final rates should be matched with temp_f
    for i in range(0, N+1):
        for j in range(0, N+1):
            final_rates[j, i] = temp_f[j, i]

    # Except for 0 column, every column returns adjust rate and then add to previous forward rate
    for i in range(0, N+1):
        if i == 0:
            continue
        else:
            temp_res = minimize(rate_gap, adj, args=(final_rates, i), method="L-BFGS-B")
            print(temp_res.x[0])
            for j in range(0, i+1):
                final_rates[j, i] = temp_f[j, i] + temp_res.x[0]

    print("*"*100,"\nfinal adjusted rates\n",final_rates, "\n","*"*100)