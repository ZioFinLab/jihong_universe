import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.stats import bernoulli
from scipy.optimize import fsolve
from scipy.stats import norm
import os    # import and read files
import os.path

def stocktree(S0):
    stock_prices = np.zeros( (N+1, N+1) )
    stock_prices[0,0] = S0
    for i in range(1, N+1 ):
        M = i + 1
        stock_prices[0, i] = u * stock_prices[0, i-1]
        for j in range(1, M ):
            stock_prices[j, i] = d * stock_prices[j - 1, i - 1]
    
    return stock_prices

def strikeprice(K):
    K_prices = np.zeros( (N+1, N+1) )
    K_prices[0,0] = K
    for i in range(0, N+1 ):
        M = i + 1
        for j in range(0, M ):
            K_prices[j, i] = K
    
    return K_prices


def puttree():
    p_prices = np.zeros( (N+1, N+1) )
    p_gr=np.log(1+p_guaranteed)
    for i in range(1, N+1 ):
        M = i + 1
        if i >= 157 : # 상환권 행사 가능 이후에만 고려함
            if i == 157: # 상환권 행사할 수 있는 최초 시기
                p_prices[0, i] = I*np.exp(p_gr)**(i*node_term/365) 
                
            elif i == N:        # final node = red 
                p_prices[0, i] = red
                    
            else: #i % 4 == 0:  # 상환권 행사가 주기적으로 반영되는지 확인할 것
                p_prices[0, i] = I*np.exp(p_gr)**(i*node_term/365)
        for j in range(1, M ):
            p_prices[j, i] = p_prices[j-1 , i]

    return p_prices

def calltree():
    c_prices = np.zeros( (N+1, N+1) )
    c_gr=np.log(1+c_guaranteed)

    return c_prices

def ratetree():
    rd = rates.iloc[12,1:].to_numpy()     # count
    rf = rates.iloc[11,1:].to_numpy()     # count
    rd_matrix = np.zeros( (N+1, N+1) )
    rd_matrix[0,0] = rd[0]

    for i in range(1, N+1 ):
        M = i + 1
        rd_matrix[0, i] = rd[i]
        for j in range(1, M ):
            rd_matrix[j, i] = rd_matrix[j-1, i]

    rf_matrix = np.zeros( (N+1, N+1) )
    rf_matrix[0,0] = rf[0]

    for i in range(1, N+1 ):
        M = i + 1
        rf_matrix[0, i] = rf[i]
        for j in range(1, M ):
            rf_matrix[j, i] = rf_matrix[j - 1, i]

    return rd_matrix, rf_matrix


def qtree():
    q_value = np.zeros( (N+1, N+1) )
    rd_matrix, rf_matrix = ratetree()
    for i in range(1, N+1 ):
        M = i + 1
        q_value[0, i] = (np.exp(rf_matrix[0,i]*dt) - d)/(u-d)
        for j in range(1, M ):
            q_value[j, i] = (np.exp(rf_matrix[j,i]*dt) - d)/(u-d)
    
    return q_value


def final_value(stock_prices, p_prices, K_prices, c_prices, q_value, rd_matrix, rf_matrix):
    coupon = 0 # error 수정을 위해 임시로 집어 넣은 것
    eq_value = np.zeros( (N+1, N+1) )
    for i in range(0, N+1):
        if stock_prices[i, N] > p_prices[i, N]:
            eq_value[i, N] = (stock_prices[i, N]*K)/K_prices[i, N]
        else:
            eq_value[i, N] = 0

    db_value = np.zeros( (N+1, N+1) )
    for i in range(0, N+1):
        if stock_prices[i, N] > p_prices[i, N]:
            db_value[i, N] = 0
        else:
            db_value[i, N] = p_prices[i, N]

    hp_value = np.zeros( (N+1, N+1) )
    for i in range(0, N+1):
        hp_value[i, N] = eq_value[i, N] + db_value[i, N]


    cb_value = np.zeros( (N+1, N+1) )
    for i in range(0, N+1):
        if c_prices[i, N] != 0: cb_value[i,N] = max(min(hp_value[i,N], c_prices[i,N]), (stock_prices[i,N]*K)/K_prices[i,N], p_prices[i,N]) # Callable option이 있을 때만 활성화
        else: cb_value[i, N] = max(hp_value[i,N], (stock_prices[i,N]*K)/K_prices[i,N], p_prices[i,N]) # call 없을 때 적용 가격


    #  Fill out the remaining values
    for i in range( N-1, -1, -1 ):
        for j in range( 0, i+1 ):
            if i==0 and j==0: coupon = 0 # 기초에는 coupon을 받지 않음
            # Holding Value 먼저 계산해줌
            hp_value[j, i] = ( eq_value[j,i+1]*q_value[j,i+1] + eq_value[j+1,i+1]*(1-q_value[j+1,i+1]) )*np.exp(-rf_matrix[j+1,i+1]*dt)+ \
                            ( db_value[j,i+1]*q_value[j,i+1] + db_value[j+1,i+1]*(1-q_value[j+1,i+1]) )*np.exp(-rd_matrix[j+1,i+1]*dt)+  coupon
            # 계산된 Holding Value기준으로 최종가치산출
            if c_prices[j, i] != 0: cb_value[j, i] = max(min(hp_value[j,i], c_prices[j, i]), (stock_prices[j,i]*K)/K_prices[j,i], p_prices[j,i]) # Callable option이 있을 때만 활성화
            else: cb_value[j, i] = max(hp_value[j,i], (stock_prices[j,i]*K)/K_prices[j,i], p_prices[j,i]) # call 없을 때 적용 가격
            
            # Equity Value 산출, 앞서 계산된 최종가치에 따라 값이 변동 됨: 주식가치 - 주식가치/ 조기상환 등 - 0/ Holding Value - Equity Value의 이전 노드 할인금액
            if cb_value[j, i] == (stock_prices[j,i]*K)/K_prices[j,i]:
                eq_value[j, i] = (stock_prices[j,i]*K)/K_prices[j,i]
            elif cb_value[j, i] == hp_value[j,i]:
                eq_value[j, i] = ( eq_value[j,i+1]*q_value[j,i+1] + eq_value[j+1,i+1]*(1-q_value[j+1,i+1]) )*np.exp(-rf_matrix[j+1,i+1]*dt)
            else:
                eq_value[j, i] = 0
            
            # Debt Value 산출, Equity Value와 반대
            if cb_value[j, i] == (stock_prices[j,i]*K)/K_prices[j,i]:
                db_value[j, i] = 0
            elif cb_value[j, i] == hp_value[j, i]:
                db_value[j, i] = ( db_value[j, i+1]*q_value[j, i+1] + db_value[j+1, i+1]*(1-q_value[j+1, i+1]) )*np.exp(-rd_matrix[j+1, i+1]*dt) + coupon
            elif cb_value[j, i] == c_prices[j, i]:
                db_value[j, i] = c_prices[j, i]
            elif cb_value[j, i] == p_prices[j, i]:
                db_value[j, i] = p_prices[j, i]
    
    return hp_value, cb_value, eq_value, db_value


if __name__ ==  '__main__':
    # Initialise parameters
    S0 = 1000000      # initial stock price       => 추후 업데이트 要
    K = 1236500       # strike price
    T = 10            # time to maturity in years
    N = 522           # number of time steps
    I= 1236500       # Issue price
    sigma = 0.458894943397503  # volatility
    node_term=7       # 7일단위
    p_guaranteed=0.05         # Put_guaranteed rate of return_YTM                
    c_guaranteed=0.00         # Call_guaranteed rate of return_YTM => 해당사항 없음
    coupon = 0 # 액면이자
    div = 0

    dt = 1/(365/node_term)
    u = np.exp(sigma * np.sqrt(dt))       # up-factor in binomial models
    d = 1/u       # ensure recombining tree 
    red=I*(1+p_guaranteed)**T

    curDir = os.getcwd()  
    os.chdir('C:/Users/jihon/Desktop/Python Codes/Option') 
    rates = pd.read_excel('RCPS 평가_summary.xlsx', sheet_name='Dashboard_1차 발행일(17.05.16)') # => 일드커브 추후 직접 구현 要

    stock_prices = stocktree(S0)
    K_prices = strikeprice(K)
    p_prices = puttree()
    c_prices = calltree()
    rd_matrix, rf_matrix = ratetree()
    q_value = qtree()
    hp, cb, eq, db = final_value(stock_prices, p_prices, K_prices, c_prices, q_value, rd_matrix, rf_matrix)
    print(hp[0,0])
