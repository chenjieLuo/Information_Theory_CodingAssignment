#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 14:45:19 2019

@author: Chenjie
"""

import numpy as np
import matplotlib.pyplot as plt
import operator as op
from functools import reduce
import operator
import math

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

n = 1000
k = np.linspace(0,n, n+1, endpoint=True)
num_elements = np.zeros(n+1)
for i in range(0,n+1):
    num_elements[i] = ncr(n,i)
    
alpha = k/n
f1 = plt.figure()
plt.plot(alpha,num_elements)
plt.xlabel('alpha')
plt.ylabel('num of elements')


prob_element = np.zeros(n+1)
for i in range(0,n+1):
    prob_element[i] = np.power(0.1,i)*np.power(0.9,n-i)
f2 = plt.figure()
plt.plot(alpha[0:200],np.log(prob_element[0:200]))
plt.xlabel('alpha')
plt.ylabel('log(P(_each_element))')

prob_occurred = np.zeros(n+1)
for i in range(0,n+1):
    prob_occurred[i] = ncr(n,i)*np.power(0.9,n-i)*np.power(0.1,i)
    
f3 = plt.figure()
plt.plot(alpha, prob_occurred)
plt.xlabel('alpha')
plt.ylabel('Probability of type')

entropy = -0.1*np.log2(0.1) - 0.9*np.log2(0.9)
epsilon = 0.06
delta = 0.0506
A_set_prob = 0
tA= 0
[prob_lower, prob_higher] = [np.exp2(-n*(entropy+epsilon)),np.exp2(-n*(entropy-epsilon))]
for i in range(0,n+1):
    if prob_element[i] > prob_lower and prob_element[i] < prob_higher:
        A_set_prob += prob_element[i]*ncr(n,i)
        tA += ncr(n,i)

B_set_prob = 0
tB = 0
probability = prob_occurred
for i in range(0,n+1):
    if B_set_prob < 1-delta:    
        index, value = max(enumerate(probability), key=operator.itemgetter(1))
        B_set_prob += value
        tB += ncr(n,index)
        probability[index] = 0
        
num_elements_inA = 1/n*np.log(tA)
num_elements_inB = 1/n*np.log(tB)   

S = 0
for k in range(12,20):
    S= S + ncr(25,k)
