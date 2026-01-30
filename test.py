prices = [4,2,8]
strategy = [-1,0,1]
k = 2
l = []
def maxProfit(prices,strategy, k):
    n = len(prices)   
    len = n - k + 1
    P = prices
    S = strategy
    for i in range((len + 1)):
        k = 0
        for i in range(n):
            l = P[i]
            j = S[i]
            add = l * j 
            k += add
        l.append(k)
        S =

        
           



