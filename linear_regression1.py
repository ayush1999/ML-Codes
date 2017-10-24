# Implementing linear regression WITHOUT using scikit-learn.

import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt as gft

#estimate coeffcients

def estimate_coefficients(x, y):
    n= np.size(x)
    m_x,m_y= np.mean(x),np.mean(y)
    SS_xy = np.sum(x*y - n*m_x*m_y)
    SS_xx = np.sum(x*x - n*m_x*m_x)
    b_1 = SS_xy/SS_xx
    b_0 = m_y- b_1*m_x

    return (b_0,b_1)
#plotting the graph of regression

def graph_of_regression(x, y, b):
    plt.scatter(x,y,color="m", marker="o",s= 30)
    y_pred= b[0]+ b[1]*x
    plt.plot(x, y_pred, color="g")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()
#the main block
def main():
    c=input('Enter manually or read from a csv file?')
    if(c=="Y"):
        size= int(input("Enter the size of X attribute:"))
        x=[]
        y=[]
        print("Enter the X Elements:")
        for i in range(size):
            element= int(input())
            x.append(element)
        print("Enter the Y Elements:")
        for i in range(size):
            element= int(input())
            y.append(element)

        x= np.array(x)
        y= np.array(y)

    else:
        data= gft('test_file.csv',delimiter=',')
        X=[]
        Y=[]
        for major in data:
            X.append(major[0])
            Y.append(major[1])
        x= np.array(X)
        y= np.array(Y)

    print("Estimated coefficients of linear regression are: ")
    print(estimate_coefficients(x, y))
    print("Graph:")
    graph_of_regression(x, y, estimate_coefficients(x, y))
    

if __name__=="__main__":
    main()

