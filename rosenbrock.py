import numpy as np
from smt.applications.ego import EGO 
from smt.sampling_methods import LHS

def rosenbrock(x):
    inputLength = len(x[:,0])
    outputArr = np.ones([inputLength,1])
    numInput = 0
    a = 1
    b = 100
    for input in x:
        print(input)
        outputArr[numInput,0] = pow(a-input[0],2) + b*pow(input[1]-pow(input[0],2),2)
        numInput+=1
    
    return outputArr


xlimits=np.array([[-2.0,2.0], [-1.0,3.0]])
criterion='SBO' #'EI' or 'SBO' or 'LCB'

#number of points in the initial DOE
ndoe = 100 #(at least number of design variables + 1)
#number of iterations with EGO 
n_iter = 100

#Build the initial DOE, add the random_state option to have the reproducibility of the LHS points
sampling = LHS(xlimits=xlimits, random_state=1)
xdoe = sampling(ndoe)

#EGO call
ego = EGO(n_iter=n_iter, criterion=criterion, xdoe=xdoe, xlimits=xlimits)
x_opt, y_opt, ind_best, x_data, y_data = ego.optimize(fun=rosenbrock)
print('Optimized design variables and the minimized objective: ', x_opt, y_opt, ' obtained using EGO criterion = ', criterion)