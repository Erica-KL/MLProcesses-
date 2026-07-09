#the infinite bag of curves example
#essentially creating curves from a gaussian distributions and randomly selecting one to plot
import numpy as np
import matplotlib.pyplot as plt
#data seed 
np.random.seed(42)  
#Kernel 
def rbf_kernel(x1, x2, lengthscale=1.0, variance=1.0): #establishes data and knobs
    sqdist = (x1[:, None] - x2[None, :]) ** 2 #sorts data into a square matrix
    return variance * np.exp(-0.5 * sqdist / lengthscale**2) #returns the distance lengthscale turns the lengthscale knob. creates a bell curve

test_points = np.array([0.0, 1.0, 2.0]) #test points for rbf_kernel
K = rbf_kernel(test_points, test_points) #create the layout for the kernal matrix
print(K) #print the matrix in the terminal


x_grid = np.linspace(0, 10, 100) #create a grid with 100 points betweem 0 and 10 


K = rbf_kernel(x_grid, x_grid, lengthscale=1.5) #build the matrix


L = np.linalg.cholesky(K + 1e-8 * np.eye(len(x_grid))) #use the parameters established to create a cholesky decompisition 


sample = L @ np.random.randn(len(x_grid)) #pick a random sample curve

plt.plot(x_grid, sample) #plot it print it
plt.title("One random curve pulled")
plt.show()