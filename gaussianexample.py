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
plt.show() #save and close display first :)

# a "hidden" true function we're pretending we don't know
def true_function(x):
    return np.sin(x) + 0.5 * np.cos(2 * x)

x_obs = np.array([1.0, 3.0, 5.0, 7.0, 9.0]) #fake observations
noise_std = 0.2 #noise deviation
y_obs = true_function(x_obs) + noise_std * np.random.randn(len(x_obs)) #add realistic noise to observations

print(y_obs) #print observations


K_xx = rbf_kernel(x_obs, x_obs, lengthscale=1.5) + noise_std**2 * np.eye(len(x_obs)) #build the matrix for the observations 
K_xs = rbf_kernel(x_obs, x_grid, lengthscale=1.5) #

# solve for the GP's best guess (posterior mean) at every grid point
alpha = np.linalg.solve(K_xx, y_obs)
posterior_mean = K_xs.T @ alpha

# how related is the grid to itself (needed for uncertainty)
K_ss = rbf_kernel(x_grid, x_grid, lengthscale=1.5)


v = np.linalg.solve(K_xx, K_xs) #the uncertainty and trust of GP's guess
posterior_cov = K_ss - K_xs.T @ v #familliarity with noise measuring reduction of uncertainty

posterior_var = np.diag(posterior_cov) #extract variance at each point
posterior_std = np.sqrt(posterior_var) 

print(posterior_std)

plt.plot(x_grid, posterior_mean, label="GP's guess") #for the plot
plt.fill_between(x_grid, #mean of graph
                  posterior_mean - 2 * posterior_std,
                  posterior_mean + 2 * posterior_std,
                  alpha=0.2, label="uncertainty (95%)")
plt.plot(x_obs, y_obs, 'ko', label="noisy data")
plt.legend()
plt.title("GP posterior mean and uncertainty")
plt.show() #show (dont forget to close and save) :)
