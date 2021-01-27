import numpy as np
import math
import csv
import matplotlib.pyplot as plt

## Reading the csv values into arrays:

data_x0 = []
data_x1 = []
y = []

# Using CSV functions to read the values row-wise
with open("/home/vedant/Downloads/COL774/Ass1/data/q4/q4x.dat") as f:
	for line in f:
		data = line.split()
		data_x0.append(data[0])
		data_x1.append(data[1])			
		
with open("/home/vedant/Downloads/COL774/Ass1/data/q4/q4y.dat") as f:
	reader = csv.reader(f)
	for row in reader:
		y.append(row[0])

# Converting the values in float
data_x0 = np.asarray(data_x0).astype(np.float64)
data_x1 = np.asarray(data_x1).astype(np.float64)

## Normalization step:
data_x0 = (data_x0-(np.mean(data_x0)))/(math.sqrt(np.var(data_x0)))
data_x1 = (data_x1-(np.mean(data_x1)))/(math.sqrt(np.var(data_x1)))

# Total number of tests
m = len(y)


### Part (A): Applying GDA:

## here: y=0 corresponds to "Alaska" and y=1 corresponds to "Canada"
phi_sum = 0.
u0_num = np.matrix([0.,0.])
u0_denom = 0.
u1_num = np.matrix([0.,0.])
u1_denom = 0.

# Doing the Calculations as per the ones derived in class (Also described in Andrew NG's notes)
for i in range(m):
	if(y[i]=="Alaska"):
		u0_num = u0_num + np.array([data_x0[i],data_x1[i]])
		u0_denom = u0_denom + 1
	else:
		phi_sum = phi_sum + 1
		u1_num = u1_num + np.array([data_x0[i],data_x1[i]])
		u1_denom = u1_denom + 1

phi = phi_sum/m
u0 = u0_num/u0_denom
u1 = u1_num/u1_denom

# Obtaining the Sigma matrix
X_sig = np.zeros((m,2))
for i in range(m):
	X_sig[i,0] = data_x0[i]
	X_sig[i,1] = data_x1[i]

sigma_num = 0
for i in range(m):
	if(y[i] == "Alaska"):
		sigma_num += np.dot((X_sig[i] - u0).reshape((2,1)),(X_sig[i]-u0))
	else:
		sigma_num += np.dot((X_sig[i] - u1).reshape((2,1)),(X_sig[i]-u1))
	
sigma = sigma_num/m

# Printing the values
print("Phi, u0, u1, Sigma are ")
print(phi)
print(u0)
print(u1)
print(sigma)


### Part (B): Scatter Plots

# Plotting the scatters
for i in range(m):
	if (y[i]=="Alaska"):
		plt.scatter(data_x0[i],data_x1[i],color = 'red')
	else: 
		plt.scatter(data_x0[i],data_x1[i],color = 'blue')
		
# Getting the labels for the individual points by taking known points from the CSV files
plt.scatter(data_x0[0],data_x1[0],color = 'red', label = 'Alaska')
plt.scatter(data_x0[51],data_x1[51],color = 'blue', label = 'Canada')

#Setting the graph
#plt.legend()
#plt.xlabel('x1 axis')
#plt.ylabel('x2 axis')	
#plt.savefig('Q4_plot_scatter.png', dpi=1000, bbox_inches='tight')	


### Part (C): Plotting the Boundary line

## The equation of the line obtained
x = np.linspace(-2.5,2.5,1000)
plt.plot(x, (-(sigma[0,0]*u0[0,0] + sigma[0,1]*u0[0,1])/(sigma[1,1]*u0[0,1] + sigma[1,0]*u0[0,0]))*x, color='black', label = 'Boundary line')


# Setting the Graph
#plt.legend()
#plt.xlabel('x1 axis')
#plt.ylabel('x2 axis')	
#plt.savefig('Q4_plot_boundary.png', dpi=1000, bbox_inches='tight')	


### Part (D): With Different values of Sigma

# Keeping the means same as before
phi_new = phi
u0_new = u0
u1_new = u1

# Calculations for the Sigma matrix
sigma0_num = 0
sigma1_num = 0

for i in range(m):
	if(y[i] == "Alaska"):
		sigma0_num += np.dot((X_sig[i] - u0).reshape((2,1)),(X_sig[i]-u0))
	else:
		sigma1_num += np.dot((X_sig[i] - u1).reshape((2,1)),(X_sig[i]-u1))
		
sigma0 = sigma0_num/u0_denom
sigma1 = sigma1_num/u1_denom

# Printing the new values
print("New Phi, u0, u1, Sigma0, Sigma1 are ")
print(phi_new)
print(u0_new)
print(u1_new)
print(sigma0)
print(sigma1)


### Part (E): Obtaining the Quadratic Boundary

## The Boundary obtained now is: Ax2 + Bxy + Cy2 + Dx + Ey + F = 0 and B^2 - 4AC = 0
# Defining the temporary variables

isigma0 = np.linalg.inv(sigma0)
isigma1 = np.linalg.inv(sigma1)

# Corresponding to the: X' * (Sigma)-1 * X terms 
a = isigma1[0,0] - isigma0[0,0]
b = 2 * (isigma1[1,0] - isigma0[1,0])
c = isigma1[1,1] - isigma0[1,1]

# Corresponding to the constant terms
d = np.log(np.linalg.det(sigma0)) - np.log(np.linalg.det(sigma1))

# The Linear constants
temp1 = np.dot(u0.reshape((1,2)),isigma0) - np.dot(u1.reshape((1,2)),isigma1)
temp2 = np.dot(isigma0,u0.reshape((2,1))) - np.dot(isigma1,u1.reshape((2,1)))

# Bug fix
temp1 = temp1.flatten()
temp2 = temp2.flatten()

A = a
B = b
C = c
D = temp1[0,0] + temp2[0,0]
E = temp1[0,1] + temp2[0,1]
F = d + np.dot(u1.reshape((1,2)),np.dot((isigma1),(u1.reshape((2,1))))) - np.dot(u0.reshape((1,2)),np.dot((isigma0),(u0.reshape((2,1)))))

# Plotting the curve
x = np.linspace(-4,4,1000)
y = np.linspace(-4,4,1000)
x,y = np.meshgrid(x,y)

plt.contour(x,y,(A*x**2 + B*x*y + C*y**2 + D*x + E*y + F),[0], colors = 'Green')


# Setting the Graph
plt.legend()
plt.xlabel('x1 axis')
plt.ylabel('x2 axis')	
plt.savefig('Q4_plot_combined.png', dpi=1000, bbox_inches='tight')	






