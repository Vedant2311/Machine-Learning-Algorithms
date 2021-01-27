import numpy as np
import math
import csv
import matplotlib.pyplot as plt

## Reading the csv values into arrays:

data_x0 = []
data_x1 = []
y = []

# Using CSV functions to read the values row-wise
with open("/home/vedant/Downloads/COL774/Ass1/data/q3/logisticX.csv") as f:
	reader = csv.reader(f)
	for row in reader:
		data_x0.append(row[0])
		data_x1.append(row[1])

with open("/home/vedant/Downloads/COL774/Ass1/data/q3/logisticY.csv") as f:
	reader = csv.reader(f)
	for row in reader:
		y.append(row[0])

# Converting the values in float
data_x0 = np.asarray(data_x0).astype(np.float64)
data_x1 = np.asarray(data_x1).astype(np.float64)
y = np.asarray(y).astype(np.float64)

## Normalization step:
data_x0 = (data_x0-(np.mean(data_x0)))/(math.sqrt(np.var(data_x0)))
data_x1 = (data_x1-(np.mean(data_x1)))/(math.sqrt(np.var(data_x1)))

### Part (A) : Implementing Newton's method 

# Initializing the thetha vector
thetha = np.array([0,0,0])
m = len(y)

## Defining the functions to calculate h(x), gradient, and Hessian

# Getting the h(x)
def h(data_x0,data_x1,thetha):
	z = thetha[0] + thetha[1]*data_x0 + thetha[2]*data_x1
	return (1/(1 + math.exp(-z)))
	
# Getting the gradient
def grad(data_x0,data_x1,y,thetha,m):
	output = np.array([0,0,0])
	for j in range(3):
		sumVal = 0
		for i in range(m):
			if (j==0):
				xj=1
			elif (j==1):
				xj = data_x0[i]
			else:
				xj = data_x1[i]
		
			partial_grad = (y[i] - h(data_x0[i],data_x1[i],thetha))*xj
			sumVal = sumVal + partial_grad
			
		output[j] = sumVal	
				
	return output

# Getting the Hessian
def hessian(data_x0,data_x1,y,thetha,m):
	a = [[0,0,0],[0,0,0],[0,0,0]]
	output = np.array(a)
	
	# Obtaining H(j,k)
	for j in range(3):
		for k in range(3):
			sumVal=0
			for i in range(m):
				
				# xj value
				if (j==0):
					xj=1
				elif (j==1):
					xj = data_x0[i]
				else:
					xj = data_x1[i]
				
				# xk value			
				if (k==0):
					xk=1
				elif (k==1):
					xk = data_x0[i]
				else:
					xk = data_x1[i]
					
				sumVal = sumVal - (h(data_x0[i],data_x1[i],thetha)) * (1-h(data_x0[i],data_x1[i],thetha))*xj*xk
			
			output[j][k] = sumVal
	
	return output

## The newton's method

thetha_old = thetha

# Running the algorithm for at max 1000 steps and breaking when the consequtive thethas are very close to each other, which will be a stopping condition. 
for i in range(8):
	thetha = thetha - np.dot((np.linalg.inv(hessian(data_x0,data_x1,y,thetha,m))),(grad(data_x0,data_x1,y,thetha,m)))
	
	diff = np.linalg.norm(thetha - thetha_old)
	# The comparision between the consequtive values of the parameters
	if (diff < 10**-10):
		break
	thetha_old = thetha

print(thetha)

### Part (B): Plots

# Plotting the scatters
for i in range(m):
	if (y[i]==0):
		plt.scatter(data_x0[i],data_x1[i],color = 'red')
	else: 
		plt.scatter(data_x0[i],data_x1[i],color = 'blue')
		
# Getting the labels for the individual points by taking known points from the CSV files
plt.scatter(data_x0[0],data_x1[0],color = 'red', label = 'Points with y=0')
plt.scatter(data_x0[51],data_x1[51],color = 'blue', label = 'Points with y=1')


# Plotting the line: (thetha)' * X = 0
x = np.linspace(-3,3,1000)
plt.plot(x,-(thetha[0]/thetha[2]) - (thetha[1]/thetha[2])*x, color='black', label = 'Boundary line')

# Setting the Graph
plt.legend()
plt.xlabel('x1 axis')
plt.ylabel('x2 axis')	
plt.savefig('Q3_plot.png', dpi=1000, bbox_inches='tight')	



