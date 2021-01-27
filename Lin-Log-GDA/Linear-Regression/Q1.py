# Importing all the required libraries
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from matplotlib import cm

# Function to calculate the error:
def J(y,data_x,thetha,m):
	sumVal = 0.
	for i in range(m):
		sumVal = sumVal + (y[i] - thetha[0] - thetha[1] * data_x[i])**2
	return (sumVal/m)		

## Reading the csv values into arrays:
data_x = []
y = []

# Using CSV functions to read the values row-wise
with open("/home/vedant/Downloads/COL774/Ass1/data/q1/linearX.csv") as f:
	reader = csv.reader(f)
	for row in reader:
		data_x.append(row[0])

with open("/home/vedant/Downloads/COL774/Ass1/data/q1/linearY.csv") as f:
	reader = csv.reader(f)
	for row in reader:
		y.append(row[0])

# Converting the values in float
data_x = np.asarray(data_x).astype(np.float64)
y = np.asarray(y).astype(np.float64)

## Normalization step:

# Obtaining the mean and variance of the data_x
mean = np.mean(data_x)
var = np.var(data_x)

# Obtaining the normalization co-efficients
a = math.sqrt(1/var)
b = -a*mean

# Doing the normalization
data_x = a*data_x + b

### Part (A) : Implementing the batch gradient descent

# Initializing the thetha vector
thetha = np.array([0.,0.])
m = len(y)
alpha = 0.025

## Applying the gradient descend:

# The parameters of the previous iteration
thetha_old = np.array([0.,0.])

# The data points to be plotted
fig_x = []
fig_y = []
fig_z = []

# Running the algorithm for at max 1000 steps and breaking when the consequtive thethas are very close to each other, which will be a stopping condition
for p in range(10000):
	
	fig_x.append(thetha[0])
	fig_y.append(thetha[1])
	fig_z.append(J(y,data_x,thetha,m))

	for j in range(2):
		sumVal = 0.
		for i in range(m):
			temp = (y[i] - thetha[0] - thetha[1]*data_x[i])
			if j==1:
				temp = temp * (data_x[i])
			sumVal = sumVal + temp
			
		thetha[j] = thetha[j] + alpha*sumVal/m	

	diff = np.linalg.norm(thetha - thetha_old)
	
	# The comparision between the consequtive values of the parameters
	if (diff < 10**-10):
		break
	
	# Saving the older value in thetha_old
	a = thetha[0]
	b = thetha[1]
	thetha_old = np.array([a,b])

# Number of frames required
num = len(fig_x)

'''
# Variables for the plots:
fig = plt.figure()
ax = p3.Axes3D(fig)


# Setting the plot
ax.set_xlim3d([0.0, 1.0])
ax.set_xlabel('Thetha[0]')

ax.set_ylim3d([0.0, 0.002])
ax.set_ylabel('Thetha[1]')

ax.set_zlim3d([0.0, 1.1])
ax.set_zlabel('Error Value')

ax.set_title('3D Test')
ax.set_yticklabels([])


# The update function to make a animation as per the different frames
def update_lines(frames, fig_x,fig_y,fig_z):
	frames = int(frames)
	print(frames)
	ax.scatter(fig_x[:frames],fig_y[:frames],fig_z[:frames], c ='red')
	return 
	
# The Animate function
line_ani = FuncAnimation(fig, update_lines, frames = list(range(0,num)), fargs=(fig_x, fig_y, fig_z), interval = 200)

# Printing the final Value	
print(thetha)	
plt.show()	
'''
		

### Part (B): Getting the 2D plot
'''
# Plotting the scatters
for i in range(m):
	plt.scatter(data_x[i],y[i],color = 'blue')

# Getting the labels for the individual points by taking known points from the CSV files
plt.scatter(data_x[0],y[0],color = 'blue', label = 'Data points')

# Plotting the line: Y = (thetha)' * X 
x = np.linspace(-4,5,1000)
plt.plot(x,thetha[0] + thetha[1]*x, color='black', label = 'Hypothesis function')

# Setting the Graph
plt.legend()
plt.xlabel('x axis')
plt.ylabel('y axis')	
plt.savefig('Q1_plot.png', dpi=1000, bbox_inches='tight')	
'''


### Part (C) and ahead: Included in the above code

## To make the mesh 
ax = plt.axes(projection='3d')
X1 = np.arange(-5, 5, 0.25)
Y1 = np.arange(-5, 5, 0.25)
X1, Y1 = np.meshgrid(X1, Y1)
plt.xlabel('Thetha[0]')
plt.ylabel('Thetha[1]')
ax.set_zlabel('Error Value')
surf = ax.plot_surface(X1, Y1, J(y,data_x,[X1,Y1],m), cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.savefig('Q1_mesh.png', dpi=1000, bbox_inches='tight')	

		
		
		
		
		
		
