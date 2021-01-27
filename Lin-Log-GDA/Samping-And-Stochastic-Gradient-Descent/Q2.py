import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from matplotlib import cm


### Part (A) : Obtaining the random samples
 
## Initializing the Variables
thetha = [3,1,2]
x1 = np.random.normal(3,2,10**6)
x2 = np.random.normal(-1,2,10**6)
epsilon = np.random.normal(0,math.sqrt(2),10**6)

## Obtaining the random distribution
y_random = thetha[0] + thetha[1]*x1 + thetha[2]*x2 + epsilon

## Writing it to a file

# Converting to a numpy array
q2_random = np.asarray(y_random)
q2_random = q2_random.T

# Using CSV functions to write the values row-wise
#with open("q2_random.csv", "w", newline = "") as f:
#	writer = csv.writer(f)
#	writer.writerows(map(lambda x: [x], q2_random))


### Part (B) : Impementing the SGD

# Function to calculate the error:
def J(y,x1,x2,thetha,m):
	sumVal = 0.
	for i in range(m):
		sumVal = sumVal + (y[i] - thetha[0] - thetha[1] * x1[i] - thetha[2] * x2[i])**2
	return (sumVal/m)		

# Initializing the parameters
thetha = [0,0,0]

# Total Size
m = len(x1)

# Batch size 
r = 1

# Learning rate
alpha = 0.001

# The parameters of the previous iteration
thetha_old = np.array([0.,0., 0.])
flag = 0

# Size to get the Error result
if r==100:
	size_err = 5000
elif r==1:
	size_err = 3000
else:
	size_err = 2000
	
# Setting the Arrays for the Mesh plot
fig_x = []
fig_y = []
fig_z = []


# Running the algorithm for at max 1000 steps and breaking when the consequtive thethas are very close to each other, which will be a stopping condition
for p in range(10000):
	
	if flag==1:
		break
	
	for i in range(int(m/r)):
	
		if flag==1:
			break
	
		for j in range(3):
			sumVal = 0.
			for k in range(r):
				temp = 	(y_random[i*r+k] - thetha[0] - thetha[1]*x1[i*r+k] - thetha[2]*x2[i*r+k])
				if j==1:
					temp = temp * x1[i*r+k]
				elif j==2:
					temp = temp * x2[i*r+k]
				sumVal = sumVal + temp
			thetha[j] = thetha[j] + alpha * sumVal/r
		
		# Appending the values of the parameters	
		fig_x.append(thetha[0])
		fig_y.append(thetha[1])
		fig_z.append(thetha[2])
	
		if r==10000:
			if p<180:
				continue
		elif r==1000000:
			if p<5000:
				continue		
		else:
			if (p+1)*i<(m):
				continue
		
		
		J1 = J(y_random,x1,x2,thetha_old,size_err)
		J2 = J(y_random,x1,x2,thetha,size_err)		
		diff = abs(J1-J2)
		
		# The comparision between the consequtive values of the parameters
		if (r==100):		
			if (diff < 10**-3):
				flag=1
				break
		elif(r==1):
			if (diff < 5 * 10**-5):
				flag=1
				break
		else:
			if (diff < 10**-3):
				flag=1
				break
		
		# Saving the older value in thetha_old
		a = thetha[0]
		b = thetha[1]
		c = thetha[2]
		thetha_old = np.array([a,b,c])
		


print("done")

'''
# Variables for the plots:
fig = plt.figure()
ax = p3.Axes3D(fig)

# Setting the plot
ax.set_xlim3d([0.0, 3.2])
ax.set_xlabel('Thetha[0]')

ax.set_ylim3d([0.0, 1.2])
ax.set_ylabel('Thetha[1]')

ax.set_zlim3d([0.0, 2.2])
ax.set_zlabel('Error Value')

ax.set_title('3D Test')

num = len(fig_x)

# The update function to make a animation as per the different frames
def update_lines(frames, fig_x,fig_y,fig_z):
	frames = int(frames)
	print(frames)
	ax.scatter(fig_x[:frames],fig_y[:frames],fig_z[:frames], c ='red')
	return 
	
# The Animate function

line_ani = FuncAnimation(fig, update_lines, frames = np.linspace(0,num,1000), fargs=(fig_x, fig_y, fig_z), interval = 200)

#frames = num
#ax.scatter(fig_x[:frames],fig_y[:frames],fig_z[:frames], c ='red')

# Printing the final Value	
print(thetha)	
plt.show()	
'''	


### Part (C) : Obtaining the error from the given CSV file

data_x0 = []
data_x1 = []
y = []

# Using the CSV functions to read the given file
with open("/home/vedant/Downloads/COL774/Ass1/data/q2/q2test.csv") as f:
	# Assumed that the first line of "X1,X2,Y" is removed
	for line in f:
		data = line.split(",")
		data_x0.append(data[0])
		data_x1.append(data[1])
		y.append(data[2])

# Converting the values in float
data_x0 = np.asarray(data_x0).astype(np.float64)
data_x1 = np.asarray(data_x1).astype(np.float64)
y = np.asarray(y).astype(np.float64)

# Total number of values
m = len(y)

# The different values of thetha
thetha_guess = [3,1,2]
thetha_1 = [3.038301642588017, 1.00928428823021, 1.9811896842795802]
thetha_2 = [2.9920771531749293, 1.0036560602622213, 2.001015275358533]
thetha_3 = [2.9768598287664783, 1.0047669699104917, 1.9979146661810465]
thetha_4 = [3.0019348435234035, 0.9993534593453501, 2.0002348235234019]
    

# Function to calculate the error:
def J(y,x1,x2,thetha,m):
	sumVal = 0.
	for i in range(m):
		sumVal = sumVal + (y[i] - thetha[0] - thetha[1] * x1[i] - thetha[2] * x2[i])**2
	return (sumVal/(2*m))	
	
print("The error for the Assumed hypothesis is: %f" %(J(y,data_x0,data_x1,thetha_guess,m)))
print("The error for the First hypothesis is: %f" %(J(y,data_x0,data_x1,thetha_1,m)))
print("The error for the Second hypothesis is: %f" %(J(y,data_x0,data_x1,thetha_2,m)))
print("The error for the Third hypothesis is: %f" %(J(y,data_x0,data_x1,thetha_3,m)))
print("The error for the Fourth hypothesis is: %f" %(J(y,data_x0,data_x1,thetha_4,m)))

			

### Part (D) : Obtaining the mesh plots

