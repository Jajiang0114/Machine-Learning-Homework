#James Jiang 11/7/2018
import numpy as np

xarray = []
yarray = []

xarray2 = []
yarray2 = []

#Read in the Training file for use
with open("a2-train-data.txt", "r") as infile:
	data = infile.readlines()
infile.close()

#Format the Data from the training file into X array to calculate
for line in data:
	temp = []
	points = line.split()
	#print(len(points))
	for i in range(0,len(points)):
		temp.append(float(points[i]))
	xarray.append(temp)
#Convert x and y arrays into numpy arrays for calculation
x_train = np.array(xarray)

#Open and read in the test file for use
with open("a2-train-label.txt", "r") as tstfile:
	data2 = tstfile.readlines()
tstfile.close()

#Format the test file into a Y array for calculation
for line in data2:
	points2 = line.split()
	for i in range(0,len(points2)):
			yarray.append(float(points2[i]))

#Convert the test array into a numpy array
y_train = np.array(yarray)

#print(X)
#--------------------------------------------------------------------------
with open("a2-test-data.txt", "r") as infile:
	data3 = infile.readlines()
infile.close()
#Format the Data from the test file into X arrays to calculate
for line in data3:
	temp = []
	points3 = line.split()
	#print(len(points))
	for i in range(0,len(points3)):
		temp.append(float(points3[i]))
	xarray2.append(temp)

#Convert x and y arrays into numpy arrays for calculation
x_test = np.array(xarray2)

#Open and read in the test file for use
with open("a2-test-label.txt", "r") as tstfile:
	data4 = tstfile.readlines()
tstfile.close()

#Format the test file into a x array for calculation
for line in data4:
	line = line.strip('[')
	line = line.strip(']')
	points4 = line.split(',')
	for i in range(0,len(points4)):
			yarray2.append(float(points4[i]))

#Convert the test array into a numpy array
y_test = np.array(yarray2)
#-----------------------------


#tanh Function 
def tanh (x): 
	return np.tanh(x)

#Derivative of tanh Function 
def tanh_derivative(x): 
	return 1 - (tanh(x)*tanh(x))

def accuracy(x, y):
	errors = 0
	#print(x)
	#xI = list(map(float, x))
	#yI = list(map(float, y))
	for i in range(0,len(x)):
		#print("X: ", x[i])
		#print("Y: ", y[i])
		if abs(x[i] - y[i]) < .01:
			errors += 0
		else:
			#print("Difference: ", x[i] - y[i])
			errors +=1
	return errors

def NN_calculations(X, y, XX, yy):
	learning_rate = .01
	HIDDEN_LAYER_SIZE = 10
	#print(XX.shape[1])
	INPUT_LAYER_SIZE = X.shape[1]
	#print(INPUT_LAYER_SIZE)
	OUTPUT_LAYER_SIZE = y.shape[0]

	#weights:
	W1 = np.random.randn(HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE) 
	W2 = np.random.randn(HIDDEN_LAYER_SIZE)

	#bias:
	Bh = np.full((HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE), 0.1)
	Bo = np.full(HIDDEN_LAYER_SIZE, 0.1)

	for k in range(100):
		for w in range(X.shape[0]):
			#feed_forward:
			y_h = []
			y_out = 0
			sum_h = 0
			sum_y = 0
			#Hidden Layer
			#print("Feed Foward: ")
			for i in range(HIDDEN_LAYER_SIZE):
				sum_h = 0
				for j in range(X.shape[1]):
					sum_h += ((X[w][j]*W1[i][j]) + Bh[i][j])
				y_h.append(tanh(sum_h))
			#Output Layer

			for j in range(HIDDEN_LAYER_SIZE):
				sum_y += ((y_h[j]*W2[j]) + Bo[j])
			y_out = (tanh(sum_y))
			#print(y_h)
			#print(len(y_out))
			#backprop:
			#print("Back Propagation: ")
			Z0 = 0
			Zi = []
		#	for i in range(y.shape[0]):
			#Delta 0 Error
			Z0 = (2 * (y_out - y[w]) * tanh_derivative(y_out))
			#print(Z0)
			for i in range(HIDDEN_LAYER_SIZE):

				Zi.append(Z0 * W2[i] * tanh_derivative(y_h[i]))
			# Cost derivative for weights
			#print(len(Z0))
			#print(len(Zi))
			#Gradient
			dW1 = []
			dW2 = []
			#print("Gradient: ")
			for i in range(HIDDEN_LAYER_SIZE):
				dW2.append(Z0 * y_h[i])
			#print(len(Zi))
			for i in range(HIDDEN_LAYER_SIZE):
				temp2 = []
				for j in range(X.shape[1]):
					temp2.append(np.multiply(Zi[i],X[w][j]))
				dW1.append(temp2)
			#print(len(dW1))
			#print(len(dW2))
			# Update weights
			#print("Updating Weights: ")
			for i in range(HIDDEN_LAYER_SIZE):	
				W2[i] = W2[i] - (dW2[i] * learning_rate)
				Bo[i] = Bo[i] - (dW2[i] * learning_rate)
			for i in range(HIDDEN_LAYER_SIZE):
				for j in range(INPUT_LAYER_SIZE):	
					W1[i][j] = W1[i][j] - (dW1[i][j] * learning_rate)
					Bh[i][j] = Bh[i][j] - (dW1[i][j] * learning_rate)
	#"""
	#Calculating Output
	output = []
	for w in range(X.shape[0]):
		y_temp = []
		for i in range(HIDDEN_LAYER_SIZE):
			sum_h = 0
			for j in range(X.shape[1]):
				sum_h += ((X[w][j]*W1[i][j]) + Bh[i][j])
			y_temp.append(tanh(sum_h))
		#Output Layer
		#print(len(y_h))
		sum_y = 0
		for j in range(HIDDEN_LAYER_SIZE):
			sum_y += ((y_temp[j]*W2[j]) + Bo[j])
		output.append(tanh(sum_y))
	accurate = accuracy(y.tolist(), output)
	print("Errors Train: ", accurate)
	#-----------------------------------------------------
	#Writing Weightss to file
	NNfile = open("NN_classifier.txt", "w")
	NNfile.write("Hidden Units: ")
	NNfile.write(str(HIDDEN_LAYER_SIZE))
	NNfile.write("\n")
	#Output Weights W2
	NNfile.write("Output Weights: ")
	NNfile.write("\n")

	for i in range(HIDDEN_LAYER_SIZE):	
		NNfile.write(str(W2[i]))
		NNfile.write(" ")
	NNfile.write("\n")
	#Input Weihts W1
	NNfile.write("Input Weights: ")
	NNfile.write("\n")
	for i in range(HIDDEN_LAYER_SIZE):
		for j in range(INPUT_LAYER_SIZE):	
			NNfile.write(str(W1[i][j]))
			NNfile.write(" ")
		NNfile.write("\n")
	NNfile.close()
	#--------------------------------------------
	#Testing set
	output_test = []
	for w in range(XX.shape[0]):
		y_temp = []
		for i in range(HIDDEN_LAYER_SIZE):
			sum_h = 0
			for j in range(XX.shape[1]):
				sum_h += ((X[w][j]*W1[i][j]) + Bh[i][j])
			y_temp.append(tanh(sum_h))
		#Output Layer
		#print(len(y_h))
		sum_y = 0
		for j in range(HIDDEN_LAYER_SIZE):
			sum_y += ((y_temp[j]*W2[j]) + Bo[j])
		output_test.append(tanh(sum_y))
	accurate2 = accuracy(yy.tolist(), output_test)
	print("Errors Test: ", accurate2)


NN_calculations(x_train, y_train, x_test, y_test)
#print(out_f)


#test_output = NN_calculations(x_test, y_test, "Test")
#print(out_f)
#accurate2 = accuracy(y_test.tolist(), test_output)
#print("Errors Test: ", accurate2)