from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import matrix_power as matPow
import pandas as pd

A = [[0.928, 0.125], [-0.217, 1.270]]
eigenVal1 = 1.145
eigenVal2 = 1.053
eigenVect1 = np.array([1, 0.576])
eigenVect2 = np.array([1, 1])

# print("A = ", A)
# print("A[0][0] = ", A[0][0])
# print("A[0][0] = ", A[0][1])

def getVector(constant1, constant2):
	c1 = constant1
	c2 = constant2
	v1 = eigenVect1
	v2 = eigenVect2
	x0 = np.add(np.multiply(c1,v1), np.multiply(c2,v2)) #The vector x0
	return (x0)




def xk(k, vector): #Returns the Xk vector
	newA = matPow(A,k)
	val1 = np.add(np.multiply(newA[0][0],vector[0]), np.multiply(newA[0][1],vector[1]))
	val2 = np.add(np.multiply(newA[1][0],vector[0]), np.multiply(newA[1][1],vector[1]))
	vectToRetrun = np.array([val1,val2])
	return(vectToRetrun)

def plotVector(vectorList, color):

	origin = [0], [0] # origin point
	for x in vectorList:
		plt.plot(x[0],x[1], color)	#plt.quiver(*origin, vector[0],vector[1])

	plt.axhline()
	plt.axvline()
	# origin = [0], [0]
	# pyplot.quiver(*origin, vector, color=['r'], scale=21)
	# pyplot.show()



def main():
	x0 = getVector(50,100)
	x1 = getVector(150,200)
	x2 = getVector(40,55)
	x3 = getVector(100,340)
	x4 = getVector(80,60)
	x5 = getVector(175,325)

	#print(pd.DataFrame(x0,columns=list('0'),index=list('01')))
	#plotVector(x0)
	#print('Now printing the other vectors:')
	vectList = []
	for x in range (1,20):
		vector = xk(x,x0)
		if(vector[1] >= 0.576*vector[0]):
			vectList.append(vector)
	plotVector(vectList, 'bo')
	vectList.clear()
	for x in range (1,20):
		vector1 = xk(x,x1)
		if(vector1[1] >= 0.576*vector1[0]):
			vectList.append(vector1)
	plotVector(vectList, 'ro')
	vectList.clear()
	for x in range (1,20):
		vector2 = xk(x,x2)
		if(vector2[1] >= 0.576*vector2[0]):
			vectList.append(vector2)
	plotVector(vectList, 'go')
	vectList.clear()
	for x in range (1,20):
		vector3 = xk(x,x3)
		if(vector3[1] >= 0.576*vector3[0]):
			vectList.append(vector3)
	plotVector(vectList, 'mo')
	vectList.clear()
	for x in range (1,20):
		vector4 = xk(x,x4)
		if(vector4[1] >= 0.576*vector4[0]):
			vectList.append(vector4)
	plotVector(vectList, 'co')
	vectList.clear()
	for x in range (1,20):
		vector5 = xk(x,x5)
		if(vector5[1] >= 0.576*vector5[0]):
			vectList.append(vector5)
	plotVector(vectList, 'ko')
	vectList.clear()

	x = np.linspace(0,800,100)
	y = x
	y2 = 0.576*x
	plt.plot(x, y2, '-b', label='y=0.576x | (Line containing the Dominant Eigenvector)')
	plt.title('Leopard and Mongoose Population Dynamic System')
	plt.xlabel('Number of Leopards', color='#1C2833')
	plt.ylabel('Number of Mongooses', color='#1C2833')
	plt.legend(loc='upper left')
	# for y in range (1,3):
	# 	vectList.append(xk(y*5,x0))
	# 	vectList.append(xk(y*5,x1))
	# 	vectList.append(xk(y*5,x2))
	plt.show()

	#xk(2)


if __name__ == '__main__':
	main()
