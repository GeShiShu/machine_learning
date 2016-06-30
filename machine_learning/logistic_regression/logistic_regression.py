import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x));

#load data from a file
def loadData(datafile):
    dataMat=[]
    labelMat=[]
    with open(datafile,'r') as f:
        for line in f.readlines():
            lineArray=line.strip().split()
            for i in range(len(lineArray)):
                if i==0:
                    dataMat.append(1.0) #NOTE:there is a constant x0=1
                if i==len(lineArray)-1:
                    labelMat.append(int(lineArray[i]))
                else:
                    dataMat.append(float(lineArray[i]))
    return dataMat,labelMat 

#use gradient decent method to train data
def trainData(dataMat,labelMat):
    labelMatrix=np.mat(labelMat).transpose()
    p,q=np.shape(labelMatrix)
    dataMatrix=np.mat(dataMat).reshape(p,-1)
    m,n=np.shape(dataMatrix)
    alpha=0.005
    maxCycles=10000
    weights=np.matrix(np.random.random_sample((n,1)))

    print "dataMatrix:\n",dataMatrix 
    print "\nlabelMatrix:\n",labelMatrix
    print "\ninitial weights:\n",weights
    print "\n" 
    last_mean_err=0.0;
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        error=labelMatrix-h
        mean_error=np.mean(np.abs(error))
        if abs(mean_error-last_mean_err)<1.0*np.power(10,-4):
            break
        print "iterate",k+1,"times\t","mean error:",mean_error
        weights=weights+alpha*dataMatrix.transpose()*error
        print "weights:\n",weights
        last_mean_err=mean_error 
    print "\nDONE!\nultimate weights\n",weights 
    return dataMatrix,labelMatrix,weights 

#plot 2-D pictures if the data is 2 dimensions
def plotResult(weights,dataMatrix,labelMatrix):
    if np.shape(dataMatrix)[1] != 3:  #if it is not 2 dimensions data
        return np.False_
    x0=[];y0=[]
    x1=[];y1=[]
    n=np.shape(dataMatrix)[0]
    for i in range(n):
        if(labelMatrix[i,0]==0):
            x0.append(dataMatrix[i,1])
            y0.append(dataMatrix[i,2])
        else:
            x1.append(dataMatrix[i,1])
            y1.append(dataMatrix[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(x0,y0,s=25,c='red',marker='s')
    ax.scatter(x1,y1,s=25,c='blue',marker='o')
    x=np.arange(-4.0,4.0,0.1)
    y=(-weights[0,0]-weights[1,0]*x)/weights[2,0]
    plt.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title("Logistic Regression")
    plt.show()
    return np.True_ 

if __name__=='__main__':
    dataMat,labelMat=loadData("data.txt");
    dataMatrix,labelMatrix,weights=trainData(dataMat,labelMat)
    plotResult(weights,dataMatrix,labelMatrix)
        

