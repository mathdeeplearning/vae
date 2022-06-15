import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

class VaeGMM(object):
	def __init__(self,data,num_guassian,sigma=1):
		self.data = data
		self.N = data.shape[0]
		self.K = num_guassian
		self.sigma = sigma

		# 为每个样本初始化phi向量
		self.phi = np.random.random([self.N,self.K])

		# 初始化mu的均值
		self.m = np.random.random(self.K)

		# 初始化mu的方差
		self.s2 = np.random.random(self.K)

	'''
	计算当前数据集的ELBO
	'''
	def elbo(self):
		A = (-0.5 * np.add.outer(self.data**2, self.m**2 + self.s2) + np.outer(self.data, self.m)) * self.phi

		B = -np.sum(self.m**2 + self.s2)/(2 * self.sigma**2)

		D = -np.sum(np.log(self.phi))

		E = np.sum(self.phi * np.log(self.phi + 1e-5))

		return np.sum(A) + B + D + E

	'''
	坐标上升迭代变分推断
	'''
	def cavi(self):

		e = np.outer(self.data, self.m) + (-0.5 * (self.m**2 + self.s2))[np.newaxis, :] 
		self.phi = np.exp(e) / np.sum(np.exp(e), axis=1)[:, np.newaxis] 

		self.m = np.sum(self.data[:, np.newaxis] * self.phi, axis=0)/(1.0 / self.sigma**2 + np.sum(self.phi, axis=0))

		self.s2 = 1.0 / (1.0 / self.sigma**2 + np.sum(self.phi, axis = 0))

	def train(self, epsilon, iters):
		elbo = []
		elbo.append(self.elbo())

		# use cavi to update elbo until epsilon-convergence
		for i in range(iters):

			self.cavi()

			elbo.append(self.elbo())

			print("Iter[{}] elbo is: {}".format(i,elbo[i]))

			if np.abs(elbo[-1] - elbo[-2]) <= epsilon:
				print("Iter[{}] of convergence:".format(i))
				break

		return elbo

	def plot(self,size):
		sns.set_style("whitegrid")

		for i in range(self.N//size):

			sns.distplot(data[size*i : (i+1)*size], rug=True)

			x = np.linspace(self.m[i] - 3*self.sigma, self.m[i] + 3*self.sigma, 100)

			plt.plot(x,norm.pdf(x, self.m[i], self.sigma),color='black')

		plt.show()

if __name__ == '__main__':
   
    number = 5000
    clusters = 5
    mu = np.array([1, 5, 7, 9, 15])
    data = []

    for i in range(clusters):
        data.append(np.random.normal(mu[i], 1, number))

    # concatenate data
    data = np.concatenate(np.array(data))

    model = VaeGMM(data, clusters)
    
    model.train(1e-3, 1000)

    print("converged_means:", sorted(model.m))

    model.plot(number)
