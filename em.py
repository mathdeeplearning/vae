import numpy as np
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn import mixture

class EMModel(object):
	def __init__(self,data,num_guassian):
		self.data = data
		self.N = data.shape[0]
		self.K = num_guassian

		# 为每个样本初始化phi向量
		self.phi = np.random.random([self.N,self.K])

		# 初始化高斯分量的权重
		self.alpha = np.random.random(self.K)/self.K

		# 初始化mu的均值
		self.m = np.random.random(self.K)

		# 初始化mu的方差
		self.s2 = np.random.random(self.K)

	def elbo(self):
		
		A = -0.5 * (np.add.outer(self.data, -self.m) **2)/(self.s2 + 1e-5)

		B = np.log(self.alpha + 1e-5) - 0.5 * np.log(self.s2 + 1e-5)

		return np.sum(self.phi * (A + B))	


	def e_step(self):

		# update phi
		self.phi = np.exp(-0.5*(np.add.outer(self.data, -self.m) **2)/(self.s2 + 1e-5))

		self.phi = self.phi/(np.sqrt(self.s2 + 1e-5))

		self.phi = self.phi *( self.alpha + 1e-5)

		# normalize
		self.phi = self.phi/(np.sum(self.phi,axis=1)[:,np.newaxis])

	def m_step(self):
		
		# update alpha
		self.alpha = (np.sum(self.phi,axis=0))/ self.N

		# update m
		self.m = np.sum(self.phi * data[:,np.newaxis],axis=0)/(np.sum(self.phi,axis = 0))

		# update s2
		self.s2 = np.sum(self.phi * np.add.outer(self.data, -self.m) **2,axis=0)/ (np.sum(self.phi,axis = 0))


	def em_update(self):
		self.e_step()
		self.m_step()

	def train(self, epsilon, iters):

		last = self.elbo()

		# use em to update elbo until epsilon-convergence
		for i in range(iters):

			self.em_update()

			curr = self.elbo()

			# print("Iter[{}] elbo : {}".format(i,curr))

			if np.abs(curr - last) <= epsilon:
				print("Iter[{}] of convergence:".format(i))
				break

			last = curr

	def plot(self,size):
		sns.set_style("whitegrid")

		for i in range(self.N//size):

			sns.distplot(data[size*i : (i+1)*size], rug=True)

			x = np.linspace(self.m[i] - 3*math.sqrt(self.s2[i]), self.m[i] + 3*math.sqrt(self.s2[i]), 100)

			plt.plot(x,norm.pdf(x, self.m[i], math.sqrt(self.s2[i])),color='black')

		plt.show()

if __name__ == '__main__':
   
	number = 5000

	clusters = 5
	mu = [1, 2, 3, 4, 5]

	sigma = [2, 2, 2, 2, 2]

	data = []

	for i in range(clusters):
		data.append(np.random.normal(mu[i], sigma[i], number))

	# for i in range(clusters):
	# 	d = np.random.randn(number)
	# 	data.append(d * sigma[i] + mu[i])

	# concatenate data
	data = np.concatenate(np.array(data))

	new_data = data.reshape(-1,1)

	gmm = mixture.GaussianMixture(n_components=clusters, max_iter=5000, covariance_type='full').fit(new_data)

	print('sk-means\n',sorted(gmm.means_.reshape(-1)))
	print('sk-std\n',np.sqrt(gmm.covariances_).reshape(-1))

	model = EMModel(data, clusters)

	model.train(1e-5, number)

	print("em-converged_means:\n", sorted(model.m))
	print("em-converged_vars:\n",np.sqrt(model.s2))

	# model.plot(number)
