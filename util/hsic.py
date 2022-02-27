from __future__ import division

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import gamma

def to_numpy(x):
    """convert Pytorch tensor to numpy array
    """
    return x.clone().detach().cpu().numpy()
"""
python implementation of Hilbert Schmidt Independence Criterion
hsic_gam implements the HSIC test using a Gamma approximation
Python 2.7.12
Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Scholkopf, B., 
& Smola, A. J. (2007). A kernel statistical test of independence. 
In Advances in neural information processing systems (pp. 585-592).
Shoubo (shoubo.sub AT gmail.com)
09/11/2016
Inputs:
X 		n by dim_x matrix
Y 		n by dim_y matrix
alph 		level of test
Outputs:
testStat	test statistics
thresh		test threshold for level alpha test
"""


def rbf_dot(pattern1, pattern2, deg):
	size1 = pattern1.shape
	size2 = pattern2.shape

	G = torch.sum(pattern1*pattern1, 1).reshape(size1[0],1)
	H = torch.sum(pattern2*pattern2, 1).reshape(size2[0],1)

	Q = torch.tile(G, (1, size2[0]))
	R = torch.tile(H.T, (size1[0], 1))

	H = Q + R - 2* torch.matmul(pattern1, pattern2.T)

	H = torch.exp(-H/(deg**2))

	return H


def hsic_gam(X, Y, alph = 0.5, kernel_param_average_method=None):
	"""
	X, Y are numpy vectors with row - sample, col - dim
	alph is the significance level
	auto choose median to be the kernel width
	"""
	if kernel_param_average_method == 'median':
		average_func = torch.median
	elif kernel_param_average_method == 'mean':
		average_func = torch.mean
	else :
		print("kernel average function should be specified, 'median' or 'mean'")
	#print(X.shape, "X.shape")
	# #print(Y.shape, "Y.shape")
	if type(X) == list :
		X = torch.unsqueeze(X, 1)
	if type(Y) == list :
		Y = torch.unsqueeze(Y, 1)
	X = torch.reshape(X, (X.shape[0], -1))
	Y = torch.reshape(Y, (Y.shape[0], -1))
	n = X.shape[0]

	# ----- width of X -----
	Xmed = X
	#print(Xmed.shape, "X shape")

	# print(Xmed*Xmed.shape, "Xmed*Xmed.shape")
	# print(torch.matmul(Xmed.T,Xmed).shape, "Xmed.T @ Xmed. shape")
	# print(torch.sum(Xmed*Xmed, 1).shape, "np.sum(Xmed*Xmed,1) shape")


	G = torch.sum(Xmed*Xmed,1).reshape(n,1)  #
	#print(G.shape)
	Q = torch.tile(G, (1, n) )
	#print(Q.shape)
	R = torch.tile(G.T, (n, 1) )
	#print(R.shape)

	dists = Q + R - 2* torch.matmul(Xmed, Xmed.T)
	#print(dists.shape)
	dists = dists - torch.tril(dists)
	dists = dists.reshape(n**2, 1)
	dists = torch.sqrt(dists)
	#print(dists.shape)

	print(dists[dists>0])

	print(average_func(dists[dists>0]), "average value")

	width_x = average_func(dists[dists>0]) # delta 값이 median으로 # kernel matrix 값을 살펴봐야 할 것 같음 > 직접 값을 보여드리는 게 좋을 것 같습니다.
	# ----- -----

	# ----- width of Y -----
	Ymed = Y

	G = torch.sum(Ymed*Ymed, 1).reshape(n,1)
	Q = torch.tile(G, (1, n) )
	R = torch.tile(G.T, (n, 1) )

	dists = Q + R - 2* torch.matmul(Ymed, Ymed.T)
	dists = dists - torch.tril(dists)
	dists = dists.reshape(n**2, 1)
	dists = torch.sqrt(dists)


	width_y = average_func(dists[dists>0])
	# ----- -----
	print('width_x : ', width_x)
	print('width_y : ', width_y)



	bone = torch.ones((n, 1)).to(torch.float)
	H = torch.eye(n) - (torch.ones((n,n)) / n).float()

	K = rbf_dot(X, X, width_x)
	bone = bone.to(K.device)

	L = rbf_dot(Y, Y, width_y)

	H = H.to(K.device)
	Kc = torch.matmul(torch.matmul(H, K), H)
	Lc = torch.matmul(torch.matmul(H, L), H)

	testStat = torch.sum(Kc.T * Lc) / n

	varHSIC = (Kc * Lc / 6)**2

	varHSIC = ( torch.sum(varHSIC) - torch.trace(varHSIC) ) / n / (n-1)

	varHSIC = varHSIC * 72 * (n-4) * (n-5) / n / (n-1) / (n-2) / (n-3)

	K = K - torch.diag(torch.diag(K))
	L = L - torch.diag(torch.diag(L))

	muX = torch.matmul(torch.matmul(bone.T, K), bone) / n / (n-1)
	muY = torch.matmul(torch.matmul(bone.T, L), bone) / n / (n-1)

	mHSIC = (1 + muX * muY - muX - muY) / n

	#al = (mHSIC**2 / varHSIC)
	#bet = (varHSIC*n / mHSIC)

	#thresh = gamma.ppf(1-alph, al, scale=bet)[0][0]


	return (testStat, (width_x, width_y))

def normalized_HSIC(X, Y, alph = 0.5, return_width=False, kernel_param_average_method='median'):
	HSIC_xx,_ = hsic_gam(X, X, kernel_param_average_method=kernel_param_average_method)
	HSIC_yy,_ = hsic_gam(Y, Y, kernel_param_average_method=kernel_param_average_method)
	if return_width == False:
		return hsic_gam(X, Y, alph=alph, kernel_param_average_method=kernel_param_average_method)[0] / (torch.sqrt(HSIC_xx) * torch.sqrt(HSIC_yy))
	else :
		HSIC, (width_x, width_y) = hsic_gam(X, Y, alph=alph, kernel_param_average_method=kernel_param_average_method)
		normalized_HSIC = HSIC / (torch.sqrt(HSIC_xx) * torch.sqrt(HSIC_yy))
		return normalized_HSIC, (width_x, width_y)



'''
from PIL import Image
import time
# img_cifar10_a = Image.open("./dataset/cifar10_224x224/train/airplane/0001.png")
# img_cifar10_b = Image.open("./dataset/cifar10_224x224/train/airplane/0002.png")
#
# img_cifar10_a = np.reshape(img_cifar10_a, (-1,3))[0]
# img_cifar10_b = np.reshape(img_cifar10_b, (-1,3))[0]

test_data_a = torch.normal(3,2.5, size = (20, 10))
test_data_b = torch.normal(-3,1, size = (20, 10))
print(test_data_a.shape, test_data_b.shape)


start = time.time()
hsic_stat_1, thresh = hsic_gam(test_data_a, test_data_a)
hsic_stat_1_normalized = normalized_HSIC(test_data_a, test_data_a)

end = time.time()
print("time:", end-start)
print(hsic_stat_1, hsic_stat_1_normalized)

start = time.time()
hsic_stat_2, thresh = hsic_gam(test_data_a, test_data_b)
hsic_stat_2_normalized = normalized_HSIC(test_data_a, test_data_b)

end = time.time()
print("time:", end-start)
print(hsic_stat_2, hsic_stat_2_normalized)


#print(hsic_stat_1, hsic_stat_2)
'''

if __name__ == "__main__" :

	import numpy as np
	import matplotlib.pyplot as plt

	# helper func
	def gaussian_kernel(X, sigma):
		if np.ndim(X[0])==0:
			Y = np.exp(-(X**2)/(sigma**2))
		elif np.ndim(X[0])>0:
			print(X.shape, "X shape should be N * 2")
			size=X.shape
			H = np.sum(X * X, 1).reshape(size[0], 1)
			Y = np.exp(- H /(sigma**2) ) # in case sigma is scalar.
			print("Y shape should be N", Y.shape )
		return X, Y


	# two 1d cases
	X = [100, 101, 102, 103, 104, 1000]
	zero = [0, 0, 0, 0, 0, 0]
	Y = [-400, -404, -408, -412, -416, -4000]


	torch_X = torch.tensor(X, dtype =torch.float)
	torch_Y = torch.tensor(Y, dtype =torch.float)
	HSIC, (width_x, width_y) = normalized_HSIC(torch_X, torch_Y, return_width=True, kernel_param_average_method='median')

	# print(width_x, width_y, "widths")
	X = np.asarray(X)
	Y = np.asarray(Y)
	# print("HSIC", HSIC)


	# 2d case
#	X_2d = [[100, -400], [101, -404], [102, -408], [103, -412], [400,-1600]]
	X_2d = [[1, 1], [2, 2]]

	torch_X_2d = torch.tensor(X_2d, dtype =torch.float)
	HSIC_2d, (width_x_2d, width_x_2d_) = normalized_HSIC(torch_X_2d, torch_X_2d, return_width=True, kernel_param_average_method='median')

	print(width_x_2d, width_x_2d, "widths_x_2d")
	X_2d = np.asarray(X_2d)
	print("HSIC_X 2d", HSIC_2d)


	# # figure (1d)
	# plt.figure(figsize=(20, 4))
	#
	# plt.xlabel('x')
	# sigma_x = width_x
	# sigma_y = width_y
	#
	# xlinspace = np.linspace(-3*sigma_x, 3*sigma_x, 100)
	# kernel_X, kernel_Y = gaussian_kernel(xlinspace, sigma_x)
	# for i, x in enumerate(X) :
	# 	#print(kernel_Y, "Y values")
	# 	plt.plot(kernel_X+x, kernel_Y, color=np.random.rand(1,3), linewidth=0.5)  # #4799FF blue
	#
	# plt.scatter(X, zero, color='red')
	# x_axis_min = min( min(X)-(max(X)-min(X))*0.1, min(X)-3*sigma_x)
	# x_axis_max = max( max(X)+(max(X)-min(X))*0.1, max(X)+3*sigma_x)
	# plt.axis((x_axis_min, x_axis_max, min(kernel_Y)-(max(kernel_Y)-min(kernel_Y))*0.1, max(kernel_Y)+(max(kernel_Y)-min(kernel_Y))*0.1 ))
	# plt.show()

	# figure (2d)
	plt.figure(figsize=(10, 10))
	plt.xlabel('x1')
	plt.ylabel('x2')

	sigma_x_2d = width_x_2d # scalar case

	n_grid = 51
	x1linspace = np.linspace(-3*sigma_x_2d, 3*sigma_x_2d, n_grid)
	x2linspace = np.linspace(-3*sigma_x_2d, 3*sigma_x_2d, n_grid)
	x1x1, x2x2 = np.meshgrid(x1linspace, x2linspace)



	kernel_Y = np.exp(- (x1x1*x1x1 + x2x2*x2x2)/ (sigma_x_2d)**2 ) # scalar case
#	print(kernel_Y[:10], "kernel_Y[:10]")

	x1_min = 0
	x1_max = 0
	x2_min = 0
	x2_max = 0

	step = 0.02
	m = torch.amax(kernel_Y)
	levels = np.arange(0.0, m, step) + step
	cmap = ["Reds", "Blues", "Greens", "Purples", "Oranges"]

	from matplotlib.colors import Normalize
	from matplotlib.colors import LinearSegmentedColormap

	ncolors = 256
	color_array = plt.get_cmap('gist_rainbow')(range(ncolors))
	color_array[:, -1] = np.linspace(0.05, 0.7, ncolors)
	# create a colormap object
	map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha', colors=color_array)
	# register this new colormap with matplotlib
	plt.register_cmap(cmap=map_object)

	alphas = Normalize(0, .3, clip=True)(np.abs(kernel_Y))
	alphas = np.clip(alphas, .4, 1)  # alpha value clipped at the bottom at .4


	for i, x in enumerate(X_2d) :
		# contour = plt.contour(x[0]+x1x1, x[1]+x2x2, kernel_Y, cmap = "rainbow_alpha", levels=7, alpha=0.6, linewidths=1, vmin=0)
		contourf = plt.contourf(x[0] + x1x1, x[1] + x2x2, kernel_Y, cmap="rainbow_alpha", levels=levels, vmin=0)
		#if i == 0 :
		#	plt.clabel(contour, inline=True, fontsize=10)
		plt.title('isotropic kernel')

	x1_min = min(X_2d[:,0])
	x1_max = max(X_2d[:,0])
	x2_min = min(X_2d[:,1])
	x2_max = max(X_2d[:,1])

	x1_axis_min = min( x1_min - (x1_max - x1_min)*0.2, min(X_2d[:,0])-2*sigma_x_2d)
	x1_axis_max = max( x1_max + (x1_max - x1_min)*0.2, max(X_2d[:,0])+2*sigma_x_2d)
	x2_axis_min = min( x2_min - (x2_max - x2_min)*0.2, min(X_2d[:,1])-2*sigma_x_2d)
	x2_axis_max = max( x2_max + (x2_max - x2_min)*0.2, max(X_2d[:,1])+2*sigma_x_2d)


	plt.axis((x1_axis_min, x1_axis_max, x2_axis_min, x2_axis_max))
	plt.scatter(X_2d[:,0],X_2d[:,1], color='red')
	plt.colorbar(contourf)

	plt.show()







