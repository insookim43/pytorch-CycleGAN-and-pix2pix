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

	H = torch.exp(-H/2/(deg**2))

	return H


def hsic_gam(X, Y, alph = 0.5):
	"""
	X, Y are numpy vectors with row - sample, col - dim
	alph is the significance level
	auto choose median to be the kernel width
	"""
        #print(X.shape, "X.shape")
        #print(Y.shape, "Y.shape")
	X = torch.reshape(X, (X.shape[0], -1))
	Y = torch.reshape(Y, (Y.shape[0], -1))
	n = X.shape[0]

	# ----- width of X -----
	Xmed = X
	print(Xmed.shape, "Xmed shape")

	# print(Xmed*Xmed.shape, "Xmed*Xmed.shape")
	# print(torch.matmul(Xmed.T,Xmed).shape, "Xmed.T @ Xmed. shape")
	# print(torch.sum(Xmed*Xmed, 1).shape, "np.sum(Xmed*Xmed,1) shape")


	G = torch.sum(Xmed*Xmed,1).reshape(n,1)
	Q = torch.tile(G, (1, n) )
	R = torch.tile(G.T, (n, 1) )

	dists = Q + R - 2* torch.matmul(Xmed, Xmed.T)
	dists = dists - torch.tril(dists)
	dists = dists.reshape(n**2, 1)

	width_x = torch.sqrt( 0.5 * torch.median(dists[dists>0]) )
	# ----- -----

	# ----- width of Y -----
	Ymed = Y

	G = torch.sum(Ymed*Ymed, 1).reshape(n,1)
	Q = torch.tile(G, (1, n) )
	R = torch.tile(G.T, (n, 1) )

	dists = Q + R - 2* torch.matmul(Ymed, Ymed.T)
	dists = dists - torch.tril(dists)
	dists = dists.reshape(n**2, 1)

	width_y = torch.sqrt( 0.5 * torch.median(dists[dists>0]) )
	# ----- -----

	bone = torch.ones((n, 1)).float()
	H = torch.eye(n) - (torch.ones((n,n)) / n).float()

	print(H.device, "H.device()") 
	K = rbf_dot(X, X, width_x)
	bone = bone.to(K.device)
	print(K.device, "K.device()") 
	L = rbf_dot(Y, Y, width_y)

	H = H.to(K.device)
	Kc = torch.matmul(torch.matmul(H, K), H)
	Lc = torch.matmul(torch.matmul(H, L), H)

	print(H.device, "H.device()") 
	testStat = torch.sum(Kc.T * Lc) / n

	print(testStat.device, "testStat.device()") 
	varHSIC = (Kc * Lc / 6)**2

	print(varHSIC.device, "varHSIC.device()") 
	varHSIC = ( torch.sum(varHSIC) - torch.trace(varHSIC) ) / n / (n-1)

	print(varHSIC.device, "varHSIC.device()") 
	varHSIC = varHSIC * 72 * (n-4) * (n-5) / n / (n-1) / (n-2) / (n-3)

	print(varHSIC.device, "varHSIC.device()") 
	K = K - torch.diag(torch.diag(K))
	L = L - torch.diag(torch.diag(L))

	print(K.device, "K.device()") 
	print(L.device, "L.device()") 
	print(bone.T.device, "bone.T.device()") 
	print(torch.matmul(bone.T, K) , "torch.matmul(bone.T, K).device()") 
	muX = torch.matmul(torch.matmul(bone.T, K), bone) / n / (n-1)
	print(muX.device, "muX.device()") 
	muY = torch.matmul(torch.matmul(bone.T, L), bone) / n / (n-1)
	print(muY.device, "muY.device()") 

	mHSIC = (1 + muX *                                                                                                                            muY - muX - muY) / n

	al = (mHSIC**2 / varHSIC).cpu()
	bet = (varHSIC*n / mHSIC).cpu()

	thresh = gamma.ppf(1-alph, al, scale=bet)[0][0]

	return (testStat, thresh)

def normalized_HSIC(X, Y, alph = 0.5):
	HSIC_xx,_ = hsic_gam(X, X)
	HSIC_yy,_ = hsic_gam(Y, Y)
	return hsic_gam(X, Y, alph=alph)[0] / (torch.sqrt(HSIC_xx) * torch.sqrt(HSIC_yy))

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
