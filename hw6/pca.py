from skimage import io
import numpy as np
from numpy import matlib
from numpy import linalg
from skimage import transform
import sys

#img = np.zeros((1080000,415))
pixel = 600
pixsqr = pixel*pixel
print(pixel,pixsqr)
img = np.zeros((pixsqr*3,415))
a=sys.argv[1]
for i in range (415):
	file_name = a+'/'+str(i)+".jpg"
	#file_name = "Aberdeen/"+str(i)+".jpg"
	temp = io.imread(file_name)
	#resize only test
	#temp = transform.resize(temp, (pixel,pixel,3))
	temp = temp.flatten()
	img[:,i]=temp


#img = transform.resize(img, (43200,415))
img_mean = np.mean(img, axis=1)

for i in range (415):
	img[:,i] = img[:,i] - img_mean

print ("SVD is my wifi")
U, s, V = np.linalg.svd(img, full_matrices=False)
k = 4
eigenface = U[:,0:k]

#import matplotlib.pyplot as plt
#io.use_plugin('matplotlib', 'imread')
# rebuild img
inputimg = sys.argv[2]
print(inputimg)
i = inputimg.replace(".jpg", "")
print(i)
i = int(float(i))
print(i)
for i in range(i,i+1,1):
	print("hi")
	img_number = i
	print(img_number)
	weight = np.dot(np.transpose(img[:,img_number]),eigenface)
	new_img = np.zeros((pixsqr*3,))
	for i in range(k):
		new_img = new_img + eigenface[:,i]*weight[i]
	#new_img = eigenface[:,1]*weight[1]
	#new_img = np.dot(eigenface,weight)
	new_img = new_img + img_mean
	new_img -= np.min(new_img)
	new_img /= np.max(new_img)
	new_img = (new_img * 255).astype(np.uint8)
	#new_img = new_img.reshape(60,60,3)
	#new_img = transform.resize(new_img, (120,120,3))
	new_img = new_img.reshape(pixel,pixel,3)
	file_name = "reconstruction.jpg"
	#file_name = "test"+str(img_number)+".png"
	io.imsave(file_name, new_img)
	#print (weight)
#plot mean1st	
#io.imsave("img_mean.png",img_mean.reshape(120,120,3))


'''
for i in range(k):
	new_img = eigenface[:,i]
#	new_img = new_img + img_mean
	new_img -= np.min(new_img)
	new_img /= np.max(new_img)
	new_img = (new_img * 255).astype(np.uint8)
	#new_img = new_img.reshape(60,60,3)
	new_img = new_img.reshape(120,120,3)
	#new_img = transform.resize(new_img, (120,120,3))
	file_name = "eigenface"+str(i+1)+".png"
	io.imsave(file_name, new_img)

'''

