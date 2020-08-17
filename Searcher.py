import h5py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import time
from VGGNet import VGGNet

query = 'ImageSource/2.jpg'
index = 'Models/vgg_feature_oxbuild.h5'
result = 'oxbuild_images'

h5f = h5py.File(index, 'r')
feats = h5f['features'][:]
imgNames = h5f['names'][:]
h5f.close()

print('-----------------------------------')
print('searching starts')
print('-----------------------------------')

queryImg = mpimg.imread(query)
plt.title("Query Image")
plt.imshow(queryImg)
plt.show()

model = VGGNet()

print('start: ' + str(time.time()))

queryVec = model.vgg_extract_feat(query)
scores = np.dot(queryVec, feats.T)
rank_ID = np.argsort(scores)[::-1]
rank_score = scores[rank_ID]

maxres = 5
imlist = []
for i, index in enumerate(rank_ID[0:maxres]):
    imlist.append(imgNames[index])
    print('image names: ' + str(imgNames[index]) + ' scores: %f' % rank_score[i])
print('top %d images in order are: ' % maxres, imlist)

print('end: ' + str(time.time()))

for i, im in enumerate(imlist):
    image = mpimg.imread(result + '/' + str(im, 'utf-8'))
    plt.title("search output %d" % (i + 1))
    plt.imshow(image)
    plt.show()
