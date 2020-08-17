import os
import numpy as np
import h5py

from VGGNet import VGGNet


def get_img_list(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


if __name__ == '__main__':
    databse = 'oxbuild_images'
    index = "Models/vgg_feature_oxbuild.h5"
    img_list = get_img_list(databse)

    print('------------------------------------------')
    print('feature extraction starts')
    print('------------------------------------------')

    feats = []
    names = []

    model = VGGNet()
    for i, img_path in enumerate(img_list):
        norm_feat = model.vgg_extract_feat(img_path)
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name)
        print("extracting feature from image No. %d, %d images in total" % ((i + 1), len(img_list)))

    feats = np.array(feats)
    output = index
    print('--------------------------------------------------')
    print('writing feature extraction results ...')
    print('--------------------------------------------------')

    h5f = h5py.File(output, 'w')
    h5f.create_dataset('features', data=feats)
    h5f.create_dataset('names', data=np.string_(names))
    h5f.close()
