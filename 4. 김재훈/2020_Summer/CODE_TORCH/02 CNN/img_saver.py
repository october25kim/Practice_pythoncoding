import os
import numpy as np
import gzip

gz_origin_path = 'D:/CIFAR10/ORIGINAL'
gz_saved_path = 'D:/CIFAR10/DATASET'
folder_names = ['train', 'test', 'val']
f_origin_lst = os.listdir(gz_origin_path)
for folder_name in folder_names:
    os.makedirs(os.path.join(gz_saved_path, folder_name), exist_ok=True)

# dict_keys([b'batch_label', b'labels', b'data', b'filenames'])

def load_img(path):

    import pickle
    with open(path, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict


### save train & validation set ###

cifars_set_data = list()
cifars_set_label = list()
for f in [_ for _ in f_origin_lst if 'data_batch' in _]:

    f_path = os.path.join(gz_origin_path,f)
    cifars = load_img(f_path)

    cifars_set_data.append(cifars[b'data'].reshape((-1,3,32,32)).transpose(0,2,3,1).astype(np.float32) / 255.)
    cifars_set_label.append(cifars[b'labels'])

sample_imgs = np.concatenate(cifars_set_data, axis=0)
sample_labels = np.concatenate(cifars_set_label, axis=0) 

sample_length = sample_labels.shape[0]

train_val_cut_idx = int(sample_length * 0.8)

#train save
for idx, (l, d) in enumerate(zip(sample_labels[:train_val_cut_idx], sample_imgs[:train_val_cut_idx])):
    np.savez(f'D:/CIFAR10/DATASET/train/train_{idx}', label=l, img=d)
print('train save finished')


#val save
for idx, (l, d) in enumerate(zip(sample_labels[train_val_cut_idx:], sample_imgs[train_val_cut_idx:])):
    np.savez(f'D:/CIFAR10/DATASET/val/val_{idx}', label=l, img=d)
print('val save finished')




### save test set ###

cifars_set_data = list()
cifars_set_label = list()
for f in [_ for _ in f_origin_lst if 'test_batch' in _]:

    f_path = os.path.join(gz_origin_path,f)
    cifars = load_img(f_path)

    cifars_set_data.append(cifars[b'data'].reshape((-1,3,32,32)).transpose(0,2,3,1).astype(np.float32) / 255.)
    cifars_set_label.append(cifars[b'labels'])

sample_imgs = np.concatenate(cifars_set_data, axis=0)
sample_labels = np.concatenate(cifars_set_label, axis=0) 

sample_length = sample_labels.shape[0]

#test save
for idx, (l, d) in enumerate(zip(sample_labels[:], sample_imgs[:])):
    np.savez(f'D:/CIFAR10/DATASET/test/test_{idx}', label=l, img=d)
print('test save finished')
