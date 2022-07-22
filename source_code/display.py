import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def display_im(paths, 
                out_shape = [224, 224], 
                num_rows = 1, 
                show_im = True,
                print_file = None):
    images  = []
    for im_path in paths:
        im = Image.open(im_path)
        im = im.resize(out_shape)
        images.append(np.array(im))
            

    num_cols = len(paths) // num_rows
    out_im = np.zeros([num_rows * out_shape[0], num_cols * out_shape[1], 3], dtype = 'uint8')

    cur = 0
    for i in range(num_rows):
        for j in range(num_cols):
            cur_row = i * out_shape[0]
            cur_col = j * out_shape[1]
            if images[cur].shape[0] * images[cur].shape[1] == images[cur].size:
                images[cur] = np.repeat(images[cur][:,:,None], 3, axis=2)
            out_im[cur_row: cur_row + out_shape[0],  cur_col: cur_col + out_shape[1], :] = images[cur][:,:,:3]
            cur = cur + 1

    
    if show_im:
        plt.imshow(out_im)
        plt.axis('off')

    if not (print_file is None):
        im = Image.fromarray(out_im)
        im.save(print_file)

    return images


def display_minst(images, 
                out_shape = [28, 28], 
                num_rows = 1, 
                show_im = True,
                print_file = None):

    images = np.reshape(images, [-1] + out_shape)
    images = np.concatenate(images, axis=1)
    
    num_cols = images.shape[1] // num_rows
    cur = 0
    all_sub_im = []
    for i in range(num_rows):
        sub_im = images[:,  cur: (cur + num_cols)]
        all_sub_im.append(sub_im)
        cur = cur + num_cols
    images = np.concatenate(all_sub_im, axis=0)
    

    
    # convert to balck and white
    images = np.repeat(images[:,:,None], 3, axis=2)
    images = images - np.min(images)
    images = images / np.max(images)
    images = np.uint8(images * 255)
    
    if show_im:
        plt.imshow(images)
        plt.axis('off')

    if not (print_file is None):
        im = Image.fromarray(images)
        im.save(print_file)

    return images
