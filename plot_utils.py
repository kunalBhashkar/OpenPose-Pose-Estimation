import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import numpy as np
from PIL import Image
import math


#subplot utils
def add_outline_path_effect(o, lw=4):
    o.set_path_effects([path_effects.Stroke(linewidth=lw, foreground='black'),
                       path_effects.Normal()])
    
def add_text_to_subplot(ax, pos, label, size='x-large', color='white'):
    text = ax.text(pos[0], pos[1], label, size=size, weight='bold', color=color, va='top')
    add_outline_path_effect(text, 2)
    
def hide_subplot_axes(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def plot_image_tensor_in_subplot(ax, img_tensor):
    ima= img_tensor.cpu().numpy().transpose((1,2,0))
    ax.imshow(ima)

def show_img_in_subplot(ax, pil_image):
    im = np.array(pil_image)
    ax.imshow(im)
    
def plot_bbox_annotation(ax, bb, cat_label):
    draw_bbox(ax, bb)
    add_text_to_subplot(ax, (bb[0], bb[1]), cat_label)

def tensor_to_scalar(t):
    if t.dim()==0:
        return t.item()
    else:
        return t.numpy()

def plot_horizontal_bar_chart(counts, labels, title='', x_tick_step=200):    
    sorted_items = sorted(zip(counts, labels), reverse=True)
    sorted_counts, sorted_labels = zip(*sorted_items)
    
    y_pos = np.arange(len(sorted_labels))
    
    fig, ax = plt.subplots(figsize=(10,30))
    ax.barh(y_pos, sorted_counts)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_labels)
    ax.set_xticks(range(0, int(sorted_counts[0]) + x_tick_step, x_tick_step))
    ax.invert_yaxis() 
    ax.set_facecolor('#f7f7f7')
    ax.set_title(title)
    
    for idx, val in enumerate(ax.patches):
        x_value = val.get_width() + 5
        y_value = 0.1 + val.get_y() + val.get_height()/2
        ax.text(x_value, y_value, int(sorted_counts[idx]))

    plt.show()        
    
def plot_model_predictions_on_sample_batch(batch, pred_labels, actual_labels, get_label_fn, n_items=12, plot_from=0, figsize=(16,12)):
    n_rows, n_cols = (1,n_items) if n_items<=4 else (math.ceil(n_items/4), 4)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i,ax in enumerate(axes.flat):
        plot_idx = plot_from + i
        plot_image_tensor_in_subplot(ax, batch[plot_idx])

        pred_label = get_label_fn(tensor_to_scalar(pred_labels[plot_idx])) 
        actual_label = get_label_fn(tensor_to_scalar(actual_labels[plot_idx]))  

        hide_subplot_axes(ax)
        add_text_to_subplot(ax, (0,0), 'Pred: '+pred_label)
        add_text_to_subplot(ax, (0,30), 'Actual: '+actual_label, color='yellow')

    plt.tight_layout()
    
       
# plots used in multi class classifier   
def add_bar_height_labels(ax, rects):
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = int(rect.get_height())
        ax.text(rect.get_x() + rect.get_width()*offset['center'], 1.01*height,
                '{}'.format(height), ha=ha['center'], va='bottom')
          

def plot_model_result_on_test_image(pred_bbox, pred_cat_id, get_label_fn, im_path):
    im = Image.open(im_path)
    w,h = im.size
    fig, ax = plt.subplots(1, 1)
    
    bbox = pred_bbox[0].clone()
    bbox = bbox/224
    bbox[0] = bbox[0]*h
    bbox[1] = bbox[1]*w
    bbox[2] = bbox[2]*h
    bbox[3] = bbox[3]*w
    
    pred_bbox = [int(x) for x in yxyx_to_xywh(bbox)]
    
    show_img_in_subplot(ax, im)
    draw_bbox(ax, pred_bbox)

    add_text_to_subplot(ax, (pred_bbox[0], pred_bbox[1]), 'Pred:'+get_label_fn(tensor_to_scalar(pred_cat_id[0])))
    hide_subplot_axes(ax)

    plt.tight_layout()
    plt.show()
