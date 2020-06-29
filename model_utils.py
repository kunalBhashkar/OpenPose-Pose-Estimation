import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet34
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import numpy as np
#from logger import Logger

def get_trn_val_idxs(len_ds, val_percent=0.2):
    np.random.seed(1)
    idxs = np.random.permutation(len_ds)

    num_val_ex = int(len_ds*val_percent)
    train_idxs = idxs[:-num_val_ex]
    val_idxs = idxs[-num_val_ex:]
    return (train_idxs, val_idxs)

def freeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = False
        
def unfreeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = True        
        
def get_resnet34_model_with_custom_head(custom_head):
    model = resnet34(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-2])
    
    freeze_all_layers(model)
    
    model.add_module('custom head', custom_head)
    model = model.to(device)
    return model

def get_model_predictions_on_a_sample_batch(model, dl):
    model.eval()
    with torch.no_grad():
        batch, actual_labels = iter(dl).next()
        batch = batch.to(device)
        actual_labels = actual_labels.to(device)
        predictions = model(batch)
    
    return (predictions, batch, actual_labels)
        
def print_training_loss_summary(loss, total_steps, current_epoch, n_epochs, n_batches, print_every=10):
    #prints loss at the start of the epoch, then every 10(print_every) steps taken by the optimizer
    steps_this_epoch = (total_steps%n_batches)
    
    if(steps_this_epoch==1 or steps_this_epoch%print_every==0):
        print ('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}' 
               .format(current_epoch, n_epochs, steps_this_epoch, n_batches, loss))

#Test model on single image
def test_on_single_image(test_im_tensor, model, sz):
    model.eval()
    with torch.no_grad():
        preds = model(test_im_tensor)
        pred_bbox, pred_class_scores = preds[:,:4], preds[:, 4:]
        pred_bbox = torch.sigmoid(pred_bbox)*sz
        pred_cat_id = pred_class_scores.argmax(dim=1)
    return pred_bbox, pred_cat_id             
