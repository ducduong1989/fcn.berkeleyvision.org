from __future__ import division
import caffe
import numpy as np
import os
import sys
from datetime import datetime
from PIL import Image
import argparse
import time
import shutil

# The format to save image.
_IMAGE_FORMAT = '%06d_image'
# The format to save prediction
_PREDICTION_FORMAT = '%06d_prediction'

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(net, save_dir, dataset, layer='score', gt='label'):
    n_cl = net.blobs[layer].channels
    if save_dir:
        os.mkdir(save_dir)
    hist = np.zeros((n_cl, n_cl))
    loss = 0
    for idx in dataset:
        net.forward()
        hist += fast_hist(net.blobs[gt].data[0, 0].flatten(),
                                net.blobs[layer].data[0].argmax(0).flatten(),
                                n_cl)

        if save_dir:
            im = Image.fromarray(net.blobs[layer].data[0].argmax(0).astype(np.uint8), mode='P')
            im.save(os.path.join(save_dir, idx + '.png'))
        # compute the loss as well
        loss += net.blobs['loss'].data.flat[0]
    return hist, loss / len(dataset)

def seg_tests(test_net, save_format, dataset, logfile, layer='score', gt='label'):
    print '>>>', datetime.now(), 'Begin seg tests'    
    do_seg_tests(test_net, 0, save_format, dataset, logfile, layer, gt)

def do_seg_tests(net, iter, save_format, dataset, logfile, layer='score', gt='label'):    
    log_data = list()
    n_cl = net.blobs[layer].channels
    if save_format:
        save_format = save_format.format(iter)
    hist, loss = compute_hist(net, save_format, dataset, layer, gt)
    # mean loss
    print '>>>', datetime.now(), 'Iteration', iter, 'loss', loss
    log_data.append(datetime.now().strftime('%m/%d/%Y/%H-%M-%S') + ', Iteration: ' + str(iter) + 'loss: ' + str(loss) + '\n')
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'overall accuracy', acc
    log_data.append(datetime.now().strftime('%m/%d/%Y/%H-%M-%S') + ', Iteration: ' + str(iter) + 'accuracy: ' + str(acc) + '\n')
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print '>>>', datetime.now(), 'Iteration', iter, 'mean accuracy', np.nanmean(acc)
    log_data.append(datetime.now().strftime('%m/%d/%Y/%H-%M-%S') + ', Iteration: ' + str(iter) + 'mean accuracy: ' + str(acc) + '\n')
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', datetime.now(), 'Iteration', iter, 'mean IU', np.nanmean(iu)
    log_data.append(datetime.now().strftime('%m/%d/%Y/%H-%M-%S') + ', Iteration: ' + str(iter) + 'mean IU: ' + str(np.nanmean(iu)) + '\n')
    freq = hist.sum(1) / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'fwavacc', \
            (freq[freq > 0] * iu[freq > 0]).sum()
    
    with open(logfile, 'w') as handler:
        handler.writelines(log_data)
    
    return hist


def inference(deploy_prototxt_file, train_model, class_input_images_dir, class_output_dir):    
    assert os.path.exists(class_input_images_dir), class_input_images_dir + " does not exist. Please check it."
    shutil.rmtree(class_output_dir, ignore_errors=True)
    os.makedirs(class_output_dir)
    # load net
    net = caffe.Net(deploy_prototxt_file, train_model, caffe.TEST)    
    mapping_file = os.path.join(class_output_dir, "mapping_file.csv")

    image_paths = os.listdir(class_input_images_dir)      
    ls_image_name = list()
    ls_pred_name = list()  
    for index, image_path in enumerate(image_paths):  
      print(image_path)
      image_path_full = os.path.join(class_input_images_dir, image_path)
      # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe    
      im = Image.open(image_path_full).convert('RGB')
      in_ = np.array(im, dtype=np.float32)
      in_ = in_[:,:,::-1]
      in_ -= np.array((104.00699,116.66877,122.67892))
      in_ = in_.transpose((2,0,1))
      # shape for input (data blob is N x C x H x W), set data
      net.blobs['data'].reshape(1, *in_.shape)
      net.blobs['data'].data[...] = in_
      # run net and take argmax for prediction
      net.forward()
      out = net.blobs['score'].data[0].argmax(axis=0)
      segmap = out.astype(np.uint8)   
      # Save prediction image
      prediction_filename = _PREDICTION_FORMAT % index
      pil_image = Image.fromarray(segmap)
      pil_image.save(os.path.join(class_output_dir, prediction_filename + ".png"), 'PNG')
      # Save original image
      image_filename = _IMAGE_FORMAT % index     
      pil_image = Image.open(image_path_full)    
      pil_image.save(os.path.join(class_output_dir, image_filename + ".png"), 'PNG')        
      # Save mapping information
      print("image name: ", image_path.split(".")[0])
      print("prediction_filename: ", prediction_filename)      
      ls_image_name.append(image_path.split(".")[0])
      ls_pred_name.append(prediction_filename)
    
    with open(mapping_file, 'w') as handler:
      handler.write("origin,generated\n")
      for ind in range(len(ls_pred_name)):
        handler.write(ls_image_name[ind] + "," + ls_pred_name[ind] + "\n")


def main():
    start = time.time()
    parser = argparse.ArgumentParser(description='Evaluate mIoU score of trained model')
    parser.add_argument('--from_scratch',
                        action='store_true',
                        help='Whether use model trained from scratch or user model with 21 classes.')    
    parser.add_argument('--weight',
                        type=str,
                        help='path to trained model.')    
                                            
    args = parser.parse_args()
    from_scratch = args.from_scratch
    train_file = '/input_medical_data/tosei/ImageSets/Segmentation/trainval.txt'
    val_file = '/input_medical_data/KCRC/ImageSets/Segmentation/trainval.txt'
    if from_scratch:
        train_prototxt_file = 'train.prototxt'
        val_prototxt_file = 'val.prototxt'
        deploy_prototxt_file = 'deploy.prototxt'     
        print ("Using trained model from Scratch.........")
    else:
        train_prototxt_file = 'train_origin.prototxt'
        val_prototxt_file = 'val_origin.prototxt'    
        deploy_prototxt_file = 'deploy_origin.prototxt'
        print ("Using trained model from pretrained model on Pascal-VOC.........")
    
    assert os.path.isfile(val_file), val_file + " does not exist. Please check it."
    assert os.path.isfile(train_file), train_file + " does not exist. Please check it."
    assert os.path.isfile(train_prototxt_file), train_prototxt_file + " does not exist. Please check it."
    assert os.path.isfile(val_prototxt_file), val_prototxt_file + " does not exist. Please check it."
    train_model = args.weight
    assert os.path.isfile(train_model), train_model + " does not exist. Please check it."
    #validate on val set
    val_dataset = list()
    with open(val_file, 'r') as handler:
        for line in handler:
            val_dataset.append(line.strip())            
    val_logfile = os.path.join('/output_medical_data', 'val_log_' + str(time.time()).replace(".", "") + '.txt')
    val_net = caffe.Net(val_prototxt_file, train_model, caffe.TEST)    
    seg_tests(test_net=val_net, save_format=None, dataset=val_dataset, logfile=val_logfile)

    #validate on train set
    train_dataset = list()
    with open(train_file, 'r') as handler:
        for line in handler:
            train_dataset.append(line.strip())            
    train_logfile = os.path.join('/output_medical_data', 'train_log_' + str(time.time()).replace(".", "") + '.txt')
    train_net = caffe.Net(train_prototxt_file, train_model, caffe.TEST)    
    seg_tests(test_net=train_net, save_format=None, dataset=train_dataset, logfile=train_logfile)

    tosei_class_input_images_dir = '/input_medical_data/tosei/PNGImages'
    tosei_class_output_dir = '/output_medical_data/tosei/segmentation_results'
    kcrc_class_input_images_dir = '/input_medical_data/KCRC/PNGImages'
    kcrc_class_output_dir = '/output_medical_data/KCRC/segmentation_results'
    inference(deploy_prototxt_file, train_model=train_model, class_input_images_dir=tosei_class_input_images_dir,
                class_output_dir=tosei_class_output_dir)
    inference(deploy_prototxt_file, train_model=train_model, class_input_images_dir=kcrc_class_input_images_dir,
                class_output_dir=kcrc_class_output_dir)   
                
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")


if __name__ == '__main__':
    main()