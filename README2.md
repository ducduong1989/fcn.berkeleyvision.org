# FCN Alexnet Caffe training/validation/Viz

1. Redirect to project folder (./fcn.berkeleyvision.org) <br />
2. Build docker image: sudo docker build -t voc_fcn_alexnet_caffe:1.0 .  <br />
3. Prepare input data for tosei and KCRC datasets on host (local) machine as result of pre-processing step: <br />
    Output structure: <br />
    output_ipf_dataset/ (for example: /mnt/data2/output_ipf_dataset) <br />
                      /tosei/ <br />
                            /ImageSets/Segmentation/ <br /> 
                            /PNGImages/ <br />
                            /SegmentationClass/ <br />
                            /AI_ILD_list_final.csv <br />
                            /resultrf.csv <br />
                      /KCRC/ <br />
                           /ImageSets/Segmentation/ <br />
                           /PNGImages/ <br />
                           /SegmentationClass/ <br />
                           /df_test.csv <br />

4. Create directory where stores trained weights output (trained_weight_output) (fcn_alexnet_trained_weights). for example: /mnt/data2/fcn_alexnet_trained_weights <br />
5. Create directory where stores visualization output (fcn_alexnet_output). for example: /mnt/data2/fcn_alexnet_output <br />

6. Download fcn_alexnet caffe pretrained model and extract to (http://dl.caffe.berkeleyvision.org/fcn-alexnet-pascal.caffemodel) pretrained_path for example: /mnt/data2/data/voc_fcn_caffe/pretrained <br />
7. Edit solver.prototxt for training from scratch or solver_origin.prototxt for training from pretrained Pascal-VOC to change hyperparameters (max_iter, test_interval, snapshot-iteration iterval to save weight....) 
8. Run docker container: <br />    
    - sudo docker run -it -v {output_ipf_dataset}:/input_medical_data -v {fcn_alexnet_output}:/output_medical_data -v {pretrained_path}:/pretrained_model -v {trained_weight_output}:/output_segmentor -v {fcn_alexnet_output}:/output_medical_data  --gpus all --shm-size 64G voc_fcn_alexnet_caffe:1.0 <br />    
    - for example: <br />
    docker run -it -v /mnt/data2/data/IPF:/input_medical_data -v /mnt/data2/data/log:/output_medical_data -v /mnt/data2/data/voc_fcn_caffe/trained_alexnet:/output_segmentor -v /mnt/data2/data/voc_fcn_caffe/pretrained:/pretrained_model --gpus all --shm-size 64G voc_fcn_alexnet_caffe:1.0 <br />

9. Train FCN Alexnet caffe and generate mask with 4 classes from scratch <br />
    9.1. Train:  <br />
    - **caffe train --solver=solver.prototxt --gpu=0**  <br /> 

    9.2. Eval and generate mask:  <br />
    - python **validate_n_generate_mask.py --weight=/output_segmentor/train_iter_40000.caffemodel --from_scratch** <br>
    - IoU per each class in val/test will be saved at {fcn_alexnet_output}/validation_iou.txt & {fcn_alexnet_output}/test_iou.txt <br>
  
    9.3. Generate colored mask for visualization  <br />
    - python **generate_color_mask.py --weight=/output_segmentor/train_iter_40000.caffemodel --from_scratch** <br />

10. Train FCN Alexnet caffe and generate mask with 21 classes from Pascal-VOC with [0,1,2,3] as our data <br />
9.1. Train:  <br />
    - **caffe train --solver=solver_origin.prototxt --weights=/pretrained_model/fcn-alexnet-pascal.caffemodel --gpu=1** <br>
  
    9.2. Eval and generate mask:  <br />
    - **python validate_n_generate_mask.py --weight=/output_segmentor/train_iter_40000.caffemodel** <br>
    - IoU per each class in val/test will be saved at {fcn_alexnet_output}/validation_iou.txt & {fcn_alexnet_output}/test_iou.txt <br>
   
    9.3. Generate colored mask for visualization  <br />
    - **python generate_color_mask.py --weight=/output_segmentor/train_iter_40000.caffemodel** <br>


