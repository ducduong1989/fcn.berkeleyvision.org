export OUTPUT_IPF_DATASET="/mnt/data2/manvd/fcn.berkeleyvision.org/data/ipf"
export FCN_ALEXNET_OUTPUT="/mnt/data2/manvd/fcn.berkeleyvision.org/outputs/viz"
export PRETRAINED_PATH="/mnt/data2/manvd/fcn.berkeleyvision.org/pretrained"
export TRAINED_WEIGHTS_OUTPUTS="/mnt/data2/manvd/fcn.berkeleyvision.org/outputs/weights"


sudo docker run --rm -it \
-v ${OUTPUT_IPF_DATASET}:/input_medical_data \
-v ${FCN_ALEXNET_OUTPUT}:/output_medical_data \
-v ${PRETRAINED_PATH}:/pretrained_model \
-v ${TRAINED_WEIGHTS_OUTPUTS}:/output_segmentor \
-v ${PWD}:/workspace \
--gpus all --shm-size 64G voc_fcn_alexnet_caffe:1.0
