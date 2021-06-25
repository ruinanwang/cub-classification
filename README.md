## CUB Classification with Concept Bottleneck

### Dataset
Caltech-UCSD Birds-200- 2011 Dataset [download here](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).  
Please insert the `CUB_200_2011/images` folder from the download into our repo's `data/CUB_200_2011` folder

### Training

A machine with GPU is needed.  
For ete_train.py training file, run `ete_train.py -n [saving_name_required] -model_name [model_used_to_train_optional]`  
For cb_train.py training file, run `cb_train.py -n [saving_name_required] -m [1 or 2] -model_name [model_used_to_train_optional]`

### Files

`split_data`: used for preprocessing images and annotations.  
`dataloader`: used for both ete and cb models.  
`ete_models`: AlexNet, ResNet, VGG, DenseNet, InceptionV3 fine tuning models.  
`ete_train`: train end-to-end network.  
`ete_test`: test/evaluation end-to-end network.  
`cb_xxx`: similar to ete.  
`cb_train`: includes training image-to-annotation prediction and annotation-to-label prediction (2 models train separately).  
`cb_train2`: currently deprecated.  
`cb_combine_train`: training two models (image-to-annotation and annotation-to-label) together.  
`FullyConnectedModel`: used in cb_train (annotation-to-label prediction).  
`plot`: currently deprecated. We switched to tensor board which is embedded in the training files.  
`qualitative_eval`: qualitative evaluation after experiments were run.  

