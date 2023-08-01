# SILO (S-shaped Improved Learning rate Optimization)

Code for the paper: Optimizing Learning Rate Schedules for Iterative Pruning of Deep Neural Networks

(1) Dependencies  Required:

* Numpy

* Pandas

* Pytorch

* Scikit-Learn

(2) How to Run the Code?

* To obtain results in Table.2, please run resnet_20_cifar_10.py.
  
* To obtain results in Table.3, please run vgg_19_cifar_10.py. 

  Please note that the CIFAR-10 dataset will be automatically downloaded once the code is running. The ImageNet-200 is not uploaded due to its large size. Lastly, all experimental results will be saved in a local csv file. Please check the csv file for detailed results. 

(3) Description of Each Folder:

* arch: store all different neural network architectures.
 
* plots: save all plots
 
* saves: save some temporal files/models
 
* runs: save some results in each run

* dumps: save some dumped files

 
