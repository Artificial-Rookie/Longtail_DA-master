# **Before running**

**check parser parameters: dataset, num_meta, epochs**

**check line 88 whether using full_data**

**check line 220 maximum epoch number**

**check line 223 train(imbalanced_train_loader) which loader should be used**

****

## **In either datasets, validation loader is the balanced dataset which should not be changed**

# cifar 10

1. **dataset**: sample num for each class  5000, total 10 classes. num_meta default is 10
2. **curation**: 
   - split out the meta data (balanced set): train_data_meta and get train_data in data_utils.py
   - then create the imbalance problem with train_data, get remaining set imbalanced_train_dataset
   - full dataset should could be gained by curating the train_dataset with imabalance factor (set the full_data=True at line 88)

# BDD100K

1. **dataset**: 10 classes, only one class 95 samples, others over 2k. num_meta set as 2k
2. **curation**:
   - split out the meta_data in data_utils.py just like cifar10, further curation is not necessary, the imbalanced_train_loader is the remaining set
   - full dataset training should use the full_train_loader for only the first stage with 200 epoch
   - two stage training should use imbalanced_train_loader with 160 first, 40 second



