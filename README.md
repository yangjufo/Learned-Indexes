# Learned Indexes
Implementation of BTree part for paper 'The Case for Learned Index Structures'.
>Language: Python  
Support content: Integer values, Random and Exponential distribution

## Files Structures
> data/: test data  
model/: learned NN model  
perfromance/：NN and BTree performance  
Learned_BTree.py: main file  
Trained_NN.py: NN structures

## HOW TO RUN
> First, you need to install python2.7.x and package tensorflow, pandas, numpy, enum.   
Second, use command to run the Learned_BTree.py fule, that is,  
```python Learned_BTree.py -t <Type> -d <Distribution> [-p|-n] [Percent]|[Number] [-c] [New data] [-h]```.  
  
>Parameters:  
'type': 'Type: sample, full',  
'distribution': 'Distribution: random, exponential',  
'percent': 'Percent: 0.1-1.0, default value = 0.5; sample train data size = 300,000',  
'number': 'Number: 10,000-10,000,000, default value = 300,000',  
'new data' 'New Data: INTEGER, 0 for no creating new data file, others for creating'  
  
>Example:  
```python Learned_BTree.py -t full -d random -n 100000 -c 1```  
  

## Other Content
### Sample Training
> Sample training is also included in this project, you can use parameter 'sample' for '-t' to test sample training, while '-p' is used for change the sample training percent.  
  
>Example:  
```python Learned_BTree.py -t sample -d random -p 0.3 -c 0```
### Storage Optimization
>More Information will be added soon.
