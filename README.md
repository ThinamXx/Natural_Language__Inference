# **NATURAL LANGUAGE INFERENCE**

### **INTRODUCTION:**
- **Natural Language Inference** is a study where a hypothesis can be inferred from a premise where both are a text sequence. It determines the logical relationship between a pair of text sequences.

### **LIBRARIES AND DEPENDENCIES:**
- I have downloaded all the libraries and dependencies required for the project in one particular cell.

```python
from d2l import torch as d2l
import os, re
import torch     
from torch import nn      
from torch.nn import functional as F
from IPython import display
```

### **GETTING THE DATASET:**
- I have used google colab for this notebook so the process of downloading and reading the data might be different in other platforms. I will use **Stanford Natural Language Inference Corpus** for this project. The SNLI Corpus is a collection of over 500000 labeled english pairs.

### **READING THE DATASET:**
- I will define a function to only extract part of the dataset and then return list of premises, hypothesis and their labels. I have presented the implementation of **Reading SNLI Dataset** using PyTorch here in the snapshot.

![Image](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%20179.PNG)

### **LOADING THE DATASET:**
- I will define a class for loading the SNLI Dataset. The num steps argument in the class constructor specifies the length of a text sequence so that each minibatch of sequences will have the same shape. The token sequences which are longer than num steps are trimmed while special tokkens are appended to shorter sequences. I have presented the implementation of **Loading SNLI Dataset** using PyTorch here in the snapshots. 

![IMAGE](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%20180.PNG)
![Image](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%20180a.PNG)

### **ATTENDING CLASS:**
- I will align words in one text sequnce to each word in other sequence. I will implement soft alignment using attention mechanism. I will define the Attend Class to compute the soft alignment of the hypotheses beta with input premises and soft alignment of premises alpha with input hypotheses. I have presented the implementation of MLP and Attention Mechanism using PyTorch here in the snapshot. 

![Image](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%20181.PNG)

