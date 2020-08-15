# GCN-for-Synergy
Combination therapy has multiple advantages in efficacies and dosage. However, traditional identification of effective drug combinations is often serendipitous. With the rapid growth of vast volumes of biological databases, various computational methods such as systems biology models and machine learning models are used in exploring the combination space. Deep Neural Networks also have been used in predicting synergy score of drugs pairs. Drug combinations efficacy are related to a various type of interactions which underlying biological process. Nevertheless, classic deep learning algorithm such as Convolutional Neural Network fail to extract the information of topological structures. Graph neural networks (GNNs), a kind of deep learning methods that have powerful abilities of graph representation, have been a widely applied graph analysis method in various fields. We extracted the features of these graphs with GNNs and built a model for predicting synergy of drug combinations on diverse cell lines. The model can predict synergy of novel drug combinations with the highest ROC AUC, PR AUC and accuracy over the five test sets. Some of our predicted synergy scores are in line with the previous studies. 

# Code
Example code for model optimization (main.py) and testing (main for testing.py) are provided. 

# Data
Data for the example code are stored in dataset folder.

# Citation
