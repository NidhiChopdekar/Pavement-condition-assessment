# PavemnetConditionPrediction

Pavement Condition Index is required to be performed to identify the road condition and further to take the maintenance or upgradation of the work.
This AI assisted module would be able to automatically assess the picture and identify common issues such as potholes, alligator cracking, block cracking etc.
Thus based on the road photographs, the model will be able to predict the pavement condition index.

Dataset structure->
dataset

  --alligator
  
     --low(l)
     --medium(m)   
     --high(h)     
     --not applicable(na)
     
  --block
  
     --low 
     --medium   
     --high
     --not applicable
 --potholes
 
    --low
    --medium
    --high
    --not applicable
    
 The Pavement Condition Index (PCI) is a numerical index between 0 and 100, which is used to indicate the general condition of a pavement section.
 PCR = 100-sum(deduct of each fault)
 where deduct = distressweight * extent_weight * severity
 
 
 
 Model:
 There are 3 different classifiers each for alligator cracking, block cracking and potholes.
 In each of the classifiers, we import the dataset of the perticular fault and pass it through a number of neural network layers which will be able to classify the image as 
 low, medium, high severity of the fault or the fault is not present at all.
 For new image predictions we pass the image through ll the classifiers as there may be multiple faults present in a single image.
