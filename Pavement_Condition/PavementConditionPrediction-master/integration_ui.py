from keras.utils import plot_model
from keras.models import load_model
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
from keras.preprocessing import image
import tkinter as tk

#load the models
model_a=load_model("alligater_classifier.h5")
model_b=load_model("block_classsifier.h5")
model_p=load_model("pothole_classsifier.h5")

#load any image form the local file system
filename=askopenfilename()

#resize the image as reqired by the classsifier
myimage=cv2.imread(filename,0)
myimage=cv2.resize(myimage,(128,128))
m=image.img_to_array(myimage)
m=np.expand_dims(m,axis=0)
images=np.vstack([m])

#check whether the image has alligator cracking, if yes assign severity based on the extent of alligator cracking present
y_pred=model_a.predict(images)
print(y_pred)
classes=model_a.predict_classes(images)
print(classes)
a=classes[0]

distressweight_a=5
extent_weight_a=0.7

if a==0:
    severity_a=0.4
    alli="alligator cracking is low"
elif a==1:
    severity_a=0.7
    alli="alligator cracking is medium"
elif a==2:
    severity_a=1
    alli="alligator cracking is high"
elif a==3:
    severity_a=0
    alli="alligator cracking is not present"

deduct_a=distressweight_a*extent_weight_a*severity_a
print(deduct_a)


#check whether the image has potholes, if yes assign severity based on the extent of potholes present
y_pred=model_p.predict(images)
#print(y_pred)
classes=model_p.predict_classes(images)
#print(classes)
p=classes[0]

distressweight_p=10
extent_weight_p=0.8

if p==0:
    severity_p=0
    potna="pothole is not present"
elif p==1:
    severity_p=1
    potna="pothole is high"
elif p==2:
    severity_p=0.4
    potna="pothole is low"
elif p==3:
    severity_p=0.7
    potna="pothole is not medium"

deduct_p=distressweight_p*extent_weight_p*severity_p
#print(deduct_p)

#check whether the image has block cracking, if yes assign severity based on the extent of block cracking present
y_pred=model_b.predict(images)
#print(y_pred)
classes=model_b.predict_classes(images)
#print(classes)
b=classes[0]

distressweight_b=10
extent_weight_b=0.7

if b==0:
    s_b=1
    bl="block cracking is high"
elif b==1:
    s_b=.4
    bl="block cracking is low"
elif b==2:
    s_b=0.7
    bl="block cracking is medium"
elif b==3:
    s_b=0
    bl="block cracking is not present"

deduct_b=distressweight_b*extent_weight_b*s_b
#print(deduct_b)

#calculate the overall pavement condition rate
pcr=100-(deduct_a+deduct_b+deduct_p)

if(pcr>=0 and pcr<10):
    var1="failed road condition\n"
elif(pcr>=10 and pcr<40):
    var1="poor road condition"
elif(pcr>=40 and pcr<70):
    var1="good road condition"
elif(pcr>=70 and pcr<85):
    var1="very good road condition"
else:
    var1="excellent road condition"

#UI
root=tk.Tk()
T=tk.Text(root,height=10,width=50)
T.pack()
T.insert(tk.END,potna+'\n')
T.insert(tk.END,bl+'\n')
T.insert(tk.END,alli+'\n')
T.insert(tk.END,var1+'\n')


tk.mainloop()
