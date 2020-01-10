#----------------Face Recognition using Eigenfaces-------------------------
#Dataset Used: AT & T Dataset | Olivetti Research Laboratory
#Algorithm developed by Vision and Modelling group
#---------------------------------------------------------------------------
#importing libraries
from matplotlib import pyplot as plt
from scipy import linalg as LA #for finding eigen values and eigen vectors
import numpy as np
import time #for calculating the running time
import glob
import cv2

#to decipher the processing time
start_time=time.time();


#Converting an array (N x N) to vector (N^2 x 1)
def array_to_vector(Array):
    rows=len(Array)*len(Array[0]);
    vector=[];
    for i in range(len(Array)):
        for j in range(len(Array[0])):
            vector.append(Array[i][j])
    return(vector)

#Converting a vector (N^2 x 1) to array (N x N)
def vector_to_array(vector,row,column):
    counter=0;
    Array=np.zeros((row,column)) #not taking complex values!
    for i in range(row):
        for j in range(column):
            Array[i][j]=vector[counter];    
            counter+=1;        
    return Array


#70% training dataset
training_path='C:\\Shreya\\SEM-5\\PRP\\train\\s';
training_images=[];

for i in range(1,41):
    folder_inside=training_path+str(i)+'\\*.*';
    for file in glob.glob(folder_inside):
        #reading the .pgm image files
        Img=cv2.imread(file,cv2.IMREAD_GRAYSCALE) #3 channel input image is converted into grayscale
        training_images.append(Img)

row=len(training_images[0]); #number of rows in each input image array
column=len(training_images[0][0]); #number of columns in each input image array
#print(len(A[0]))
#print(len(A[0][0]))

Avg_Img=np.zeros((len(training_images[0]),len(training_images[0][0]))); #psi=>Avg_Img

for i in range(len(training_images)):
    Avg_Img=Avg_Img+training_images[i];

for i in range(len(Avg_Img)):
    for j in range(len(Avg_Img[0])):
        Avg_Img[i][j]=int(round((Avg_Img[i][j]/len(training_images))))

fig=plt.figure()
plt.imshow(Avg_Img, cmap='gray')
plt.show()

#(fi) list of arrays after subtracting them from the average matrix (Avg_Img)
array_fi=[];

for i in range(len(training_images)):
    fi=np.zeros((row,column));
    fi=(training_images[i]-Avg_Img);
    array_fi.append(fi)

#converting those arrays into vectors
vector_fi=[]; 
for array in array_fi:
    vector=array_to_vector(array);
    vector_fi.append(vector)

A=np.array(vector_fi)
A_T=A.T #Transpose of A

#Covariance Matrix: 280 x 280
C=np.dot(A,A_T)
#print('Covariance Matrix',C)
#print(len(C),len(C[0]))

#Determining eigen values and eigen vectors from a covariance matrix
eigen_values,eigen_vectors=LA.eig(C)

#print(len(eigen_vectors))
#print(len(eigen_vectors[0]))
#print(eigen_vectors)

#Now, sorting the eigen vectors in decreasing order of the eigen values
dict_val_vec={}; #creating a dictionary where keys are the eigen values and values are the eigen vectors
for i in range(len(eigen_values)):
    if (eigen_values[i]>100): #assumed a threshold which is passing 280 eigen values
        dict_val_vec.update({eigen_values[i]:eigen_vectors[:,i]})

sorted_list=sorted(dict_val_vec,reverse=True);
sorted_dict={};
mean_val=0;
k=20; #selecting 20 eigen faces as basis vectors
k_value=0;
for i in sorted_list:
    u=np.matmul(A_T,dict_val_vec[i]) #u is a numpy array
    norm_u=np.linalg.norm(u)
    #print(norm_u)
    u=u/norm_u
    #print(u)
    sorted_dict.update({i:u})
    mean_val=mean_val+i;
    k_value+=1
    if k_value==k:
        break

mean_val=mean_val/280;
#print(mean_val)

#copy_sorted_dict=sorted_dict; #copy stored for future use

#print(len(sorted_dict))

#for i in sorted_list:
#    if (i<mean_val):
#        del sorted_dict[i]

#print(len(sorted_dict)) #giving 37 eigen values and eigen vectors

#to retrieve key/value pairs 
random_list=list(sorted_dict.items()) 

eigfaces_vectors=[]; #list of eigen faces in vector form
eigfaces_arrays=[]; #list of eigen faces in array form

for i in range(len(random_list)):
    s=list(random_list[i][1])
    eigfaces_vectors.append(s)
    t=vector_to_array(s,row,column)
    eigfaces_arrays.append(t)

#print(eigfaces_vectors)

#Plot eigen faces (for 1st 20 eigen faces)
fig=plt.figure()
fig.subplots_adjust(hspace=0.1,wspace=0.1)
for i in range(1,21):
    ax=fig.add_subplot(5,4,i)
    ax.imshow(eigfaces_arrays[i-1], cmap='gray')
plt.show()

#Making face classes
eigfaces_vectors=np.array(eigfaces_vectors)
#Matrix Multiplication
face_classes_0=np.matmul(eigfaces_vectors,A_T)
#print(len(eigfaces_vectors),len(eigfaces_vectors[0]))
#print(len(A_T),len(A_T[0]))
face_classes=face_classes_0.T

#For all the test images
testing_path='C:\\Shreya\\SEM-5\\PRP\\test\\s'; #all are known
testing_images=[];
for i in range(1,41):
    folder_inside=testing_path+str(i)+'\\*.*';
    for file in glob.glob(folder_inside):
       #reading the .pgm files
        Img=cv2.imread(file,cv2.IMREAD_GRAYSCALE) #3 channel input image is converted into grayscale
        testing_images.append(Img)

counter_value=0;
correct=0;
incorrect=0;

for Img in testing_images:
    diff_Img=Img-Avg_Img;
    diff_Img_vec=array_to_vector(diff_Img);
    diff_Img_vec=np.array(diff_Img_vec);
    weights=np.matmul(eigfaces_vectors,diff_Img_vec)
    weights=weights.T
    epsilon=[];
    for i in range(len(face_classes)):
        epsilon.append(LA.norm(weights-face_classes[i]))
    #print(len(epsilon))
    minimum=epsilon[0]
    person=1;
    for i in range(len(epsilon)):
        if epsilon[i]<minimum:
            minimum=epsilon[i];
            person=i;
    #print(minimum,person)

    if (int(person/7)+1)!=(int(counter_value/3)+1):
        incorrect+=1;
    else:
        correct+=1;

    counter_value+=1;

    print(minimum,int(person/7)+1,'th person')

print(k, 'eigenfaces chosen')
print('Correct Prediction:',correct)
print('Incorrect Prediction:',incorrect)
print('Accurcacy:', float((correct/120)*100))
print('Processing Time:', '%s seconds'%(time.time()-start_time))

#For 1 test image
#test_image='C:\\Shreya\\SEM-5\\PRP\\test\\s36\\5.pgm';
#Image=cv2.imread(test_image,cv2.IMREAD_GRAYSCALE);

#diff_Img=Image-Avg_Img;
#diff_Img_vec=array_to_vector(diff_Img);
#diff_Img_vec=np.array(diff_Img_vec);
#weights=np.matmul(eigfaces_vectors,diff_Img_vec)
#print(weights)
#weights=weights.T
#epsilon=[];

#for i in range(len(face_classes)):
#    epsilon.append(LA.norm(weights-face_classes[i]))

#print(epsilon)

#minimum=epsilon[0]
#a=0;
#for i in range(len(epsilon)):
#    if epsilon[i]<minimum:
#        minimum=epsilon[i];
#        a=i+13;
#print(minimum,a)
#print(int(a/7)+1,'th person')
