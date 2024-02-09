# sobel
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister
from os import lstat
from re import L
from sys import displayhook
from qiskit import *
from qiskit import IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
from qiskit_aer import AerSimulator
style.use('bmh')
import cv2
from qiskit.tools.jupyter import *
from PIL import Image
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit.library.arithmetic.adders import CDKMRippleCarryAdder
image_size = 8 # Original image-width
#image_crop_size = 4  # Width of each part of image for processing
# Load the image from filesystem
#image = np.array(Image.open(r"C:\Users\User\Desktop\New folder (2)\Neqr2.jpg").convert('L')) 
image = np.array([[0,0,0,0],
                  [0,1,1,0],
                  [0,1,1,0],
                  [0,0,0,0]
                  ])                   
def plot_image(img, title: str):
    plt.title(title)
    plt.xticks(range(0,img.shape[0]))
    plt.yticks(range(0,img.shape[1]))
    plt.imshow (img,extent=[0,img.shape[1],0,img.shape[0]] ,cmap='binary')
    plt.show()
#plot_image(image, 'landscape image')
# Initialize the quantum circuit for the image 
# Pixel position
anc=QuantumRegister(14,'a')
zero=QuantumRegister(35,'zero')
idx=QuantumRegister(4,'idx')
intensity=QuantumRegister(8,'intensity')
idx1=QuantumRegister(4,'u')
intensity1=QuantumRegister(8,'intensity1')
idx2=QuantumRegister(4,'d')
intensity2=QuantumRegister(8,'intensity2')
idx3=QuantumRegister(4,'r')
intensity3=QuantumRegister(8,'intensity3')
idx4=QuantumRegister(4,'l')
intensity4=QuantumRegister(8,'intensity4')
idx5=QuantumRegister(4,'ur')
intensity5=QuantumRegister(8,'intensity5')
idx6=QuantumRegister(4,'ul')
intensity6=QuantumRegister(8,'intensity6')
idx7=QuantumRegister(4,'dr')
intensity7=QuantumRegister(8,'intensity7')
idx8=QuantumRegister(4,'dl')
intensity8=QuantumRegister(8,'intensity8')
cr=ClassicalRegister(9)
# grayscale pixel intensity value
# classical register
#cr = ClassicalRegister(12, 'cr')
# create the quantum circuit for the image
qc = QuantumCircuit(anc,intensity,idx,intensity1,idx1,intensity2,idx2,intensity3,idx3,intensity4,idx4,intensity5,idx5,intensity6,idx6,intensity7,idx7,intensity8,idx8,zero,cr)
qc.i([0,1,2,3,4,5,6,7])
qc.h([8,9,10,11])
qc.i([12,13,14,15,16,17,18,19])
qc.h([20,21,22,23])
qc.i([24,25,26,27,28,29,30,31])
qc.h([32,33,34,35])
qc.i([36,37,38,39,40,41,42,43])
qc.h([44,45,46,47])
qc.i([48,49,50,51,52,53,54,55])
qc.h([56,57,58,59])
qc.i([60,61,62,63,64,65,66,67])
qc.h([68,69,70,71])
qc.i([72,73,74,75,76,77,78,79])
qc.h([80,81,82,83])
qc.i([84,85,86,87,88,89,90,91])
qc.h([92,93,94,95])
qc.i([96,97,98,99,100,101,102,103])
qc.h([104,105,106,107])
qc.barrier()
number=0
pixel_list=[]
color_list=[]
for i in range(0,4):
    for j in range(0,4):
        number=1+number
        pixel=format(number-1,'b')
        element=len(pixel)
        zero=4-element
        #print(pixel)
        color=format(image[i][j],'b')
        num0=7
        name=""
        if color=='0':
         while 0<num0<8:
          name1="0"
          name+=name1
          num0=num0-1
          if num0==0:
            break
        else:
         while 0<num0<8:
            name1="1"
            name+=name1
            num0=num0-1
            if num0==0:
              break
        name+=color
        pixel_list.append(pixel)
        color_list.append(name)
        for counters,c in enumerate(name[::-1]):
             if (c=='1'):
               for counter,v in enumerate(pixel[::-1]):
                 if (v=='0'):
                   qc.x(counter+8)
                   qc.x(counter+20)
                   qc.x(counter+32)
                   qc.x(counter+44)
                   qc.x(counter+56)
                   qc.x(counter+68)
                   qc.x(counter+80)
                   qc.x(counter+92)
                   qc.x(counter+104)
               if(zero!=0):
                    for m in range(element,4):
                      qc.x(m+8) 
                      qc.x(m+20)
                      qc.x(m+32)
                      qc.x(m+44)
                      qc.x(m+56)
                      qc.x(m+68)
                      qc.x(m+80)
                      qc.x(m+92)
                      qc.x(m+104)
               qc.mcx([8,9,10,11],counters)
               qc.mcx([20,21,22,23],counters+12)
               qc.mcx([32,33,34,35],counters+24)
               qc.mcx([44,45,46,47],counters+36)
               qc.mcx([56,57,58,59],counters+48)
               qc.mcx([68,69,70,71],counters+60)
               qc.mcx([80,81,82,83],counters+72)
               qc.mcx([92,93,94,95],counters+84)
               qc.mcx([104,105,106,107],counters+96)
               for counter,v in enumerate(pixel[::-1]):
                if (v=='0'):
                   qc.x(counter+8)
                   qc.x(counter+20)
                   qc.x(counter+32)
                   qc.x(counter+44)
                   qc.x(counter+56)
                   qc.x(counter+68)
                   qc.x(counter+80)
                   qc.x(counter+92)
                   qc.x(counter+104)
               if(zero!=0):
                    for m in range(element,4):
                      qc.x(m+8)
                      qc.x(m+20)
                      qc.x(m+32)
                      qc.x(m+44)
                      qc.x(m+56)
                      qc.x(m+68)
                      qc.x(m+80)
                      qc.x(m+92)
                      qc.x(m+104)
        qc.barrier() 
#up
qc.x(23)
qc.cx(23,22)
#down
qc.cx(35,34)
qc.x(35)
#right
qc.x(45)
qc.cx(45,44)
#left
qc.cx(57,56)
qc.x(57)
#upright
qc.x(71)
qc.cx(71,70)
qc.x(69)
qc.cx(69,68)
#upleft
qc.x(83)
qc.cx(83,82)
qc.cx(81,80)
qc.x(81)
#downright
qc.cx(95,94)
qc.x(95)
qc.x(93)
qc.cx(93,92)
#downleft
qc.cx(107,106)
qc.x(107)
qc.cx(105,104)
qc.x(105)
adder=CDKMRippleCarryAdder(8,'full',"Full")
adder1=CDKMRippleCarryAdder(9,'full',"Full1")
adder2=CDKMRippleCarryAdder(10,'full',"Full2")
qc.barrier()

qc1=qc.compose(adder,[anc[0]]+intensity2[0:8]+intensity7[0:8]+[anc[1]])
qc1.barrier()
qc2=qc1.compose(adder,[anc[2]]+intensity3[0:8]+intensity5[0:8]+[anc[3]])
qc2.barrier()
qc3=qc2.compose(adder,[anc[4]]+intensity1[0:8]+intensity6[0:8]+[anc[5]])
qc3.barrier()
qc4=qc3.compose(adder,[anc[6]]+intensity4[0:8]+intensity8[0:8]+[anc[7]])
qc4.barrier()
qc5=qc4.compose(adder1,[anc[8]]+[anc[1]]+intensity7[0:8]+[anc[3]]+intensity5[0:8]+[anc[9]])
qc5.barrier()
qc6=qc5.compose(adder1,[anc[10]]+[anc[5]]+intensity6[0:8]+[anc[7]]+intensity8[0:8]+[anc[11]])
qc6.barrier()
qc7=qc6.compose(adder2,[anc[12]]+[anc[9]]+[anc[3]]+intensity5[0:8]+[anc[11]]+[anc[7]]+intensity8[0:8]+[anc[13]])
qc7.barrier()
#subtract ([anc[13]]+[anc[11]]+[anc[7]])+intensity8[0:8])-(zero+intensity)
#rps
qc7.csx(13,23)
qc7.cx(13,24)
qc7.cx(15,13)
qc7.csx(15,23)
qc7.csx(13,23)
qc7.barrier()
#rfs1
qc7.csx(11,25)
qc7.cx(11,26)
qc7.cx(16,11)
qc7.csx(16,25)
qc7.cx()
