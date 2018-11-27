#IN functional approch, you can try in class too

#import numpy library for omputational math
import numpy as np
#import signal from scipy for convolution or as they call signal processing stuff
from scipy import signal

#lets define some helpful function
#sigmoid Function 
def sigm(x):
  return 1/(1+ np.exp(-1*x))
# hyperbolic as activation function 
def tanh(x):
  return np.tanh(x)

#For back propagation
def d_sigm(x):
  return (sigm(x) * (1-sigm(x)) )

def d_tanh(x):
  return (1-np.tanh(x)**2)

#lets define input stuff..you can choose any value you want
#four input samples
#note we have 9 node in input layer
x1 = np.array([[0,0,0], [0,0,0], [0,0,0]])
x2 = np.array([[0,0,0], [1,1,1], [0,0,0]])
x3 = np.array([[1,1,1], [0,0,0], [1,1,1]])
x4 = np.array([[1,1,1], [1,1,1], [1,1,1]])

#Lets combine input in one variable for simplicity
x = [x1, x2, x3, x4]

#lets initialize weight for next layer
#note we have 4 node in layer 1
y = np.array([[0.53], [0.77], [0.88], [1.1]])

#lets now initialize random weight
# multiple of 4 is just for fun, you can easily not include 
w1 = np.random.randn(2,2)*4
w2 = np.random.randn(4,1)*4

#initializing hyper params
epoch = 855 #number of times you want
learning_rate = 0.7 #global practise of keeping between 0 and 1



#initializing cost to zero..value refers to the output of cost function or one can call it loss function/error function 
totalCostBefore = 0
totalCostAfter = 0

#declaring array for future use
final_out = np.array([[]])
start_out = np.array([[]])

#cost before training
for i in range(len(x)):
  layer1 = signal.convolve2d(x[i], w1, 'valid')
  layer1_act = tanh(layer1)
  layer1_act_vec = np.expand_dims(np.reshape(layer1_act, -1), axis=0)

  layer2 = layer1_act_vec.dot(w2)
  layer2_act = sigm(layer2)

  cost = np.square(layer2_act - y[i]).sum()*0.5
  totalCostBefore += cost
  start_out = np.append(start_out, layer2_act)



#for Training

for ite in range(epoch):
  for i in range(len(x)):
    layer1 = signal.convolve2d(x[i],w1,'valid')
    layer1_act = tanh(layer1)
    layer1_act_vec = np.expand_dims(np.reshape(layer1_act, -1), axis=0)
    
    layer2 = layer1_act_vec.dot(w2)
    layer2_act = sigm(layer2)

    cost = np.square(layer2_act - y[i]).sum()*0.5

    grad2_part1 = layer2_act -y[i]
    grad2_part2 = d_sigm(layer2)
    grad2_part3 = layer1_act_vec
    grad2 = grad2_part3.T.dot(grad2_part1*grad2_part2)

    grad1_part1 = (grad2_part1* grad2_part2).dot(w2.T)
    grad1_part2 = d_tanh(layer1)
    grad1_part3 =x[i]

    grad1_part1_reshape = np.reshape(grad1_part1, (2,2))
    grad1_temp1 = grad1_part1_reshape * grad1_part2
    grad1 = np.rot90(signal.convolve2d(grad1_part3, np.rot90(grad1_temp1, 2), 'valid'),2)
    w2 = w2 - grad2*learning_rate
    w1 = w1-grad1*learning_rate


#cost after training
for i in range(len(x)):
  layer1 = signal.convolve2d(x[i],w1,'valid')
  layer1_act = tanh(layer1)
  layer1_act_vec = np.expand_dims(np.reshape(layer1_act, -1), axis=0)
  layer2 = layer1_act_vec.dot(w2)
  layer2_act = sigm(layer2)

  cost = np.square(layer2_act - y[i]).sum()*0.5
  totalCostAfter += cost
  final_out = np.append(final_out, layer2_act)


#Lets print result
print('\n W1 ',w1, ' w2',w2)
print('\n')
print('training cost before', totalCostBefore)
print('training cost after', totalCostAfter)

