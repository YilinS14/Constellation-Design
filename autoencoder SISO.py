from keras.layers import Input, Dense, GaussianNoise, Lambda, Reshape,BatchNormalization
from keras.models import Model  
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import numpy as np
#np.set_printoptions(threshold='np.nan')
#import sys
import random as rd
import matplotlib.pyplot as plt
from keras.optimizers import Adam,SGD
from tensorflow.python import pywrap_tensorflow

#initial
k = 2
n = 2
M = 2**k
R = k/n
h1 = 10**5
h = 1000
eye_matrix = np.eye(M)
wer = eye_matrix
x_train = np.tile(wer, (h, 1))#纵向复制
x_train1 =  np.tile(wer, (h1, 1))
x_train2 = x_train
x_test = np.tile(wer, (10, 1))
x_try = np.tile(eye_matrix, (1000, 1))
#rd.shuffle(x_train)
#rd.shuffle(x_test)
rd.shuffle(x_try)
#print(x_train.shape)
#print(x_test.shape)
t1 = np.array([[-1,0],[0,-1]])
T1 = K.variable(t1, dtype='float32')
t2 = np.array([[1,0],[0,-1]])
T2 = K.variable(t2, dtype='float32')
t3 = np.array([[-1,0],[0,1]])
T3 = K.variable(t3, dtype='float32')
#N = K. cast_to_floatx(n)
N = K.variable(value=n, dtype='float32')
#print(wer)
def mean_squared_error2(y_true, y_pred):
    return K.mean(K.square(y_pred-y_true),axis=-1) + 1.5*(K.sum(K.abs(y_pred),axis=-1)**(1/2))

def BER(y_true, y_pred):
    return K.mean(K.not_equal(y_true, K.round(y_pred)), axis=-1)
#def constraints(self):
    #cons = {}
    #for layer in self.layers:
     #   for key, value in layer.constraints.items():
      #      if key in cons:
       #         raise Exception('Received multiple constraints '
        #                        'for one weight tensor: ' + str(key))
         #   cons[key] = value
    #return cons

#SNR
Eb_No_dB = 7
noise = 1/(10**(Eb_No_dB/10))
#noise_sigma = np.sqrt(noise)
belta = 1/(2*R*(10**(Eb_No_dB/10)))
belta_sqrt = np.sqrt(belta)

#autoencoder
input_sys = Input(shape=(M,))
print(input_sys)
encoded = Dense(M, activation='relu',bias=True)(input_sys)
#encoded_1 = Dense(M, activation='relu')(encoded)
encoded1 = Dense(n, activation= 'linear',bias=True)(encoded)
#encoded2 = BatchNormalization()(encoded1)
#encoded2 = Lambda(lambda x: np.sqrt(n) * K.l2_normalize(x, axis=1))(encoded1)
encoded2 = Lambda(lambda x: x / K.sqrt(K.mean(K.abs(x)**2)))(encoded1)
encoded_noise = GaussianNoise(belta_sqrt)(encoded2)#Noise Layer
#print(encoded_noise)
decoded1 = Dense(M, activation='relu',bias=True)(encoded_noise)
'''
y1 = Lambda(lambda x: K.dot(x, T1))(encoded_noise)
y2 = Lambda(lambda x: K.dot(x, T2))(encoded_noise)
y3 = Lambda(lambda x: K.dot(x, T3))(encoded_noise)

decoded1 = Dense(M, activation='relu')
decoded1_1 = decoded1(y1)
decoded1_2 = decoded1(y2)
decoded1_3 = decoded1(y3)
print(decoded1_1)
'''
decoded2 = Dense(M, activation='softmax',bias=True)(decoded1)

'''decoded2 = Dense(M, activation='softmax')

decoded2_1 = decoded2(decoded1_1)
print(decoded2_1)
decoded2_2 = decoded2(decoded1_2)
decoded2_3 = decoded2(decoded1_3)
'''

#autoencoder = Model(input_sys, [decoded2_1, decoded2_2, decoded2_3])
autoencoder = Model(input_sys,decoded2)
encoder = Model(inputs=input_sys, outputs=encoded2)
aaa = Model(inputs=input_sys, outputs=decoded1)
#decoder = Model(inputs=encoded_noise, outputs=decoded2)
encoded_input = Input(shape=(n,))
deco = autoencoder.layers[-2](encoded_input)
deco = autoencoder.layers[-1](deco)
decoder = Model(encoded_input, deco)

#get_output = K.function([autoencoder.layers[0].input],[autoencoder.layers[3].output])


adam = Adam(lr=0.1)

#print(encoder.layers[3].output)

autoencoder.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['binary_accuracy'])
#autoencoder.compile(optimizer='adam', loss='mse',metrics=['binary_accuracy'])
#autoencoder.compile(optimizer='adam', loss=mean_squared_error2,metrics=['binary_accuracy'])
#checkpointer = ModelCheckpoint(filepath="/Users/sunyilin/Desktops/weights.hdf5")
#hist = autoencoder.fit(eye_matrix, eye_matrix, batch_size=16, epochs=1000, validation_data=(eye_matrix, eye_matrix))
#hist = autoencoder.fit(x_train, [x_train, x_train, x_train], batch_size=h, epochs=500, validation_data=(x_test, [x_test,x_test,x_test]))
hist = autoencoder.fit(x_train, x_train, batch_size=32, epochs=100,
                       validation_data=(x_test, x_test))
#encoded_sys = encoder.predict(x_try)
#decoded_sys = autoencoder.predict(x_try)
#decoded_sys_round = np.round(decoded_sys)
#error_rate = np.mean(np.not_equal(x_try,decoded_sys_round).max(axis=1))

#part_model = Model(inputs = autoencoder.input,
                   #outputs = autoencoder.get_layer("mylayer").output)
#part_output = part_model.predict(x_train)
#print(part_output)


#model_reader = pywrap_tensorflow.NewCheckpointReader(r"model.ckpt")


#var_dict = model_reader.get_variable_to_shape_map()


#for key in var_dict:
 #   print("variable name: ", key)
  #  print(model_reader.get_tensor(key))


for Eb_No_dB1 in np.arange(0, 15, 1):
    belta1 = 1/(2*R*(10**(Eb_No_dB1/10)))
    belta_sqrt1 = np.sqrt(belta1)
def frange(x, y, jump):
  while x < y:
    yield x
    x += jump


EbNodB_range = list(frange(0,15,1))
ber = [None]*len(EbNodB_range)
for N in range(0,len(EbNodB_range)):
    EbNo=10.0**(EbNodB_range[N]/10.0)
    noise_std = np.sqrt(1/(2*R*EbNo))
    noise_mean = 0
    no_errors = 0
    nn = M*h1
    noise = noise_std * np.random.randn(nn, n)
    encoded_signal = encoder.predict(x_train1)
    final_signal = encoded_signal + noise
    pred_final_signal = decoder.predict(final_signal)
    pred_output = np.argmax(pred_final_signal, axis=1)
    #a2 = np.array([[0],1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    #print(a2.shape)
    a = np.array(range(M)).reshape((-1,1))
    #print(a.shape)
    a1 = np.tile(a, (h1, 1))
    #print(a1.shape)
    pred_output = np.reshape(pred_output,(M*h1,1))
    no_errors = (pred_output != a1)
    #print(no_errors)
    #print(pred_output.shape)
    #print(no_errors.shape)

    no_errors = no_errors.astype(int).sum()
    #no_errors = 1*no_errors
    #print(no_errors)

    ber[N] = no_errors / nn
    print('SNR:', EbNodB_range[N], 'BER:', ber[N])
print(ber)
from scipy import interpolate
plt.plot(EbNodB_range, ber, 'bo',label='Autoencoder(2,1)')
plt.yscale('log')
plt.xlabel('SNR Range')
plt.ylabel('Block Error Rate')
plt.grid()
plt.legend(loc='upper right',ncol = 1)
plt.show()

'''
#Constellation
encoded_planisphere = encoder.predict(wer)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.plot(encoded_planisphere[:,0], encoded_planisphere[:,1], 'r.')
plt.grid(True)
print(encoded_planisphere)

z = aaa.predict(wer)
print(z)
#output=sys.stdout
#outputfile=open("/Users/sunyilin/Desktop/a.rtf","a")
#sys.stdout=outputfile
#part_model = Model(inputs = autoencoder.input,
                   #outputs = autoencoder.get_layer("mylayer").output)
#part_output = part_model.predict(eye_matrix)
#print(part_output)
'''
plt.figure()
plt.plot(hist.history['loss'])
plt.show()
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

'''
weight1 = autoencoder.layers[1].get_weights()
print(weight1)
weight2 = autoencoder.layers[2].get_weights()
print(weight2)
weight3 = autoencoder.layers[-2].get_weights()
print(weight3)
'''
weight4 = autoencoder.layers[-1].get_weights()
#print(weight4)

'''
from keras.models import load_model
# if you want to save model then remove below comment
autoencoder.save('autoencoder_v_best.model')
load_model()
'''

'''
from tensorflow.python import pywrap_tensorflow

#首先，使用tensorflow自带的python打包库读取模型
model_reader = pywrap_tensorflow.NewCheckpointReader(r"model.ckpt")

#然后，使reader变换成类似于dict形式的数据
var_dict = model_reader.get_variable_to_shape_map()

#最后，循环打印输出
for key in var_dict:
    print("variable name: ", key)
    print(model_reader.get_tensor(key))
'''