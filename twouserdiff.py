from keras.layers import Input, Dense, GaussianNoise, Lambda, Reshape,BatchNormalization,Add
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback
import numpy as np
#np.set_printoptions(threshold='np.nan')
#import sys
import random as rd
import matplotlib.pyplot as plt
from keras.optimizers import Adam,SGD
from sklearn.model_selection  import KFold
from tensorflow.python import pywrap_tensorflow
'''from numpy.random import seed
seed(2)
from tensorflow import set_random_seed
set_random_seed(3)
'''



k1 = 2
k2 = 4
n = 2
M1 = 2**k1
M2 = 2**k2
R1 = k1/n
R2 = k2/n
h1 = 10**6
h1_1 =250000
h = 1000
h_1 = 250
e = 0
alpha = K.variable(0.5)
beta = K.variable(0.5)
eye_matrix1 = np.eye(M1)
wer1 = eye_matrix1

x_train = np.tile(wer1, (h, 1))
x_train1 = x_train
x_train2 = x_train
x_train3 = x_train
x_train4 = x_train
np.random.shuffle(x_train1)
np.random.shuffle(x_train2)
np.random.shuffle(x_train3)
np.random.shuffle(x_train4)
x_test =  np.tile(wer1, (h1, 1))
x_test1 = x_test
x_test2 = x_test
x_test3 = x_test
x_test4 = x_test
np.random.shuffle(x_test1)
np.random.shuffle(x_test2)
np.random.shuffle(x_test3)
np.random.shuffle(x_test4)
#x_try = np.tile(eye_matrix, (1000, 1))
#rd.shuffle(x_train)
#rd.shuffle(x_test)
#rd.shuffle(x_try)
eye_matrix2 = np.eye(M2)
wer2 = eye_matrix2

x_train5 = np.tile(wer2, (h_1, 1))
x_train5 = x_train5
x_train6 = x_train5
x_train7 = x_train5
x_train8 = x_train5
np.random.shuffle(x_train5)
np.random.shuffle(x_train6)
np.random.shuffle(x_train7)
np.random.shuffle(x_train8)
x_test5 =  np.tile(wer2, (h1_1, 1))
x_test5 = x_test5
x_test6 = x_test5
x_test7 = x_test5
x_test8 = x_test5
np.random.shuffle(x_test5)
np.random.shuffle(x_test6)
np.random.shuffle(x_test7)
np.random.shuffle(x_test8)
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
    return K.mean(K.square(y_pred-y_true),axis=-1) + 1.0*(K.sum(K.abs(y_pred),axis=-1)**(1/2))

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
Eb_No_dB = e
noise = 1/(10**(Eb_No_dB/10))
#noise_sigma = np.sqrt(noise)
belta1 = 1/(2*R1*(10**(Eb_No_dB/10)))
belta_sqrt1 = np.sqrt(belta1)
belta2 = 1/(2*R2*(10**(Eb_No_dB/10)))
belta_sqrt2 = np.sqrt(belta2)
# noise
def mixed_AWGN1(x):
    signal = x[0]
    interference = x[1]
    noise = K.random_normal(K.shape(signal),
                            mean=0,
                            stddev=belta_sqrt1)
    #signal = Add()([signal, interference])
    #signal = Add()([signal, noise])
    signal = signal + interference + noise
    return signal

def mixed_AWGN2(x):
    signal = x[0]
    interference = x[1]
    noise = K.random_normal(K.shape(signal),
                            mean=0,
                            stddev=belta_sqrt2)
    #signal = Add()([signal, interference])
    #signal = Add()([signal, noise])
    signal = signal + interference + noise
    return signal

# dynamic loss weights
class Mycallback(Callback):
    def __init__(self,alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.epoch_num = 0
    def on_epoch_end(self, epoch, logs={}):
        self.epoch_num = self.epoch_num + 1
        loss1 = logs.get('u1_receiver2_loss')
        loss2 = logs.get('u2_receiver2_loss')
        print("epoch %d" %self.epoch_num)
        print("total_loss%f" %logs.get('loss'))
        print("u1_loss %f"%(loss1))
        print("u2_loss %f" % (loss2))
        a = loss1 / (loss1 + 4*loss2)
        b = 4*loss2 / (loss1 + 4*loss2)

        K.set_value(self.alpha, a)
        K.set_value(self.beta, b)
        print("alpha %f" %K.get_value(alpha))
        print("beta %f" % K.get_value(beta))
        print("selfalpha %f" % K.get_value(self.alpha))
        print("selfbeta %f" % K.get_value(self.beta))



#autoencoder
#user 1
# Transmitter
input_sys_1 = Input(shape=(M1,))
#print(input_sys)
encoded_1 = Dense(M1, activation='relu',bias=True)(input_sys_1)
#encoded_1 = Dense(M, activation='relu')(encoded)
encoded1_1 = Dense(n, activation= 'linear',bias=True)(encoded_1)
#encoded2 = BatchNormalization()(encoded1)
#encoded2_1 = Lambda(lambda x: np.sqrt(n) * K.l2_normalize(x, axis=1))(encoded1_1)
encoded2_1 = Lambda(lambda x: x / K.sqrt(K.mean(K.abs(x)**2)))(encoded1_1)



#user 2
# Transmitter
input_sys_2 = Input(shape=(M2,))
#print(input_sys)
encoded_2 = Dense(M2, activation='relu',bias=True)(input_sys_2)
#encoded_1 = Dense(M, activation='relu')(encoded)
encoded1_2 = Dense(n, activation= 'linear',bias=True)(encoded_2)
#encoded2 = BatchNormalization()(encoded1)
#encoded2_2 = Lambda(lambda x: np.sqrt(n) * K.l2_normalize(x, axis=1))(encoded1_2)
encoded2_2 = Lambda(lambda x: x / K.sqrt(K.mean(K.abs(x)**2)))(encoded1_2)

#mixed AWGN channel
encoded_noise_1 = Lambda(lambda x: mixed_AWGN1(x))([ encoded2_1, encoded2_2])
encoded_noise_2 = Lambda(lambda x: mixed_AWGN2(x))([ encoded2_2, encoded2_1])

# Receiver
#user1
decoded1_1 = Dense(M1, activation='relu', name= 'u1_receiver1')(encoded_noise_1)

decoded2_1 = Dense(M1, activation='softmax', name= 'u1_receiver2')(decoded1_1)
#user2
decoded1_2 = Dense(M2, activation='relu',name= 'u2_receiver1')(encoded_noise_2)

decoded2_2 = Dense(M2, activation='softmax', name= 'u2_receiver2')(decoded1_2)
#autoencoder = Model(input_sys, [decoded2_1, decoded2_2, decoded2_3])
autoencoder = Model(inputs=[input_sys_1,input_sys_2],outputs =[decoded2_1,decoded2_2])

encoder_1 = Model(inputs=input_sys_1, outputs=encoded2_1)
encoder_2 = Model(inputs=input_sys_2, outputs=encoded2_2)
#aaa = Model(inputs=input_sys, outputs=decoded1)
#aaa = Model(inputs=input_sys, outputs=decoded1)
#decoder = Model(inputs=encoded_noise, outputs=decoded2)
encoded_input_1 = Input(shape=(n,))
deco_1 = autoencoder.get_layer("u1_receiver1")(encoded_input_1)
deco_11 = autoencoder.get_layer("u1_receiver2")(deco_1)
decoder_1 = Model(encoded_input_1, deco_11)

encoded_input_2 = Input(shape=(n,))
deco_2 = autoencoder.get_layer("u2_receiver1")(encoded_input_2)
deco_22 = autoencoder.get_layer("u2_receiver2")(deco_2)
decoder_2 = Model(encoded_input_2, deco_22)
#get_output = K.function([autoencoder.layers[0].input],[autoencoder.layers[3].output])


adam = Adam(lr=0.1)

#print(encoder.layers[3].output)

#autoencoder.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['binary_accuracy'])
#autoencoder.compile(optimizer='adam', loss='mse',metrics=['binary_accuracy'])
autoencoder.compile(optimizer='adam', loss=mean_squared_error2,loss_weights=[alpha,beta])
#autoencoder.compile(optimizer='adam', loss='categorical_crossentropy',loss_weights=[alpha,beta])
#checkpointer = ModelCheckpoint(filepath="/Users/sunyilin/Desktops/weights.hdf5")
#hist = autoencoder.fit(eye_matrix, eye_matrix, batch_size=16, epochs=1000, validation_data=(eye_matrix, eye_matrix))
#hist = autoencoder.fit(x_train, [x_train, x_train, x_train], batch_size=h, epochs=500, validation_data=(x_test, [x_test,x_test,x_test]))
hist = autoencoder.fit([x_train1, x_train5], [x_train1, x_train5],batch_size=32, epochs=50,
                       callbacks=[Mycallback(alpha,beta)])
#encoded_sys = encoder.predict(x_try)
#decoded_sys = autoencoder.predict(x_try)
#decoded_sys_round = np.round(decoded_sys)
#error_rate = np.mean(np.not_equal(x_try,decoded_sys_round).max(axis=1))

#part_model = Model(inputs = autoencoder.input,
                   #outputs = autoencoder.get_layer("mylayer").output)
#part_output = part_model.predict(x_train)
#print(part_output)


#model_reader = pywrap_tensorflow.NewCheckpointReader(r"model.ckpt")
#Constellation
encoded_planisphere_1 = encoder_1.predict(wer1)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.plot(encoded_planisphere_1[:,0], encoded_planisphere_1[:,1], 'r.')
plt.grid(True)
plt.show()
print(encoded_planisphere_1)

encoded_planisphere_2 = encoder_2.predict(wer2)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.plot(encoded_planisphere_2[:,0], encoded_planisphere_2[:,1], 'r.')
plt.grid(True)
plt.show()
print(encoded_planisphere_2)

#z = aaa.predict(wer)
#print(z)
#output=sys.stdout
#outputfile=open("/Users/sunyilin/Desktop/a.rtf","a")
#sys.stdout=outputfile
#part_model = Model(inputs = autoencoder.input,
                   #outputs = autoencoder.get_layer("mylayer").output)
#part_output = part_model.predict(eye_matrix)
#print(part_output)

plt.figure()
plt.plot(hist.history['loss'])
plt.show()
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')



#var_dict = model_reader.get_variable_to_shape_map()


#for key in var_dict:
 #   print("variable name: ", key)
  #  print(model_reader.get_tensor(key))


'''for Eb_No_dB1 in np.arange(0, 10, 1):
    belta1 = 1/(2*R*(10**(Eb_No_dB1/10)))
    belta_sqrt1 = np.sqrt(belta1)
def frange(x, y, jump):
  while x < y:
    yield x
    x += jump


EbNodB_range = list(frange(0,10,1))
ber1 = [None]*len(EbNodB_range)
ber2 = [None]*len(EbNodB_range)
for N in range(0,len(EbNodB_range)):
    EbNo=10.0**(EbNodB_range[N]/10.0)
    noise_std = np.sqrt(1/(2*R*EbNo))
    noise_mean = 0
    no_errors = 0
    nn = M*h1
    noise_1 = noise_std * np.random.randn(nn, n)
    noise_2 = noise_std * np.random.randn(nn, n)
    encoded_signal_1 = encoder_1.predict(x_test1)
    encoded_signal_2 = encoder_2.predict(x_test2)
    final_signal_1 = encoded_signal_1 + encoded_signal_2 + noise_1
    final_signal_2 = encoded_signal_1 + encoded_signal_2 + noise_2

    pred_final_signal_1 = decoder_1.predict(final_signal_1)
    pred_final_signal_2 = decoder_2.predict(final_signal_2)

    pred_output_1 = np.argmax(pred_final_signal_1, axis=1)
    # a2 = np.array([[0],1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    # print(a2.shape)
    pred_output_2 = np.argmax(pred_final_signal_2, axis=1)

    # a = np.array(range(M)).reshape((-1,1))
    # print(a.shape)
    # a1 = np.tile(a, (h1, 1))
    # print(a1.shape)
    target1 = np.argmax(x_test1, axis=1)
    target2 = np.argmax(x_test2, axis=1)

    pred_output_1 = np.reshape(pred_output_1, (M * h1, 1))
    pred_output_2 = np.reshape(pred_output_2, (M * h1, 1))

    target1 = np.reshape(target1, (M * h1, 1))
    target2 = np.reshape(target2, (M * h1, 1))

    no_errors_1 = (pred_output_1 != target1)
    no_errors_2 = (pred_output_2 != target2)

    # print(no_errors)
    # print(pred_output.shape)
    # print(no_errors.shape)

    no_errors_1 = no_errors_1.astype(int).sum()
    no_errors_2 = no_errors_2.astype(int).sum()

    # no_errors = 1*no_errors
    # print(no_errors)
    # no_errors = no_errors_1 + no_errors_2 +no_errors_3 + no_errors_4



    ber1[N] = no_errors_1 / nn
    print('SNR:', EbNodB_range[N], 'BER:', ber1[N])
    ber2[N] = no_errors_2 / nn
    print('SNR:', EbNodB_range[N], 'BER:', ber2[N])
print(ber1)
print(ber2)

from scipy import interpolate
ig, ax = plt.subplots()
ax.plot(EbNodB_range, ber1,label='User1')
ax.plot(EbNodB_range, ber2,label='User2')
plt.yscale('log')
plt.xlabel('SNR Range')
plt.ylabel('Block Error Rate')
plt.grid()
plt.legend(loc='upper right',ncol = 1)
plt.show()






'''
EbNodB = e
EbNo=10.0**(EbNodB/10.0)
noise_std1 = np.sqrt(1/(2*R1*EbNo))
noise_std2 = np.sqrt(1/(2*R2*EbNo))
noise_mean = 0
no_errors = 0
nn1 = M1*h1
nn2 = M2*h1_1
noise_1 = noise_std1 * np.random.randn(nn1, n)
noise_2 = noise_std2 * np.random.randn(nn2, n)


encoded_signal_1 = encoder_1.predict(x_test1)
encoded_signal_2 = encoder_2.predict(x_test5)


final_signal_1 = encoded_signal_1 + encoded_signal_2 + noise_1
final_signal_2 = encoded_signal_1 + encoded_signal_2 + noise_2


pred_final_signal_1 = decoder_1.predict(final_signal_1)
pred_final_signal_2 = decoder_2.predict(final_signal_2)


pred_output_1 = np.argmax(pred_final_signal_1, axis=1)
#a2 = np.array([[0],1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
#print(a2.shape)
pred_output_2 = np.argmax(pred_final_signal_2, axis=1)


#a = np.array(range(M)).reshape((-1,1))
#print(a.shape)
#a1 = np.tile(a, (h1, 1))
#print(a1.shape)
target1 = np.argmax(x_test1, axis=1)
target2 = np.argmax(x_test2, axis=1)


pred_output_1 = np.reshape(pred_output_1, (M1 * h1, 1))
pred_output_2 = np.reshape(pred_output_2, (M2* h1_1, 1))


target1 = np.reshape(target1, (M1 * h1, 1))
target2 = np.reshape(target2, (M2 * h1_1, 1))


no_errors_1 = (pred_output_1 != target1)
no_errors_2 = (pred_output_2 != target2)


#print(no_errors)
#print(pred_output.shape)
#print(no_errors.shape)

no_errors_1 = no_errors_1.astype(int).sum()
no_errors_2 = no_errors_2.astype(int).sum()


#no_errors = 1*no_errors
#print(no_errors)
#no_errors = no_errors_1 + no_errors_2 +no_errors_3 + no_errors_4

ber1 = no_errors_1 / nn1

ber2 = no_errors_2 / nn2





print(ber1)
print(ber2)


