# Copyright (c) 2019 American Express Travel Related Services Company, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

from keras.models import Sequential, Model
from keras.layers import Dense, Embedding,LSTM, Input, Activation, multiply
from keras.preprocessing import sequence

# User creates Neural Network Architecture for training Model.
##### Below codes is sample of Neural Network Architecture #####
# Parameters

max_features = 4000 # 
maxlen = 400 # cut sequences after this number of token 

input1 = Input(shape=(maxlen,))
X1 = Embedding(max_features, 256, dropout=0.1,mask_zero=True)(input1)
X1 = LSTM(64, dropout_W=0.2, dropout_U=0.2)(X1)
X = Dense(32)(X1)
X = Dense(1)(X)
y = Activation('sigmoid')(X)  


rnn = Model(input=[input1], output=y)
rnn.compile(loss='binary_crossentropy',optimizer = 'adam') 

#  END of sample of Neural Network Architecture #

# Below code starts training model on top of TensorFlow Distributed Deep Learning framework #
from distExec import ExecutionEnvironment
env=ExecutionEnvironment(bucket = 'easydist.data',prefix = 'rnnData/', epochs = 1, batch_size = 32, opt= 'adam')

env.fit()  

# End of training model on top of TensorFlow Distributed Deep Learning framework #