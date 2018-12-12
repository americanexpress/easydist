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

''' This module tests the dist_exec module.'''
import sys
import os
from pathlib import Path

from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_raises
from nose.tools import raises
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding,LSTM, Input, Activation, multiply
from keras.preprocessing import sequence
sys.path.append('../src/')
sys.path.append('.')
from dist_exec import ExecutionEnvironment, Execution


class TestExecutionEnvironment(object):
    
    def rnn(self):
        """This method creates a RNN using the Keras API"""
        max_features = 4000 #
        maxlen = 400 
        input1 = Input(shape=(maxlen,))
        X1 = Embedding(max_features, 256, dropout=0.1,mask_zero=True)(input1)
        X1 = LSTM(64, dropout_W=0.2, dropout_U=0.2)(X1)
        X = Dense(32)(X1)
        X = Dense(1)(X)
        y = Activation('sigmoid')(X)   
        rnn = Model(input=[input1], output=y)
        rnn.compile(loss='binary_crossentropy',optimizer = 'adam') 
        return rnn

    def resnet(self):
        """ This method creates a resnet50 network using the Keras API"""
        model = keras.applications.resnet50.ResNet50(include_top=True,weights=None)
        model.layers.pop()
        last = model.layers[-1].output
        x = Dense(102, activation="softmax")(last)
        model = Model(model.input, x)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model



    def test_resnet_save_graph(self):
        ''' Tests the save_graph function from dist_exec.py using
        a sample resnet network defined using the Keras API
        The function should extract the graph from the existing
        Keras session and save in a prototxt format in a newly
        created Graphs Directory'''
        resnet = self.resnet()
        environ = ExecutionEnvironment(bucket_name='sample1', prefix='sample2/',
                                 epochs=1, batch_size=32, opt='adam',
                                 test=True)
        graph_number=str(environ.save_graph())
        #Check if a graph is created for 0th graph
        assert_equal(os.path.isfile('graphs/graph'+graph_number+'/Graph.prototxt'), True)
        os.popen('rm -r graphs')


    def test_rnn_save_graph(self):
        ''' Tests the save_graph function from dist_exec.py using
        a sample RNN defined using the Keras API
        The function should extract the graph from the existing
        Keras session and save in a prototxt format in a newly
        created Graphs Directory'''
        rnn = self.rnn()

        environ = ExecutionEnvironment(bucket_name='sample1', prefix='sample2/',
                                 epochs=1, batch_size=32, opt='adam',
                                 test=True)
        graph_number=str(environ.save_graph())
        #Check if a graph is created for 0th graph
        assert_equal(os.path.isfile('graphs/graph'+graph_number+'/Graph.prototxt'), True)
        os.popen('rm -r graphs')



    def test_create_run_scripts(self):
        ''' Tests the create_run_scripts function from dist_exec.py.
        The function should create a seperate run script for each
        VM in the cluster. The runscripts should have required Ips.
        Movement of run scripts to the VMs from the parameter server
        is not included in this test. (depends on networkin AWS)'''
        rnn = self.rnn()
        environ = ExecutionEnvironment(bucket_name='sample1', prefix='sample2/',
                                 epochs=1, batch_size=32, opt='adam',
                                 test=True)
        environ.create_run_scripts()
        #Check if PS Run Script is created
        parameter_server_run_script_exists = os.path.isfile('run.sh')
        assert_equal(parameter_server_run_script_exists,True)
        #Check is Worker run scripts are created for all workers
        for ip in environ.worker_ips:
            worker_run_script_exists = os.path.isfile('runscripts/'+ip+'.sh')
            assert_equal(worker_run_script_exists,True)
        os.popen('rm -r runscripts')
        os.popen('rm run.sh')





     

    

   