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

# Code:
# User can pre-process the raw data based on various data sets by overriding the following # 
# preprocessing class. User needs to feed in user’s own code in the preprocess method. User should not modify the method definition and return arguments #

import pandas as pd
import numpy as np
import time
import boto3
import io
import os

class Preprocessing:
    def __init__(self):
        print('Start Pre-processing')

    def preprocess(self,keys,objects):
        #Pre Process the data and ensure that the return signature is (inputs,outputs,train_size)        
        print("Processing the Data")
        '''
        Reading a non csv file  can be done as         
        response = objects[0].read()
        Embed your pre-processing code here
        After processing, it returns inputs, outputs and train_size
        '''
        #Reading the Data and concatenating the chunks
        loop = 0
        df = None
        for i in len(objects):
            print('Reading Key: ', keys[i])
            temp = pd.read_csv(io.BytesIO(objects[i].read()), encoding='utf8')
            if loop == 0:
                df = temp
                loop += 1
            else:
                df = pd.concat([df,temp])
        x1 = df['x1'].values
        x1 = [list(map(float,y.split(' ')))for y in x1]

        labels = [int(x) for x in df['y'].values]
        labels = np.array(labels)
        labels = labels.reshape(len(labels),1)

        inputs=[np.array(x1)]
        outputs=[labels]
        train_size = len(x1)

        print('Training Size is ', train_size)
        return (inputs,outputs,train_size)
