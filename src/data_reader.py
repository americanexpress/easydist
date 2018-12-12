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

""" This module provides the user with handles
to the training files present on S3. It also
interfaces with trainer.py and provides batched
training inputs"""
import time
import boto3

class Dataset:
    """ Controls the flow of data to the network
    This class gets instantiated and used from
    trainer.py """
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.objects = []

        self.train_size = 0
        self.client = boto3.client('s3')
        self.resource = boto3.resource('s3')

    def give_num_batches(self, batch_size):
        """ Returns total number of batches"""
        return self.train_size / batch_size

    def give_next(self, batch_size, i):
        """ Gives batch number 'i' of size batch_size"""
        start = i*batch_size
        end = min((i+1)*batch_size, self.train_size)
        return [inp[start:end] for inp in self.inputs] +\
               [out[start:end] for out in self.outputs] +\
               [[1]*(end-start)]

    def read_data(self, bucket, keys):
        """This method returns the concatenation of the
        specifeid chunks as a pandas dataframe.
        Preprocessing if any should be done on this
        dataframe in the Preprocessing class of the user's
        preprocessing file.
        This function also gets the S3 Objects and return
        the list containingthe objects to the preprocessing function"""
        start = time.time()
        from preprocessing import Preprocessing
        print(len(keys), " Training Files Present")

        for key in range(len(keys)):
            obj = self.resource.Object(bucket, key).get()['Body']
            self.objects.append(obj)
        pre = Preprocessing()
        pre.preprocess(keys, self.objects)
        return time.time()-start
