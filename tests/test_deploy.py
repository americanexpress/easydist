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

''' This module tests the AWS class
within deploy.py. It checks the AWS integrations
of easydist. Aws configuration needs to be performed
prior to running these tests'''
import sys
import os
import hashlib
from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_raises
from nose.tools import raises
import boto3
sys.path.append('../src/')
sys.path.append('.')
from deploy import AWS

class TestAWS(object):

	def give_md5(self,fname):
		''' Helper function to return md5
		hash of a file'''
		hash_md5 = hashlib.md5()
		with open(fname, "rb") as f:
			for chunk in iter(lambda: f.read(4096), b""):
				hash_md5.update(chunk)
		return hash_md5.digest()

	def test_create_key(self):
		''' Test that a new key is created when no key exists
		in the aux folder'''
		cluster = AWS(expriment_name='test', iam_role='sampleIAsM', worker_type='c3.2xlarge', worker_num=1)
		os.popen('rm ../src/aux/easyDist.pem')
		cluster.create_key()
		assert_equal(os.path.isfile('../src/aux/easyDist.pem'), True)

	def test_existing_key(self):
		'''Test that create_key does not overwrite key 
		when one already exists'''
		cluster = AWS(expriment_name='test', iam_role='sampleIAsM', worker_type='c3.2xlarge', worker_num=1)
		cluster.create_key()
		assert_equal(os.path.isfile('../src/aux/easyDist.pem'), True)
		existing_key_hash = self.give_md5('../src/aux/easyDist.pem')

		#call Create Key when a key already exists
		cluster.create_key()
		assert_equal(os.path.isfile('../src/aux/easyDist.pem'), True)
		second_key_hash = self.give_md5('../src/aux/easyDist.pem')
		assert_equal(existing_key_hash, second_key_hash)


	def test_create_security_group(self):
		'''Tests that an easyDist Security Group is correctly
		created by the deploy module'''
		cluster = AWS(expriment_name='test', iam_role='sampleIAsM', worker_type='c3.2xlarge', worker_num=1)
		cluster.create_security_group()
		all_groups = cluster.ec2.describe_security_groups()
		exists = False
		#Check If the Group Has Been Created
		for group in all_groups['SecurityGroups']:
			if group['GroupName'] == 'easyDist':
				exists = True
				break
		assert_equal(exists, True)


	def test_write_file(self):
		''' Tests that the write_file function succesfully writes 
		a resources.txt file at the resource_name location'''
		cluster = AWS(expriment_name='test', iam_role='sampleIAsM', worker_type='c3.2xlarge', worker_num=1)
		cluster.resource_name = './resources.txt'
		os.popen('rm -r resources.txt')
		cluster.ps_id = ['sample_ps']
		cluster.worker_ids = ['samplew0','samplew1','samplew2', 'samplew3']
		cluster.all_ips = ['psip1','psip2','psip3','psip4']
		cluster.write_file()
		assert_equal(os.path.isfile('./resources.txt'),True)


