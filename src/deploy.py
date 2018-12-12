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

""" EasyDist's deployment module that takes UI Inputs
and uses them to provision cloud resources. Current
support is only for the AWS environment"""
from subprocess import Popen
import os
import boto3

USER = 'ec2-user'

class AWS:
    """ Class for AWS integration and cluster provisioning """
    def __init__(self, expriment_name, iam_role, worker_type, worker_num,
                 worker_size=100, ps_type='t2.nano'):

        self.experiments_directory = '../experiments/'+expriment_name
        if os.path.isdir(self.experiments_directory) is False:
            os.makedirs(self.experiments_directory)
        self.resource_name = self.experiments_directory+'/resources'+'.txt'
        self.iam_role = iam_role
        self.worker_num = worker_num
        self.worker_type = worker_type
        self.worker_size = worker_size
        self.ps_type = ps_type
        self.ec2 = boto3.client('ec2')
        self.ec2_resource = boto3.resource('ec2')
        self.sg_id = None
        self.all_instances = None
        self.all_ips = None
        self.ps_id = None
        self.worker_ids = None
        self.key = None
        self.resource_name = None




    def create_security_group(self):
        """ Checks if an easyDist Security group exists
        already and creates one if not"""
        print("Checking for EasyDist Security Group")
        all_groups = self.ec2.describe_security_groups()
        exists = False

        #Check If the Group Has Already Been Created and create if not.
        for group in all_groups['SecurityGroups']:
            if group['GroupName'] == 'easyDist':
                self.sg_id = group['GroupId']
                exists = True
                break

        #Create an easyDist Security Group If it does not exist
        if exists is False:
            print("Creating EasyDist Security Group")
            sec_group = self.ec2_resource.create_security_group(
                GroupName='easyDist',
                Description='easyDist Security Group'
                )
            sec_group.authorize_ingress(IpProtocol="-1", CidrIp="0.0.0.0/0", FromPort=-1, ToPort=-1)
            self.sg_id = sec_group.id

        print("Security Group Configured")


    def create_key(self):
        """ Checks if an easyDist key exists already
        and if not creates one to be used """
        directory_name = os.path.dirname(os.path.abspath(__file__))
        key_location = directory_name+ '/aux/easyDist.pem'
        print("Checking for EasyDist KeyPair")
        #Determine if the needed key is already present
        exists = os.path.isfile(key_location)
        if exists is False:
            print('inside')
            print("Creating the EasyDist KeyPair")
            #Check If key has been created but is not present at the requuired location
            all_keys = self.ec2.describe_key_pairs()
            for key in all_keys['KeyPairs']:
                if key['KeyName'] == 'easyDist':
                    _ = self.ec2.delete_key_pair(KeyName='easyDist')
                    break
            #Create an EasyDist Key Pair and save it in aux
            response = self.ec2.create_key_pair(KeyName='easyDist')
            self.key = response['KeyMaterial']
            with open(key_location, 'w') as key_file:
                key_file.write(self.key)

        print("EasyDist KeyPair Configured")


    def write_file(self):
        """Creates a text file containing the
        ip addresses and resournce IDs of the provisioned
        virtual machines """

        #Writes the Public Ips of the launched resources to a resource file
        with open(self.resource_name, "w") as res_file:
            res_file.write(' '.join(self.ps_id+self.worker_ids))
            res_file.write('\n')
            #res_file.write(' '.join(self.psIp+self.workerIps))
            res_file.write(' '.join(self.all_ips))
            res_file.write('\n')
        print(".......\nMachine Launch Complete\n.......\n")



    def launch(self):
        """ Launches the required parameter server
        and worker virtual machines """
        self.create_security_group()
        self.create_key()

        #First Launch PS Machine
        print("Launching PS Machine")
        #Get the ami-id for the most recent Amazon Linux Deep Learning AMI
        ami = os.popen('aws ec2 describe-images --filters "Name=name,Values=Deep Learning AMI (Amazon Linux) Version*" --query \'sort_by(Images, &CreationDate)[-1].ImageId\' --output text').read()[:-1]

        ps_response = self.ec2_resource.create_instances(
            ImageId=ami,
            InstanceType=self.ps_type,
            MinCount=1,
            MaxCount=1,
            SecurityGroupIds=[self.sg_id],
            KeyName='easyDist',
            IamInstanceProfile={'Name': self.iam_role}
            )
        self.ps_id = [p.id for p in ps_response]
        #Now Launch Worker Machines
        print("Launching Worker Machines")
        worker_response = worker_response = self.ec2_resource.create_instances(
            ImageId=ami,
            InstanceType=self.worker_type,
            MinCount=self.worker_num,
            MaxCount=self.worker_num,
            SecurityGroupIds=[self.sg_id],
            KeyName='easyDist',
            IamInstanceProfile={'Name': self.iam_role},
            BlockDeviceMappings=[{
                'DeviceName':'/dev/xvda',
                'Ebs': {
                    'DeleteOnTermination': True,
                    'VolumeSize': self.worker_size,
                    'VolumeType': 'gp2'
                    },
                'NoDevice': ''
                }])
        self.worker_ids = [w.id for w in worker_response]

        print(".......\nWaiting till Instances reach Running State...")
        ps_response[0].wait_until_running()
        worker_response[0].wait_until_running()
        print("Instances Running")
        self.all_instances = self.ec2_resource.instances.filter(
            InstanceIds=self.ps_id+self.worker_ids)
        self.all_ips = [instan.public_dns_name for instan in self.all_instances]
        self.write_file()


    def terminate(self):
        """ Terminates all virtual machines of the cluster """
        print("Terminating Instances")
        self.ec2_resource.instances.filter(InstanceIds=self.ps_id+self.worker_ids).terminate()
        print("Waiting Till Instances are terminated")

        list(self.ec2_resource.instances.filter(InstanceIds=self.ps_id))[0].wait_until_terminated()
        list(self.ec2_resource.instances.filter(
            InstanceIds=self.worker_ids))[0].wait_until_terminated()

        print("All Instances Terminated\n..........\n")

    def login(self):
        """ Logs in the user into the paramter server
        virtual machines """
        cmd = "ssh "+"-i ./aux/easyDist.pem "+ "-o StrictHostKeyChecking=no "
        cmd = cmd +USER+"@"+self.all_ips[0]
        _ = Popen(cmd, shell=True)

    def transfer(self):
        """ Moves all the required easyDist files
        and dependencies to all of the cluster virtual
        machines """
        for ip_address in self.all_ips:
            os.system('scp -i ./aux/easyDist.pem -o StrictHostKeyChecking=no' \
            ' -r ./aux %s@"%s": ' % (USER, ip_address))
            os.system('scp -i ./aux/easyDist.pem -o StrictHostKeyChecking=no' \
            ' -r ../tests %s@"%s": ' % (USER, ip_address))
            for file_name in ['./dataReader.py',
                              './codeFile.py',
                              'distExec.py',
                              self.resource_name]:
                os.system('scp -i ./aux/easyDist.pem -o StrictHostKeyChecking=no ' \
                    '-r %s %s@"%s": ' % (file_name, USER, ip_address))
