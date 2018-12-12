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

""" This module acts as the interface between the user's Keras
code and the distributed tensorflow session which will be used
to train the network.
This module is responsible mainly for graph extraction, transfer
and the issuance of the training call """
from __future__ import print_function
import subprocess
import os
import keras.backend as K
import tensorflow as tf
import boto3

KEY_NAME = 'easyDist.pem'
USER = 'ec2-user'

class ExecutionEnvironment:
    """ This class is responsible for extracting the Keras
    graph and using it to set up a distributed tensorflow cluster
    across all the VMs."""
    def __init__(self, bucket_name, prefix, epochs, batch_size, opt, port="2222",test=False):
        self.epochs = epochs
        self.batch_size = batch_size
        self.opt = opt
        self.test = test
        worker_ip_file = open("resources.txt", "r")
        #Extract IP addresses (remove the trailing new line chrachter)
        self.instance_ids, all_ips = [line[:-1].split() for line in worker_ip_file.readlines()]
        self.ps_ip = all_ips[0]
        self.worker_ips = all_ips[1:]
        self.bucket_name = bucket_name
        if test is False: 
            self.client = boto3.client('s3')
            self.data_chunks = [a['Key'] for a in
                                self.client.list_objects(Bucket=bucket_name, Prefix=prefix)['Contents']]
        else:
            self.data_chunks = ['dummy_1','dummy_2','dummy_3','dummy_4']
        self.saved = -1
        self.trained = -1
        self.create_graph_directory = True
        self.create_run_directory = True
        self.port = port
        self.executions = []
        
    def fit(self):
        """ Starts training the network after the graph
        transfer and tensorflow setup is complete"""
        print("Creating Execution Scripts on Remote Workers")
        self.create_run_scripts()
        self.save_graph()
        self.transfer_graph()
        self.trained += 1
        execution = Execution(self.trained, self.worker_ips)
        execution.start_training()
        self.executions.append(execution)
    ''' 
    def status(self, worker, error_file=True, lines=5):
        """Gives the current training status for a single
        worker"""
        self.executions[-1].give_status(worker, error_file, lines)
    '''

    def save_graph(self):
        """ Extracts the current Keras Computational
        Graph and saves it """
        self.saved += 1
        if self.create_graph_directory:
            subprocess.call(["mkdir", "graphs"])
            self.create_graph_directory = False
        sess = K.get_session()
        saver = tf.train.Saver()
        logdir_name = "graphs/graph"+str(self.saved)
        tf.train.write_graph(graph_or_graph_def=sess.graph_def,
                             logdir=logdir_name,
                             name='Graph.prototxt')
        saver.save(sess, logdir_name+"/Graph")


        graph = tf.get_default_graph()
        #Listing all placeholders in graph
        placeholders = [op for op in graph.get_operations() if op.type == "Placeholder"]
        op_names = [str(op.name)+':0' for op in placeholders]
        print('\n\nYour Placeholder Tensors are ', op_names)
        print('Please ensure that this ordering is followed \
               when in the dataReader File. (Inputs, Outputs, Weights)')
        print('By Default, the weights are set to 1')
        print('----------------------------------------')
        return self.saved

    def transfer_graph(self, graph_number=-1):
        """Transfers Most Recent Graph to all VMs """
        if graph_number == -1:
            graph_number = self.saved
        for i in range(len(self.worker_ips)):
            print('Transferring')
            os.system('ssh -i ./aux/%s -o StrictHostKeyChecking=no \
                %s@"%s" \'mkdir Graph\'' % (KEY_NAME, USER, self.worker_ips[i]))

            os.system('scp -i ./aux/%s -o StrictHostKeyChecking=no \
                dataReader.py %s@"%s":.' % (KEY_NAME, USER, self.worker_ips[i]))

            os.system('scp -i ./aux/%s -o StrictHostKeyChecking=no \
                trainer.py %s@"%s":.' % (KEY_NAME, USER, self.worker_ips[i]))

            if os.system('scp -i ./aux/%s -o StrictHostKeyChecking=no \
                ./graphs/graph"%s"/* %s@"%s":Graph/.'
                         % (KEY_NAME, str(graph_number), USER, self.worker_ips[i])) == 0:
                print('Successfully Transferred Graph to Worker ', i)
            else:
                print('Failed to Transfer Graph to Worker ', i)

    def create_run_scripts(self):
        """ Creates the run scripts for each machines,
        This function also determines which chunks of data
        are mapped to which worker"""
        if self.create_run_directory:
            os.system("mkdir runscripts > /dev/null")
            self.create_run_directory = False
        #Create IP address with ports
        ps_ip_with_port = self.ps_ip+':'+self.port
        worker_ips_with_port = [ip+':'+self.port for ip in self.worker_ips]

        file_ps = open("run.sh", "w")
        common_string = """mkdir models > /dev/null
        source activate tensorflow_p36
        nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,\
        pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,\
        utilization.memory --format=csv -l 30 > GPUlog.csv & python trainer.py \
        --epochs=%s --batch_size=%s --optimizer=%s --ps_hosts=%s \
        --worker_hosts=%s""" % (str(self.epochs),
                                str(self.batch_size), self.opt,
                                ps_ip_with_port,
                                ",".join(worker_ips_with_port))


        ps_string = common_string+""" -j_name ps -t_id 0 --bucket=. --keys=.
        zip outs_train.zip ./outs_train/*
        zip errs_train.zip ./errs_train/*
        zip outs_data.zip ./outs_data/*
        zip ps.zip outs_train.zip outs_data.zip errs_train.zip ip* GPUlog.csv"""
        file_ps.write(ps_string)
        file_ps.close()

        chunks_per_worker = len(self.data_chunks)/len(self.worker_ips)

        for worker, ip_address in enumerate(self.worker_ips):
            start_index = worker * chunks_per_worker
            end_index = min((worker+1)*chunks_per_worker, len(self.data_chunks))
            worker_string = common_string+""" -j_name worker -t_id %s --bucket=%s --keys=%s
            zip -r m_%s_logs.zip ip* GPU*
            zip -r m_%s_models.zip log.csv epochLog.csv models/*
            rm -r models/""" % (str(worker), self.bucket_name,
                                ",".join(self.data_chunks[int(start_index):int(end_index)]),
                                str(worker), str(worker))
            filename = "runscripts/%s.sh"%(ip_address)
            run_script = open(filename, "w")
            run_script.write(worker_string)
            run_script.close()
            if self.test is False:
                os.system('scp -i ./aux/%s -o StrictHostKeyChecking=no \
                    ./"%s" %s@"%s":run.sh' % (KEY_NAME, filename, USER, ip_address))


class Execution:
    """ This class starts and monitors the actual training by
    issuing the required shell commands"""

    def __init__(self, experiment_number, worker_ips):
        self.worker_ips = worker_ips
        self.experiment_number = experiment_number

    def start_training(self):
        """Starts Distributed Tensorflow Training of the graph"""
        os.system('sh ./aux/startTraining.sh %d'%(self.experiment_number))
    '''
    def give_status(self, worker, error_file=True, lines=5):
        """Gives the current training status for a single worker"""
        if worker < 0 or worker >= len(self.worker_ips):
            print('Sorry worker id out of range')
            return
        print('\n*************************************')
        if error_file:
            print('Last %d lines of Error File For Worker %d ' % (lines, worker))
            os.system('tail -"%s" experiments/exp"%s"/errs_train/"%s"' % (str(lines),
                                                                          str(self.experiment_number),
                                                                          str(self.worker_ips[worker])))
            print('----------------------------')
        print('Last %d lines of Output File For Worker %d ' % (lines, worker))
        os.system('tail -"%s" experiments/exp"%s"/outs_train/"%s"' % (str(lines),
                                                                      str(self.experiment_number),
                                                                      str(self.worker_ips[worker])))
        print('\n*************************************')
    '''
