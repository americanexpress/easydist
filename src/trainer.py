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

"""This module sets up a barebones distributed
tensorflow cluster, loads the user's keras graph
into it and then trains it using the data given
to it by data_reader.py"""
from __future__ import print_function
import argparse
import time
import keras.backend as K
import tensorflow as tf
from data_reader import Dataset

def main():
    """ The main driver function which creates
    the tensorflow cluster and communicates between the different
    nodes and trains the graph"""
    tf.reset_default_graph()
    #sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
    ps_hosts = FLAGS.ps_hosts.split(",")
    #Create list of Paramere Server IP address from commandline input
    worker_hosts = FLAGS.worker_hosts.split(",")
    # Create list of worker IP addresses from command line input
    keys = FLAGS.keys.split(",")
    bucket = FLAGS.bucket

    cluster = tf.train.ClusterSpec({"ps":ps_hosts, "worker":worker_hosts})
    #Creating Tensorflow Cluster Specification
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    #Create a server object for individual machines


    if FLAGS.job_name == "ps":
        server.join()

    elif FLAGS.job_name == "worker":
        start_time = time.time()
        epochs = FLAGS.epochs
        batch_size = FLAGS.batch_size

        with tf.device(tf.train.replica_device_setter( \
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            #Import the Computational Graph for Model to be trained
            saved = tf.train.import_meta_graph('./Graph/Graph.meta', clear_devices=True)
            graph = tf.get_default_graph()

            #Obtain Placeholders to be fed into graph while training
            placeholders = [op for op in graph.get_operations() if op.type == "Placeholder"]
            #Listing all placeholders in graph
            op_names = [str(op.name)+':0' for op in placeholders]
            feed_dict_keys = [graph.get_tensor_by_name(op) \
                              for op in op_names] + [K.learning_phase()]

            #Obtain Loss Tensor from Graph Definition and wrap it in an optimizer
            loss_name = [op.name +':0'  for op in graph.get_operations() if 'loss' in op.name][-1]
            cost_op = graph.get_tensor_by_name(loss_name)


            optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)


            global_step = tf.train.create_global_step()
            train_op = optimizer.minimize(cost_op, global_step=global_step)

            #Initilazer operation of the computational operation
            _ = graph.get_operation_by_name("init")

            #Global step to co-ordinate training across all workers

            #Initialize data class (See data_reader.py) and read the related chunks
            data = Dataset()
            read_time = data.read_data(bucket, keys)


            num_batches = int(data.give_num_batches(batch_size))


            init_op = tf.global_variables_initializer()

            #Create a TensorFlow training supervisor which will
            #coordinate across all the worker and the parameter server
            tf_supervisor = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                                logdir='./models/',
                                                global_step=global_step,
                                                init_op=init_op)

        avg_ep = 0 #To keep track of average epoch loss
        min_loss = -1 #Will be updated with lowest loss
        step = 0 #To keep track of the global step across all machines

        with tf_supervisor.prepare_or_wait_for_session(server.target) as sess:
            #Restore the weight values of the imported graph within the supervisor session
            saved.restore(sess, tf.train.latest_checkpoint('./Graph/.'))
            print('Your Input Tensors Are: ', " , ".join(op_names))
            print('Your Loss Tensor is ', loss_name)
            print(num_batches, " are there")
            for epoch in range(epochs):
                epoch_start_time = time.time()
                epoch_cost = 0

                #----------------Epoch Start----------------------
                for i in range(num_batches):

                    feed_dict_values = data.give_next(batch_size, i) + [1]
                    feed_dict = dict(zip(feed_dict_keys, feed_dict_values))

                    _, step, cost = sess.run([train_op, global_step, cost_op], feed_dict=feed_dict)

                    epoch_cost = epoch_cost+cost

                    if i%500 == 0:
                    #Print Current Progress in Epoch after every 500 batches
                        print('Finished Batch ', i,
                              '; Time elapsed in epoch is ',
                              time.time()-epoch_start_time)
                        print('At step ', step, ' ; Current Epoch Loss is ', epoch_cost)
                        print('Total Batches are ', num_batches)
                        print('---------')
                #-----------------Epoch Over----------------------

                epoch_total_time = time.time()-epoch_start_time
                avg_ep = avg_ep+epoch_total_time

                print('Finished Epoch ', epoch, ' in ', epoch_total_time)
                print('At step ', step, ' ; Epoch Loss was ', epoch_cost)

                if min_loss == -1 or min_loss > epoch_cost:
                    min_loss = epoch_cost
                    save_model(sess, tf_supervisor, step)

                print('**************************************')

            print('Total time taken was ', time.time()-start_time)
            print('Time Taken to Pre-Process Data was ', read_time)
            print('Average Epoch Training Time was', float(avg_ep)/float(epochs))
            print('Best Loss Was ', min_loss)
            print('-*-*-*-*-*-*-*-*-*')


def save_model(sess, tf_supervisor, step):
    """ Saves the current model (graph + weights) as the
    epochs proceed"""
    tf.train.write_graph(graph_or_graph_def=sess.graph_def,
                         logdir='./models/step_'+str(step)+'/',
                         name='my_graph.prototxt')
    _ = tf_supervisor.saver.save(sess, "./models/step_"+ str(step)+"/"+"mmmg_model")



if __name__ == '__main__':
    #Parses input arguments required for building model
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("-buc", "--bucket", type=str, required=True)
    PARSER.add_argument("-keys", "--keys", required=True)
    PARSER.add_argument("-j_name", "--job_name", type=str, required=True)
    PARSER.add_argument("-t_id", "--task_index", type=int, required=True)
    PARSER.add_argument("-p_hosts", "--ps_hosts", type=str, required=True)
    PARSER.add_argument("-w_hosts", "--worker_hosts", type=str, required=True)
    PARSER.add_argument("-e", "--epochs", type=int, required=True)
    PARSER.add_argument("-bs", "--batch_size", type=int, required=True)
    PARSER.add_argument("-opt", "--optimizer", type=str, required=True)
    FLAGS, UNPARSED = PARSER.parse_known_args()
    main()
