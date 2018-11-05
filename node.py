import time
import os
from multiprocessing import Queue
from threading import Thread
from queue import Queue as queue_wrapper
from collections import deque
import socket
import yaml
from keras.models import Sequential
from keras import layers
from keras.layers import InputLayer
import numpy as np
import tensorflow as tf
import ConfigParser

import avro.ipc as ipc
import avro.protocol as protocol

PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)

# data packet format definition
PROTOCOL = protocol.parse(open(DIR_PATH + '/resource/message/image.avpr').read())

# home dir
HOME = os.environ['HOME']


class Node:
    """
        Node class for handling model prediction and get according stats
        for a module.

        Attributes:
            instance: Class attributes to achieve singleton for this node
                        class.
            model: The model created for this node.
            total_time: Total timing of one data packet from being received
                        to being successfully processed.
            prediction_time: Total timing of model inference.
            input: Store the data packets from other nodes.
            ip: Store all IP addresses of available devices.
            debug: If print out verbose information.
            graph: Default graph with Tensorflow backend.
            input_shape: Input shape for model on this node.
            merge: Number of previous layers merged into this layer.
            split: Number of next layers to process current data.
            op: Operation for merging the data, could be no operation.
            threads: Collections of currently running threads for later house keeping.
            run: Variable to control thread.
    """

    instance = None

    @classmethod
    def create(cls):
        if cls.instance is None:
            cls.instance = cls()

            # Get ip address and create model according to ip config file.
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(('8.8.8.8', 80))
                ip = s.getsockname()[0]
            except Exception:
                ip = '127.0.0.1'
            finally:
                s.close()

            node_config = ConfigParser.ConfigParser()
            node_config.read(HOME + '/node.cfg')
            sys_model_name = node_config.get('Node Config', 'model', 0)
            sys_node_count = node_config.get('Node Config', 'system', 0)
            node_id = node_config.get('IP Node', ip, 0)

            with open(DIR_PATH + '/resource/system/' + sys_model_name + '/' + sys_node_count + '/config.json') as f:
                system_config = yaml.safe_load(f)[node_id]
                cls.instance.id = node_id

                model = Sequential()

                # The model config is predefined. Extract each layer's config
                # according to the config from system config.
                with open(DIR_PATH + '/resource/model/' + sys_model_name + '/config.json') as f2:
                    model_config = yaml.safe_load(f2)
                    for layer_name in system_config['model']:
                        class_name = model_config[layer_name]['class_name']
                        config = model_config[layer_name]['config']
                        input_shape = model_config[layer_name]['input_shape']
                        layer = layers.deserialize({
                            'class_name': class_name,
                            'config': config
                        })
                        model.add(InputLayer(input_shape))
                        model.add(layer)

                cls.instance.model = model if len(model.layers) != 0 else None
                cls.log(cls.instance, 'model finishes', model.summary())

                for n_id in system_config['devices']:
                    ip = node_config.get('Node IP', n_id, 0)
                    cls.instance.ip.put(ip)

                cls.instance.merge = int(system_config['merge'])
                cls.instance.split = int(system_config['split'])
                cls.instance.op = system_config['op']

                if cls.instance.model:
                    shape = list(model.input_shape[1:])
                    shape[-1] = shape[-1] / cls.instance.merge if cls.instance.op == 'cat' else shape[-1]
                    cls.instance.input_shape = tuple(shape)

        return cls.instance

    def __init__(self):
        self.model = None
        self.total_time = 0.0
        self.prepare_data = 0.0
        self.prediction_time = 0.0
        self.input = queue_wrapper(size=10)
        self.ip = Queue()
        self.id = ''
        self.debug = False
        self.graph = tf.get_default_graph()
        self.input_shape = None
        self.merge = 0
        self.split = 0
        self.op = ''
        self.frame_count = 0
        self.threads = deque([])
        self.run = True
        self.sample_output_shape = None

        Thread(target=self.inference).start()

    def inference(self):
        # wait for the first packet
        while self.total_time == 0.0:
            time.sleep(0.1)

        while self.run:
            start = time.time()
            # get data from the queue
            seq = self.input.dequeue(self.merge)

            if self.op == 'cat':
                X = np.concatenate(seq, axis=-1)
            elif self.op == 'add':
                X = np.sum(seq, axis=0)
            else:
                X = seq[0]

            if X is not None and self.model is not None:
                with self.graph.as_default():
                    output = self.model.predict(np.array([X]))
                    for _ in range(self.split):
                        Thread(target=self.send, args=(output,)).start()

                self.frame_count += 1
                self.prediction_time += time.time() - start

                if self.frame_count == 30:
                    self.stats()
                    self.run = False

    def receive(self, msg, req):
        start = time.time()
        self.total_time = time.time() if self.total_time == 0.0 else self.total_time

        bytestr = req['input']
        datatype = np.uint8 if req['type'] == 8 else np.float32

        # adapt to merge layer
        if not self.input_shape:
            shape = tuple([int(entry) for entry in req['shape'].split(' ')])
        else:
            shape = self.input_shape

        X = np.fromstring(bytestr, datatype).reshape(shape)
        self.input.enqueue(X)
        self.prepare_data += time.time() - start

    def send(self, X):
        ip = self.ip.get()

        client = ipc.HTTPTransceiver(ip, 12345)
        requestor = ipc.Requestor(PROTOCOL, client)

        data = dict()
        tmp = [str(entry) for entry in np.shape(X[0])]
        data['shape'] = ' '.join(tmp)
        data['input'] = X.tobytes()
        data['type'] = 32
        requestor.request('forward', data)

        client.close()
        self.ip.put(ip)

    def terminate(self):
        self.run = False

    def stats(self):
        with open(HOME + '/stats', 'w+') as f:
            result = '++++++++++++++++++++++++++++++++++++++++\n'
            result += '+                                      +\n'
            result += '+{:^38s}+\n'.format('SERVER: ' + self.id)
            result += '+                                      +\n'
            result += '+{:>19s}: {:6.3f}           +\n'.format('frame rate', self.frame_rate)
            result += '+                                      +\n'
            result += '+{:>19s}: {:6.3f}           +\n'.format('overhead', self.overhead)
            result += '+{:>19s}: {:6.3f}           +\n'.format('utilization', self.utilization)
            result += self.input.log()
            f.write(result)

    def log(self, step, data=''):
        """
            Log function for debug. Turn the flag on to show each step result.
            Args:
                step: Each step names.
                data: Data format or size.
        """
        if self.debug:
            print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
            for k in range(0, len(step), 68):
                print '+{:^68.68}+'.format(step[k:k + 68])
            for k in range(0, len(data), 68):
                print '+{:^68.68}+'.format(data[k:k + 68])
            print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
            print

    @property
    def utilization(self):
        return np.float32(self.prediction_time) / (time.time() - self.total_time)

    @property
    def overhead(self):
        return np.float32(self.prepare_data) / (time.time() - self.total_time)

    @property
    def frame_rate(self):
        return np.float(self.frame_count) / (time.time() - self.total_time)
