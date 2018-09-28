from multiprocessing import Queue
import time
import yaml
import socket
import os
import ConfigParser

PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)

HOME = os.environ['HOME']


class Initializer:
    """
        Singleton factory for initializer. The Initializer module has two timers.
        The node_timer is for recording statistics for block1 layer model inference
        time. The timer is for recording the total inference time from last
        fully connected layer.
        Attributes:
            queue: Queue for storing available block1 models devices.
    """
    instance = None

    @classmethod
    def create(cls):
        """ Utilize singleton design pattern to create single instance. """
        if cls.instance is None:
            cls.instance = Initializer()

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
            sys_block = node_config.get('Node Config', 'block', 0)
            node_id = node_config.get('IP Node', ip, 0)

            cls.instance.id = node_id

            # read ip resources from config file
            with open(DIR_PATH + '/resource/system/' + sys_model_name + '/' + sys_node_count
                      + '/' + sys_block + '/config.json') as f:
                configs = yaml.safe_load(f)
                config = configs[node_id]

                for n_id in config['devices']:
                    ip = node_config.get('Node IP', n_id, 0)
                    cls.instance.queue.put(ip)
                cls.instance.split = int(config['split'])
                cls.instance.input_shape = ([int(entry) for entry in config['input_shape'].split(' ')])
                cls.instance.interval = float(config['interval'])
        return cls.instance

    def __init__(self):
        self.queue = Queue()
        self.start = 0.0
        self.count = 0
        self.id = ''
        self.split = 0
        self.input_shape = None
        self.interval = 0.0
        self.run = True

    def send(self):
        self.start = time.time() if self.start == 0.0 else self.start
        self.count += 1

    def terminate(self):
        self.stats()
        self.run = False

    def stats(self):
        with open(HOME + '/stats', 'w+') as f:
            result = '++++++++++++++++++++++++++++++++++++++++\n'
            result += '+                                      +\n'
            result += '+{:^38s}+\n'.format('CLIENT: ' + self.id)
            result += '+                                      +\n'
            result += '+{:>19s}: {:6.3f}           +\n'.format('frame rate', self.frame_rate)
            result += '+                                      +\n'
            result += '++++++++++++++++++++++++++++++++++++++++\n'
            f.write(result)

    @property
    def frame_rate(self):
        return self.count / (time.time() - self.start)
