"""
    This module is called initial because it initializes all request
    from this node. It will simulates a (224, 224, 3) size image data
    packet and send to the first node in the distributed system and wait
    for the response from the last layer.
"""
import time
import threading
from threading import Thread
import os
import signal

import avro.ipc as ipc
import avro.protocol as protocol
import numpy as np

from initial import Initializer

PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)

# data packet format definition
PROTOCOL = protocol.parse(open(DIR_PATH + '/resource/message/image.avpr').read())


def send_request(frame):
    """
        This function sends data to next layer. It will pop an available
        next layer device IP address defined at IP table, and send data
        to that IP. After, it will put the available IP back.
        Args:
            bytestr: The encoded byte string for image.
            mode: Specify next layer option.
    """
    init = Initializer.create()
    queue = init.queue

    addr = queue.get()
    client = ipc.HTTPTransceiver(addr, 12345)
    requestor = ipc.Requestor(PROTOCOL, client)

    data = dict()
    tmp = [str(entry) for entry in frame.shape]
    data['shape'] = ' '.join(tmp)
    data['input'] = frame.astype(np.uint8).tobytes()
    data['type'] = 8
    requestor.request('forward', data)

    client.close()
    queue.put(addr)


def master():
    """
        Master function for real time model inference. A basic while loop
        gets one frame at each time. It appends a frame to deque every time
        and pop the least recent one if the length > maximum.
    """
    init = Initializer.create()

    for _ in range(1000):
        # current frame
        ret, frame = 'unknown', np.random.random_sample(init.input_shape) * 255
        for _ in range(init.split):
            Thread(target=send_request, args=(frame,)).start()
        time.sleep(init.interval)
        init.send()


def main():
    signal.signal(signal.SIGTERM, terminate)
    master()


def terminate(signum, stack):
    init = Initializer.create()
    init.terminate()
    for thread in threading.enumerate():
        if thread != threading.current_thread():
            thread.join()


if __name__ == '__main__':
    main()
