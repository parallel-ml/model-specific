"""
    This module shows the node for 4/6 nodes distributed system setup.
"""
import argparse
import os
import signal
import threading
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from SocketServer import ThreadingMixIn
import avro.ipc as ipc
import avro.protocol as protocol
from system.node import Node

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)

# read data packet format.
PROTOCOL = protocol.parse(open(DIR_PATH + '/resource/message/image.avpr').read())


class Responder(ipc.Responder):
    """ Responder called by handler when got request. """

    def __init__(self):
        ipc.Responder.__init__(self, PROTOCOL)

    def invoke(self, msg, req):
        """
            This function is invoked by do_POST to handle the request. Invoke handles
            the request and get response for the request. This is the key of each node.
            All models forwarding and output redirect are done here. Because the invoke
            method of initializer only needs to receive the data packet, it does not do
            anything in the function and return None.
            Because this is a node class, it has all necessary code here for handling
            different inputs. Basically the logic is load model as the previous layer
            request and run model inference. And it will send the current layer output
            to next layer. We write different model's code all here for the sake of
            convenience. In order to avoid long waiting time of model reloading, we
            make sure each node is assigned to a unique job each time, so it does not
            need to reload the model.
            Args:
                msg: Meta data.
                req: Contains data packet.
            Returns:
                None: It just acts as confirmation for sender.
            Raises:
                AvroException: if the data does not have correct syntac defined in Schema
        """

        node = Node.create()

        try:
            node.receive(msg, req)
            return
        except Exception, e:
            node.log('Error', e.message)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        """
            do_POST is automatically called by ThreadedHTTPServer. It creates a new
            responder for each request. The responder generates response and write
            response to data sent back.
        """
        self.responder = Responder()
        call_request_reader = ipc.FramedReader(self.rfile)
        call_request = call_request_reader.read_framed_message()
        resp_body = self.responder.respond(call_request)
        self.send_response(200)
        self.send_header('Content-Type', 'avro/binary')
        self.end_headers()
        resp_writer = ipc.FramedWriter(self.wfile)
        resp_writer.write_framed_message(resp_body)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """ Handle requests in separate thread. """


def main(cmd):
    signal.signal(signal.SIGTERM, terminate)

    node = Node.create()
    node.debug = cmd.debug

    server = ThreadedHTTPServer(('0.0.0.0', 12345), Handler)
    server.allow_reuse_address = True
    server.serve_forever()


def terminate(signum, stack):
    node = Node.create()
    node.terminate()
    for thread in threading.enumerate():
        if thread != threading.current_thread():
            thread.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='set to debug mode')
    cmd = parser.parse_args()
    main(cmd)
