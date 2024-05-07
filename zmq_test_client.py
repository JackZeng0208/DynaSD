import zmq
import torch
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:1919")

while True:
    tensor = socket.recv_pyobj()
    socket.send_pyobj(tensor)