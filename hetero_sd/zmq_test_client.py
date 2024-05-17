import zmq
import torch
import sys
import time
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://192.168.0.132:1919")
total_transmission_time = 0
for i in range(128):
    test_packet = torch.rand(1, 2, 32000)
    print(f"Size of tensor: {sys.getsizeof(test_packet)/(10^6)} MB")
    start_time = time.time()
    socket.send_pyobj(test_packet)
    response = socket.recv_pyobj()
    end_time = time.time()
    print(f"Received response")
    print(f"Transmission time: {end_time - start_time} seconds")
    total_transmission_time += end_time - start_time
print(f"Total transmission: {total_transmission_time} seconds")