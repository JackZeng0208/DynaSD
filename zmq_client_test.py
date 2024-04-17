import zmq

SERVER_IP = "192.168.0.132"
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect(f"tcp://{SERVER_IP}:1919")

while True:
    message = "Message from client 2"
    socket.send_string(message)
    reply = socket.recv()
    # print(f"Received reply: {reply}")