import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp:*//1919")

while True:
    name = socket.recv().decode()
    print(f"Received request: {name}")
    response = f"Hello, {name}!"
    socket.send(response.encode())