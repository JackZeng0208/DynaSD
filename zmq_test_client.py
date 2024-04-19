import zmq

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://192.168.0.132:1919")

while True:
    name = input("Enter your name (or 'quit' to exit): ")
    if name == "quit":
        break

    socket.send(name.encode())
    response = socket.recv()
    print(f"Received response: {response.decode()}")