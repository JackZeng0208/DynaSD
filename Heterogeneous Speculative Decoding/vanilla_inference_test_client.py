import zmq
import threading
import time
from datasets import load_dataset

dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")
dataset = dataset['validation'].select(range(1000))

def send_request(user_id, start_index, end_index):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:12345")

    for index in range(start_index, end_index):
        example = dataset[index]
        question = example["question"]
        socket.send_string(question)

        response = socket.recv_json()
        inference_time = response["inference_time"]
        exact_match = response["exact_match"]
        f1 = response["f1"]

        print(f"User {user_id} - Question: {question}")
        print(f"Inference Time: {inference_time} seconds")
        print(f"Exact Match: {exact_match}")
        print(f"F1 Score: {f1}")
        print()

        time.sleep(1)  # Delay between each question

num_users = 5
examples_per_user = len(dataset) // num_users

threads = []
for i in range(num_users):
    start_index = i * examples_per_user
    end_index = start_index + examples_per_user
    user_thread = threading.Thread(target=send_request, args=(i+1, start_index, end_index))
    threads.append(user_thread)
    user_thread.start()

for thread in threads:
    thread.join()