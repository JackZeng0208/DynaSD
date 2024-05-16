import zmq
import threading
import time
from datasets import load_dataset
import csv
import tqdm

dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")
dataset = dataset['validation']
dataset = dataset.filter(lambda example: len(example["question"]) <= 128)
dataset = dataset.select(range(2100))
total_start_time = 0
total_end_time = 0
SERVER_IP = "192.168.0.132"
PORT = 1919

def send_request(user_id, start_index, end_index):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{SERVER_IP}:{PORT}")
    batch_size = 4
    with open(f"benchmark_user_{user_id}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Inference Time", "Transmission Time", "Total Time", "Inference Speed"])
        for index in tqdm.tqdm(range(start_index, end_index, batch_size)):
            example = dataset[index:index+batch_size]
            questions = example["question"]
            print(questions)
            start_time = time.time()
            socket.send_pyobj({"questions": questions, "user_id": user_id, "end": False})
            response = socket.recv_pyobj()
            end_time = time.time()
            total_time = end_time - start_time

            inference_time = response["inference_time"]
            transmission_time = total_time - inference_time
            output = response["output"]
            inference_speed = output/total_time
            print(f"User {user_id} - Inference Speed: {inference_speed}")
            writer.writerow([inference_time, transmission_time, total_time, inference_speed])
        socket.send_pyobj({"user_id": user_id, "end": True})
        socket.close()
        exit()
        
num_users = 1
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