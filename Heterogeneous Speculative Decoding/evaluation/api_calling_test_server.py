import zmq
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import torch
import pynvml
import threading
import csv
pynvml.nvmlInit()

def evaluate(model, tokenizer, socket: zmq.Socket):
    device = torch.device("cuda:0")
    handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)
    with open(f"gpu_utilization_api_calling_short_test.txt", mode='w', newline='') as file:
        gpu_utilization = []
        def capture_gpu_utilization(stop_event):
            # Adjust the sample interval as needed (in seconds) -> 1ms
            sample_interval = 1
            while not stop_event.is_set():
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                # print(f"GPU utilization: {utilization}")
                gpu_utilization.append(utilization)
                time.sleep(sample_interval)
        init_flag = True
        while True:
            message = socket.recv_pyobj()
            if init_flag:
                # Start capturing GPU utilization in a separate thread
                stop_event = threading.Event()
                gpu_thread = threading.Thread(target=capture_gpu_utilization, args=(stop_event,))
                gpu_thread.start()
                init_flag = False

            end_flag = message['end']
            
            if end_flag:
                # Stop capturing GPU utilization
                stop_event.set()
                gpu_thread.join()
                file.write(str(gpu_utilization))
                socket.send_pyobj('End')
                socket.close()
                exit()

            user_id = message['user_id']
            question = message['question']
            
            start_time = time.time()
            input_str = f"Question: {question}\nAnswer:"
            input_ids = tokenizer.encode(input_str, return_tensors="pt").to("cuda:0")
            output = model.generate(input_ids, max_new_tokens=128)
            # print(f"GPU utilization: {gpu_utilization}")
            pred_answer = tokenizer.decode(output[0], skip_special_tokens=True)
            end_time = time.time()
            inference_time = end_time - start_time
            result = {
                "output": pred_answer,
                "inference_time": inference_time,
            }
            socket.send_pyobj(result)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:1919")
print("Server is running...")

evaluate(model, tokenizer, socket)
