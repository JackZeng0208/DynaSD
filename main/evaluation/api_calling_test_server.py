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

    with open(f"gpu_power_usage_api_calling.txt", mode='w', newline='') as file:
        gpu_power_usage = []

        def capture_gpu_power_usage(stop_event):
            # Adjust the sample interval as needed (in seconds) -> 1ms
            sample_interval = 1
            while not stop_event.is_set():
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert from milliwatts to watts
                # print(f"GPU power usage: {power_usage} W")
                gpu_power_usage.append(power_usage)
                time.sleep(sample_interval)

        init_flag = True
        torch.backends.cudnn.benchmark = True
        while True:
            message = socket.recv_pyobj()
            if init_flag:
                # Start capturing GPU power usage in a separate thread
                stop_event = threading.Event()
                gpu_thread = threading.Thread(target=capture_gpu_power_usage, args=(stop_event,))
                gpu_thread.start()
                init_flag = False

            end_flag = message['end']
            if end_flag:
                # Stop capturing GPU power usage
                stop_event.set()
                gpu_thread.join()
                file.write(str(gpu_power_usage))
                socket.send_pyobj('End')
                socket.close()
                exit()

            user_id = message['user_id']
            questions = message['questions']

            start_time = time.time()

            # input_str = f"Answer multiple questions: {question}\\nAnswer:"
            torch.cuda.synchronize()
            input_ids = tokenizer(questions, return_tensors="pt", padding=True).to("cuda:0")
            outputs = model.generate(**input_ids, max_new_tokens=128, num_return_sequences=1)
            torch.cuda.synchronize()
            # print(outputs.shape)
            print(f"GPU power usage: {gpu_power_usage[-1]}")

            pred_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # print(pred_answers)

            end_time = time.time()
            inference_time = end_time - start_time
            num_generated_tokens = len(outputs[0]) - len(input_ids[0])
            # print(f"Inference time: {inference_time}")
            result = {
                "output": num_generated_tokens,
                "inference_time": inference_time,
            }

            socket.send_pyobj(result)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.bfloat16).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:1919")
print("Server is running...")

evaluate(model, tokenizer, socket)