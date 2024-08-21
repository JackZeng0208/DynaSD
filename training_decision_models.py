import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tqdm import tqdm
import torch.optim as optim
import json
from model.llama_tree_attn import LlamaForCausalLM
from inference.fork_shape_tree_attn import NewTreeStrategy
import matplotlib.pyplot as plt
from DynaSD.decision_models import *
from torch.utils.data import Dataset,DataLoader
import random
import seaborn as sns
import os
SOFT_LABEL = True
# DRAFT_MODEL_NAME = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
DRAFT_MODEL_NAME = 'JackFram/llama-68m'
TARGET_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
# TARGET_MODEL_NAME = "lmsys/vicuna-7b-v1.5"

if DRAFT_MODEL_NAME == 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' and TARGET_MODEL_NAME == "lmsys/vicuna-7b-v1.5":
    DECISION_MODEL_NAME = "tinyllama_vicuna_7b"
    if SOFT_LABEL == True:
        decision_model = DecisionModelV1_Tinyllama().cuda()
    else:
        decision_model = DecisionModelVTopk().cuda()

if DRAFT_MODEL_NAME == 'JackFram/llama-68m' and TARGET_MODEL_NAME == "lmsys/vicuna-7b-v1.5":
    DECISION_MODEL_NAME = "llama_68m_vicuna_7b"
    if SOFT_LABEL == True:
        decision_model = DecisionModelV1().cuda()
    else:
        decision_model = DecisionModelVTopk().cuda()

if DRAFT_MODEL_NAME == 'JackFram/llama-68m' and TARGET_MODEL_NAME == "meta-llama/Llama-2-7b-chat-hf":
    DECISION_MODEL_NAME = "llama_68m_llama2_7b"
    if SOFT_LABEL == True:
        decision_model = DecisionModelV1().cuda()
    else:
        decision_model = DecisionModelVTopk().cuda()

if DRAFT_MODEL_NAME == 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' and TARGET_MODEL_NAME == "meta-llama/Llama-2-7b-chat-hf":
    DECISION_MODEL_NAME = "tinyllama_llama2_7b"
    if SOFT_LABEL == True:
        decision_model = DecisionModelV1_Tinyllama().cuda()
    else:
        decision_model = DecisionModelVTopk().cuda()

class BalancedDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.load(data)
        self.labels = torch.load(labels)
        self.labels = torch.tensor(self.labels)
        # Separate positive and negative samples
        pos_indices = (self.labels == 1).nonzero(as_tuple=True)[0]
        neg_indices = (self.labels == 0).nonzero(as_tuple=True)[0]
        
        # Sample negative examples to match positive examples
        print(f" length of positive indices {len(pos_indices)}, and len of negative indices {len(neg_indices)}")
        if len(pos_indices)>len(neg_indices):
            # using neg as bottom line 
            num_neg = len(neg_indices)
            sample_pos_indices = random.sample(pos_indices.tolist(),num_neg)
            self.balanced_indices = torch.cat([neg_indices, torch.tensor(sample_pos_indices)])
        else:
            # using pos as bottom line 

            num_pos = len(pos_indices)
            sampled_neg_indices = random.sample(neg_indices.tolist(), num_pos)
            self.balanced_indices = torch.cat([pos_indices, torch.tensor(sampled_neg_indices)])
        
        self.balanced_indices = self.balanced_indices[torch.randperm(len(self.balanced_indices))]
        
        # Combine indices
        
        # Shuffle the indices

    def __len__(self):
        return len(self.balanced_indices)

    def __getitem__(self, idx):
        true_idx = self.balanced_indices[idx]
        return self.data[true_idx], self.labels[true_idx]
    
class SimpleDataset(Dataset):
    def __init__(self, data_file, label_file):
        self.data = torch.load(data_file)
        self.labels = torch.load(label_file)
        # print(f"what is the self.label {self.labels}")
        # self.labels = torch.tensor(self.labels)
        # Separate positive and negative samples
        # pos_indices = (self.labels == 1).nonzero(as_tuple=True)[0]
        # neg_indices = (self.labels == 0).nonzero(as_tuple=True)[0]
        
        # # Sample negative examples to match positive examples
        # print(f" length of positive indices {len(pos_indices)}")
        # num_pos = len(pos_indices)
        # sampled_neg_indices = random.sample(neg_indices.tolist(), num_pos)
        
        # # Combine indices
        # self.balanced_indices = torch.cat([pos_indices, torch.tensor(sampled_neg_indices)])
        
        # # Shuffle the indices
        # self.balanced_indices = self.balanced_indices[torch.randperm(len(self.balanced_indices))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # real_idx = self.balanced_indices[idx]
        return self.data[idx], self.labels[idx]

def load_dataset(filename):
    dataset = []
    with open(filename, 'r') as f:
        data = json.load(f)
        for item in data:
            dataset.append(item)
    return dataset

draft_model = LlamaForCausalLM.from_pretrained(
    DRAFT_MODEL_NAME,
    torch_dtype=torch.float32,
    device_map=0,
)

target_model = LlamaForCausalLM.from_pretrained(
    TARGET_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map= 0,
)
draft_model.eval()
target_model.eval()

class TrainDecisionModel:
    def __init__(self,
        strategy,
        training_data_file_name = '/home/iasl-transformers/DynaSD/decision_model/training_datasets/combined_training_questions.json',
        training_epoch = 10,
        learning_rate = 0.001,
        batch_size = 20, # the batch size should be carefully chosen for easy code implementation since no torch.dataset is using
        decision_model_save_path = '/home/iasl-transformers/DynaSD/decision_model/',
                 ) -> None:
        self.decision_model = decision_model
        self.loss_file = 'high_quality_loss_v_topk.png'
        self.weight_save_path = decision_model_save_path+f'high_quality_{SOFT_LABEL}_{DECISION_MODEL_NAME}.pt'

        
        self.strategy = strategy
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.decision_model.parameters(),lr = learning_rate)
        lambda1 = lambda epoch: 0.5 ** epoch
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
        self.batch_size = batch_size
        self.epoch = training_epoch

        # datasets
        self.dataset = load_dataset(training_data_file_name)[:10000]
        # self.dataset = random.sample(self.dataset, 10000)
        print(self.dataset[0])
        self.train_val_split = int(len(self.dataset)*0.9) # 90% training, 10% val
        self.new_dataset_base = '/home/iasl-transformers/DynaSD/decision_model/high_quality_soft/'
        self.hard_label_path_base = '/home/iasl-transformers/DynaSD/decision_model/high_quality_hard_labels/'
        self.soft_train_x_path = self.new_dataset_base + f'soft_train_x_{DECISION_MODEL_NAME}.pt'
        self.soft_train_y_path = self.new_dataset_base + f'soft_train_y_{DECISION_MODEL_NAME}.pt'
        self.soft_val_x_path = self.new_dataset_base + f'soft_val_x_{DECISION_MODEL_NAME}.pt'
        self.soft_val_y_path = self.new_dataset_base + f'soft_val_y_{DECISION_MODEL_NAME}.pt'

        self.hard_train_x_path = self.hard_label_path_base + f'hard_train_x_{DECISION_MODEL_NAME}.pt'
        self.hard_train_y_path = self.hard_label_path_base + f'hard_train_y_{DECISION_MODEL_NAME}.pt'
        self.hard_val_x_path =   self.hard_label_path_base + f'hard_val_x_{DECISION_MODEL_NAME}.pt'
        self.hard_val_y_path =   self.hard_label_path_base + f'hard_val_y_{DECISION_MODEL_NAME}.pt'

        #training Stats
        self.training_avg_loss_epoch = []
        self.verification_loss_epoch = []
        self.best_verification_loss = float('inf')
        self.pos_y_pred_probs = []
        self.neg_y_pred_probs = []
        self.pos_y_pred_file = '/home/iasl-transformers/DynaSD/decision_model/pos_ypred_list.pkl'
        self.neg_y_pred_file = '/home/iasl-transformers/DynaSD/decision_model/neg_ypred_list.pkl'
        # if not os.path.exists(self.pos_y_pred_file):
        #     os.makedirs(self.pos_y_pred_file)
        # if not os.path.exists(self.neg_y_pred_file):
        #     os.makedirs(self.neg_y_pred_file)

        # making decision with entropy threshold: 
        self.postive_label_entropy_file = '/home/iasl-transformers/DynaSD/decision_model/entropy_decision/postive_entropy.pkl'
        self.negative_label_entropy_file = '/home/iasl-transformers/DynaSD/decision_model/entropy_decision/negative_entropy.pkl'

        # if not os.path.exists(self.postive_label_entropy_file):
        #     os.makedirs(self.postive_label_entropy_file)
        # if not os.path.exists(self.negative_label_entropy_file):
        #     os.makedirs(self.negative_label_entropy_file)
        # with open(self.postive_label_entropy_file,'wb') as pos_file:
        #     pickle.dump([1,2,3,4],pos_file)
        # with open(self.negative_label_entropy_file,'wb') as neg_file:
        #     pickle.dump([1,2,3,4],neg_file)

    def append_to_tensor_dataset(self,new_tensors,is_label = False,is_val= False):
        # training option 2: save the generated training data, 
        if SOFT_LABEL == True:
            if not is_val:
                path = self.soft_train_y_path if is_label else self.soft_train_x_path 
            else:
                path = self.soft_val_y_path if is_label else self.soft_val_x_path
        else:
            if not is_val:
                path = self.hard_train_y_path if is_label else self.hard_train_x_path 
            else:
                path = self.hard_val_y_path if is_label else self.hard_val_x_path 

        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                existing_dataset = torch.load(path)
            except EOFError:
                print(f"File {path} is corrupted or empty. Initializing a new dataset.")
                existing_dataset = []
        else:
            print(f"File {path} not found or is empty. Creating a new dataset.")
            existing_dataset = []

        if not isinstance(existing_dataset, list):
            existing_dataset = [existing_dataset]

        existing_dataset.extend(new_tensors)

        torch.save(existing_dataset, path)
    
    def generate_data(self,accept_training_example = 1):
        #due to 1 and 0 are not evenly distributed, prioritize collect accept_training_example
        # amount of training example 
        num_1 = 0
        train_x,train_y = [],[]
        for i,s in enumerate(tqdm(self.dataset[:self.train_val_split])):
            input_ids = tokenizer(s, return_tensors='pt').input_ids.to('cuda')
            output, x, y= self.strategy.generation_loop(input_ids)
            print(f"shape of x should be 11 {x.shape}")
            print(f"what is y {y}")
            # print(f"current y has negtive  {(y < 0).any().item()}")
            num_1 += x.shape[0]-torch.sum(y)
            x = x.detach().cpu()
            y = y.detach().cpu()
            train_x.extend(x)
            train_y.extend(y)
            # if num_1 >= accept_training_example:
            #     break
            if i > accept_training_example:
                break
            # self.append_to_tensor_dataset(train_x)
            # self.append_to_tensor_dataset(train_y,is_label=True)
        self.append_to_tensor_dataset(train_x)
        self.append_to_tensor_dataset(train_y,is_label=True)
        
        val_x,val_y = [],[]
        for vs in tqdm(self.dataset[self.train_val_split:self.train_val_split +5]):
            input_ids = tokenizer(vs, return_tensors='pt').input_ids.to('cuda')
            output, x, y= self.strategy.generation_loop(input_ids)
            x = x.detach().cpu()
            y = y.detach().cpu()
            val_x.extend(x)
            val_y.extend(y)
        self.append_to_tensor_dataset(val_y,is_label=True,is_val=True)
        self.append_to_tensor_dataset(val_x,is_label=False,is_val=True)
        
    def only_eval(self):
        self.decision_model.load_state_dict(torch.load(self.weight_save_path))
        if SOFT_LABEL == True:
            val_dataset = SimpleDataset(self.soft_val_x_path,self.soft_val_y_path)
            val_dataloader = DataLoader(val_dataset,batch_size=50,shuffle=True)
        else:
            val_dataset = SimpleDataset(self.hard_val_x_path,self.hard_val_y_path)
            val_dataloader = DataLoader(val_dataset,batch_size=50,shuffle=True)
        
        val_loss = 0
        val_round = 0
        pbar = tqdm(val_dataloader)
        for x,y in pbar:
            val_loss += self.eval_model(x,y)
            val_round +=1
        current_val_loss = val_loss/val_round
        print(f"current val loss is {current_val_loss}")
        self.plot_prediction_distribution()

    
    def train_with_dataset(self):
        if SOFT_LABEL == True:
            train_dataset = SimpleDataset(self.soft_train_x_path,self.soft_train_y_path)
            train_dataloader = DataLoader(train_dataset,batch_size=50,shuffle=True)
            val_dataset = SimpleDataset(self.soft_val_x_path,self.soft_val_y_path)
            val_dataloader = DataLoader(val_dataset,batch_size=50,shuffle=True)
        else:
            train_dataset = BalancedDataset(self.hard_train_x_path,self.hard_train_y_path)
            train_dataloader = DataLoader(train_dataset,batch_size=50,shuffle=True)
            val_dataset = BalancedDataset(self.hard_val_x_path,self.hard_val_y_path)
            val_dataloader = DataLoader(val_dataset,batch_size=50,shuffle=True)

        for e in tqdm(range(self.epoch)):
            self.pos_y_pred_probs = []
            self.neg_y_pred_probs = []

            running_loss = 0
            num_round = 0
            pbar = tqdm(train_dataloader,desc=f"epoch: {e} train")
            for x,y in pbar:
                running_loss += self.train_single_batch(x,y)
                num_round +=1
                if num_round % 5000 ==0:
                    print(f"current avg training loss is {running_loss/num_round}")

            
            val_loss = 0
            val_round = 0
            pbar = tqdm(val_dataloader,desc=f"epoch:{e} validation")
            for x,y in pbar:
                val_loss += self.eval_model(x,y)
                val_round +=1
            current_val_loss = val_loss/val_round
            print(f"average val loss is {current_val_loss}")
            self.training_avg_loss_epoch.append(running_loss/num_round)
            self.verification_loss_epoch.append(current_val_loss)
            if self.best_verification_loss > current_val_loss:
                print(f"saving epoch {e} weight")
                self.best_verification_loss = current_val_loss
                torch.save(self.decision_model.state_dict(),self.weight_save_path)
            # self.scheduler.step()
        

        # with open(self.pos_y_pred_file,'wb') as pos_file:
        #     pickle.dump(self.pos_y_pred_probs,pos_file)
        # with open(self.neg_y_pred_file,'wb') as neg_file:
        #     pickle.dump(self.neg_y_pred_probs,neg_file)
        self.plot_loss()
        # self.plot_prediction_distribution()

    
    def training_in_epoch(self):
        for _ in tqdm(range(self.epoch)):
            running_loss = 0
            num_round = 0
            for i,s in enumerate(tqdm(self.dataset[:self.train_val_split])):
                input_ids = tokenizer(s, return_tensors='pt').input_ids.to('cuda')
                output, x, y= self.strategy.generation_loop(input_ids)
                print(f"what is x looks like {x}")
                print(f"y distributions len is {y.shape[0]},  number of 1 is {torch.sum(y)}")
                running_loss += self.slide_window_training_data(x,y)
                if i % 100 == 99:
                    print(f"current avg training loss is {running_loss/num_round}")
                num_round = i +1
            
            val_loss = 0
            val_round = 0
            for vs in (self.dataset[self.train_val_split:]):
                input_ids = tokenizer(s, return_tensors='pt').input_ids.to('cuda')
                output, x, y= self.strategy.generation_loop(input_ids)
                val_loss += self.slide_window_training_data(x,y,is_train=False) 
                val_round += 1
            current_val_loss = val_loss/val_round
            print(f"average val loss is {current_val_loss}")
            self.training_avg_loss_epoch.append(running_loss/num_round)
            self.verification_loss_epoch.append(current_val_loss)
            if self.best_verification_loss > current_val_loss:
                self.best_verification_loss = current_val_loss
                torch.save(self.decision_model.state_dict(),self.weight_save_path)
        self.plot_loss()
    
    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.epoch + 1), self.training_avg_loss_epoch, label='Training Loss')
        plt.plot(range(1, self.epoch + 1), self.verification_loss_epoch, label='Evaluation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Evaluation Losses')
        plt.legend()
        plt.savefig(self.loss_file)
        plt.show()


    def plot_prediction_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.pos_y_pred_probs, kde=True, color='blue', label='Positive (y=1)', alpha=0.6)
        sns.histplot(self.neg_y_pred_probs, kde=True, color='red', label='Negative (y=0)', alpha=0.6)
        plt.title('Distribution of Model Predictions')
        plt.xlabel('Predicted Value')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig('y_pred_dist.png')
        plt.show()
        

    def slide_window_training_data(self,mb_x, mb_y,is_train = True):
        # sliding window slide through the training data
        # assume the length of x is divisble by batch size 
        slider = 0
        total_loss = 0
        for _ in range(int(mb_x.shape[0]/self.batch_size)):
            sb_x = mb_x[slider:slider+self.batch_size]
            sb_y = mb_y[slider:slider+self.batch_size]
            if is_train:
                total_loss +=self.train_single_batch(sb_x,sb_y)
            else:
                total_loss += self.eval_model(sb_x,sb_y)
            slider += self.batch_size
        return total_loss/(mb_x.shape[0]/self.batch_size)

    def eval_model(self,x,y):
        self.decision_model.eval()
        x = x.to(torch.float).cuda()
        y = y.to(torch.float).cuda().view(-1,1)
        y_pred = self.decision_model(x)
        loss = self.criterion(y_pred,y)
        self.pos_y_pred_probs.extend(y_pred[y >0.5].detach().cpu().numpy())
        self.neg_y_pred_probs.extend(y_pred[y <= 0.5].detach().cpu().numpy())
        return loss.item()


    def train_single_batch(self,x,y):
        # assume the x and y is in single batch shape 
        self.decision_model.train()
        x = x.to(torch.float).cuda()
        y = y.to(torch.float).cuda().view(-1,1)

        y_pred = self.decision_model(x)
        # print(f"shape of y_pred: {y_pred.shape}, y: {y.shape}")
        # print(f" y_pred: {y_pred}")
        loss = self.criterion(y_pred,y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


if __name__ == '__main__':
    if SOFT_LABEL == True:
        greedy_flag = False
    else:
        greedy_flag = True
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_NAME)
    strategy = NewTreeStrategy(
        draft_model= draft_model,
        target_model= target_model,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens= 200,
        # Set greedy = False if it is soft label decision model
        greedy = greedy_flag,
        using_decision_model=False,
        config_depth= 1,
        config_width=10,
        generate_training_data=True,
        soft_label = SOFT_LABEL
        )
    model_train = TrainDecisionModel(strategy=strategy)
    model_train.generate_data()
    model_train.train_with_dataset()
    model_train.only_eval()
