import os
import random
import pandas as pd
import numpy as np
from datasets import Dataset
from torch.utils.data import Dataset as DS
from transformers import get_linear_schedule_with_warmup, TrainerCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback, generation
import torch as th
import torch.nn as nn
import math
import pickle
from tqdm import tqdm
import nltk
import gc
import json
from evaluation import quadratic_weighted_kappa
import warnings



# 모든 경고 숨기기
warnings.filterwarnings("ignore")
trait_map = {
    1: ["overall", "content", "organization", "word choice", "sentence fluency", "conventions"],
    2: ["overall", "content", "organization", "word choice", "sentence fluency", "conventions"],
    3: ["overall", "content", "prompt adherence", "language", "narrativity"],
    4: ["overall", "content", "prompt adherence", "language", "narrativity"],
    5: ["overall", "content", "prompt adherence", "language", "narrativity"],
    6: ["overall", "content", "prompt adherence", "language", "narrativity"],
    7: ["overall", "content", "organization", "style", "conventions"],
    8: ["overall", "content", "organization", "voice", "word choice", "sentence fluency", "conventions"]
    }
prompt_map ={
1: {
    "overall": "domain1_score",
    "content": "trait1_score",
    "organization": "trait2_score",
    "word choice": "trait3_score",
    "sentence fluency": "trait4_score",
    "conventions": "trait5_score",
},
2: {
    "overall": "domain1_score",
    "content": "trait1_score",
    "organization": "trait2_score",
    "word choice": "trait3_score",
    "sentence fluency": "trait4_score",
    "conventions": "trait5_score",
},
3: {
    "overall": "domain1_score",
    "content": "trait1_score",
    "prompt adherence": "trait2_score",
    "language": "trait3_score",
    "narrativity": "trait4_score"
},
4: {
    "overall": "domain1_score",
    "content": "trait1_score",
    "prompt adherence": "trait2_score",
    "language": "trait3_score",
    "narrativity": "trait4_score"
},
5: {
    "overall": "domain1_score",
    "content": "trait1_score",
    "prompt adherence": "trait2_score",
    "language": "trait3_score",
    "narrativity": "trait4_score"
},
6 : {
    "overall": "domain1_score",
    "content": "trait1_score",
    "prompt adherence": "trait2_score",
    "language": "trait3_score",
    "narrativity": "trait4_score"
},
7 : {
    "overall": "domain1_score",
    "content": "trait1_score",
    "organization": "trait2_score",
    "style": "trait3_score",
    "conventions": "trait4_score"
},
8 : {
    "overall": "domain1_score",
    "content": "trait1_score",
    "organization": "trait2_score",
    "voice": "trait3_score",
    "word choice": "trait4_score",
    "sentence fluency": "trait5_score",
    "conventions": "trait6_score"
}}


import re

import re
from collections import OrderedDict

def parse_traits(text, exclude_keys=[]):
    text = re.sub(r'(\[\w+(?:\s\w+)?\])', r'\n\1', text)

    parts = text.split('\n')

    
    traits_dict = OrderedDict()
    current_trait = None
    for part in parts:
        match = re.match(r'\[(\w+(?:\s\w+)?)\]', part)
        if match:
            current_trait = match.group(1).strip()
            traits_dict[current_trait] =  part.strip() + " "

        elif current_trait:
            traits_dict[current_trait] += part.strip() + " "
    
    for trait in traits_dict:
        traits_dict[trait] = traits_dict[trait].strip()
    

    filtered_dict = OrderedDict({key: value for key, value in traits_dict.items() if key.lower() not in exclude_keys})

    
    return " ".join(filtered_dict.values())

def reversed_parse_traits(text, exclude_keys=[]):
    text = re.sub(r'(\[\w+(?:\s\w+)?\])', r'\n\1', text)

    parts = text.split('\n')

    
    traits_dict = OrderedDict()
    current_trait = None
    for part in parts:
        match = re.match(r'\[(\w+(?:\s\w+)?)\]', part)
        if match:
            current_trait = match.group(1).strip()
            traits_dict[current_trait] =  part.strip() + " "

        elif current_trait:
            traits_dict[current_trait] += part.strip() + " "
    
    for trait in traits_dict:
        traits_dict[trait] = traits_dict[trait].strip()
    

    filtered_dict = OrderedDict({key: value for key, value in reversed(traits_dict.items())})

    
    return " ".join(filtered_dict.values())


def transform_input(input_str):
    return ", ".join([f"[{match.group(1)}] {match.group(2)}" 
                      for match in re.finditer(r"(\w+ \w+|\w+) (\w+)", input_str)])

def preprocess_data(examples, tokenizer,args):
    essay = tokenizer([ "<essay> "+ example for example in examples["t5_input"]], max_length=512, truncation=True, padding="max_length")
    if args.criteria :
        if args.remove:
            criteria = tokenizer([ " <rationale> "+ parse_traits(example,args.exclued_trait) for example in examples["scoring_criteria2"]], max_length=512, truncation=True, padding="max_length")
        else:
            criteria = tokenizer([ " <rationale> "+ example for example in examples["scoring_criteria2"]], max_length=512, truncation=True, padding="max_length")

        essay["input_ids"] = [sublist1 + sublist2 for sublist1, sublist2 in zip(essay["input_ids"],criteria["input_ids"])]
        essay["attention_mask"] = [sublist1 + sublist2 for sublist1, sublist2 in zip(essay["attention_mask"],criteria["attention_mask"])]

    with tokenizer.as_target_tokenizer():
        labels = examples["t5_output"]
        if args.reverse :
            labels = [', '.join([item for item in data_str.split(', ')][::-1]) for data_str in labels]
        #labels = [transform_input(input_str) for input_str in examples["t5_output"]]
        labels = tokenizer(labels, max_length=256, truncation=True, padding="max_length")
        

    essay["labels"] = labels["input_ids"]
    
    return essay

def read_data(data_path):
    prompt_list = list()
    essay_list = list()
    label_list = list()
    
    df = pd.read_csv(data_path)
    
    dataset = Dataset.from_pandas(df)
    
    return dataset
import re


def read_test(test_path):
    test_df = pd.read_csv(test_path)
    essays = test_df["t5_input"].tolist()
    criteria = test_df["scoring_criteria"].tolist()
    labels = test_df["t5_output"].tolist()
    labels = [transform_input(input_str) for input_str in labels]
    print(labels)
    prompt = test_df["prompt"].tolist()
    return essays, criteria, labels, prompt




def set_seed(args):
    """
    Ensure reproducibility by setting the seed for random number generation.
    """
    np.random.seed(args.seed)
    random.seed(args.seed)
    if th.cuda.is_available():
        th.manual_seed(args.seed)
        th.cuda.manual_seed(args.seed)
        th.cuda.manual_seed_all(args.seed)  # if use multi-GPU
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False


def train(model, tokenizer, train_dataset, dev_dataset, args=None):

    training_args = Seq2SeqTrainingArguments(
                        output_dir=f"./results_{args.result_path}",           # 출력 디렉토리
                        evaluation_strategy="steps",      # 평가 전략
                        eval_steps=5000,                 # 평가 스텝
                        per_device_train_batch_size=args.train_batch_size,    # 배치 크기
                        per_device_eval_batch_size=args.train_batch_size,     # 평가 배치 크기
                        num_train_epochs=args.train_epochs,             # 총 에포크
                        predict_with_generate=True,       # 생성을 통한 예측 활성화
                        load_best_model_at_end=True,      # 최고 모델 로드
                        metric_for_best_model="loss",     # 모델 평가 기준
                        greater_is_better=False,          # 낮은 loss가 더 좋은 결과
                        save_steps=5000,                  # 저장 스텝
                        save_total_limit=15          # 최대 저장 파일 수
                    )
        
    trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=dev_dataset,
                tokenizer=tokenizer,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience), SaveTopModelsCallback(args.save_model_fold_path)]
            )

    trainer.train()
    
    return model
                
from torch.nn import functional as F
def test(tokenizer, model, test_data, args):

    pred_dic = dict()
    true_dic = dict()
    qwk_result = dict()
    trait_map = {
    1: ["overall", "content", "organization", "word choice", "sentence fluency", "conventions"],
    2: ["overall", "content", "organization", "word choice", "sentence fluency", "conventions"],
    3: ["overall", "content", "prompt adherence", "language", "narrativity"],
    4: ["overall", "content", "prompt adherence", "language", "narrativity"],
    5: ["overall", "content", "prompt adherence", "language", "narrativity"],
    6: ["overall", "content", "prompt adherence", "language", "narrativity"],
    7: ["overall", "content", "organization", "style", "conventions"],
    8: ["overall", "content", "organization", "voice", "word choice", "sentence fluency", "conventions"]
    }
    compound_keys = {
    'sentence fluency': 'sentence-fluency',
    'word choice': 'word-choice',
    'prompt adherence': 'prompt-adherence'
    }
    for p in range(1,9):
        pred_dic[p] = dict()
        true_dic[p] = dict()
        qwk_result[p] = dict()
        trait_list = trait_map[p]
        for trait in trait_list:
            pred_dic[p][trait] = list()
            true_dic[p][trait] = list()
            qwk_result[p][trait] = 0.0


    model.eval()
    batch_size = 128
    with th.no_grad():
        for i in tqdm(range(0, len(test_data), batch_size)):
            test = test_data[i:i+batch_size]
            input_ids_all  = th.tensor(test['input_ids']).to(args.device)
            attention_mask =  th.tensor(test['attention_mask']).to(args.device)

            input_ids = input_ids_all[:,:512]
            attention_mask = th.ones(input_ids.size(), dtype=th.long).to(model.device)

            if 'bart' in args.model_name:
                encoder_outputs = model.model.encoder(input_ids=input_ids,attention_mask=attention_mask)
            elif 't5' in args.model_name:
                encoder_outputs = model.encoder(input_ids=input_ids,attention_mask=attention_mask)
                

            if args.criteria :
                criteria_ids = input_ids_all[:,512:]
                criteria_attention_mask = th.ones(criteria_ids.size(), dtype=th.long).to(model.device)
                criteria_encoder_outputs = model.encoder(input_ids=criteria_ids,attention_mask=criteria_attention_mask )
                if args.aggreagate == 'linear':
                    encoder_outputs.last_hidden_state = model.t5_proj(th.concat([encoder_outputs[0],criteria_encoder_outputs[0]],dim=1).permute(0,2,1)).permute(0,2,1) 
                elif args.aggreagate == 'conv':   
                    encoder_outputs.last_hidden_state = model.conv1d(th.concat([encoder_outputs[0],criteria_encoder_outputs[0]],dim=1).permute(0,2,1)).permute(0,2,1)
          
            else: 
                encoder_outputs.last_hidden_state = encoder_outputs[0]

            labels = test['t5_output']
            prompts = test["essay_set"]

            decoder_start_token_id = model.config.decoder_start_token_id
            input_ids = th.tensor([[decoder_start_token_id] for _ in range(encoder_outputs[0].size(0))]).to(args.device)

            outputs = model.generate(input_ids=input_ids,encoder_outputs=encoder_outputs,max_new_tokens = 256, num_beams =1)

            for i, (output, true) in enumerate(zip(outputs, labels)):
                pred = tokenizer.decode(output, skip_special_tokens=True)
                print(pred)
                try:
                    for key, replacement in compound_keys.items():
                        pred = pred.replace(key, replacement)

                    items = pred.split(', ')

                    pred_result = {}
                    for item in items:
                        key, value = item.split(' ', 1)
                        #key = key.replace('-', ' ')[1:-1] 
                        key = key.replace('-', ' ') 
                        if value == 'nan':
                            value = np.nan
                        else:
                            value = int(value)
                        pred_result[key] = value
                
                    true_text = true
                    for key, replacement in compound_keys.items():
                        true_text = true_text.replace(key, replacement)
                    items = true_text.split(', ')
                    true_result = {}
                    for item in items:
                        key, value = item.split(' ', 1)
                        #key = key.replace('-', ' ')[1:-1]
                        key = key.replace('-', ' ')  
                        if value == 'nan':
                            value = np.nan
                        else:
                            value = int(value)
                            true_result[key] = value

                    prompt = prompts[i]
                
                    trait_list = trait_map[prompt]

                    for trait in trait_list:
                        if np.isnan(pred_result[trait]):
                            continue
                        pred_dic[prompt][trait].append(pred_result[trait])
                        true_dic[prompt][trait].append(true_result[trait])
                except:
                    #print("skipped")
                    continue
        for prompt in range(1,9):
            trait_list = trait_map[prompt]
            
            for trait in trait_list:
                qwk_result[prompt][trait] = quadratic_weighted_kappa(np.array(pred_dic[prompt][trait]), np.array(true_dic[prompt][trait]))
                                           
        log = "Test Result"
        for prompt in range(1,9):
            log += f"\n\n| Prompt: {prompt} |"
            log += f"\n| {qwk_result[prompt]} |"
        print(log)

        

    return qwk_result, pred_dic, true_dic

class TestDataset(DS):
    def __init__(self, text_list, criteria_list , prompt_list, label_list):
        self.text_list = text_list
        self.label_list = label_list
        self.prompt_list = prompt_list
        self.criteria_list = criteria_list
    
    def __len__(self):
        return len(self.text_list)
    
    def __getitem__(self, idx):
        return self.text_list[idx],self.criteria_list[idx], self.prompt_list[idx], self.label_list[idx]

    def collate_fn(self, batch):
        texts = [text for text,criteria, prompt, label in batch]
        labels = [(prompt, label) for text, prompt, label in batch]
        
        return texts, labels




def deep_copy_state_dict(state_dict):
    copy_dict = {}
    for key, value in state_dict.items():
        copy_dict[key] = value.clone()
    return copy_dict

class SaveTopModelsCallback(TrainerCallback):
    """ 트레이닝 중 Eval step에서 가장 낮은 loss를 보인 top 2 모델만 저장하는 콜백 """
    def __init__(self, save_path, top_k=2):
        self.save_path = save_path
        self.top_k = top_k
        self.top_models = []  # (loss, step, model_state_dict)를 저장할 리스트

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        current_loss = metrics['eval_loss']
        current_step = state.global_step
        kwargs["model"] = kwargs["model"].cpu()
        model_state_dict = deep_copy_state_dict(kwargs['model'].state_dict())  # 가중치만 복사하여 저장
        kwargs["model"] = kwargs["model"].to(args.device)

        # Top 모델 리스트 업데이트
        self.top_models.append((current_loss, current_step, model_state_dict))
        self.top_models.sort(key=lambda x: x[0])  # loss 기준 정렬
        self.top_models = self.top_models[:self.top_k]  # top k 모델 유지

        # 오래된 체크포인트 삭제 및 최신 top k 모델 저장
        self.cleanup_and_save_top_models()

    def cleanup_and_save_top_models(self):
        # 모든 체크포인트 삭제
        for filename in os.listdir(self.save_path):
            if filename.startswith("checkpoint"):
                os.remove(os.path.join(self.save_path, filename))
        
        # 현재 Top k 모델 저장
        for rank, (loss, step, state_dict) in enumerate(self.top_models):
            model_path = os.path.join(self.save_path, f"checkpoint-{rank+1}-loss-{loss:.4f}")
            th.save(state_dict, model_path)
            print(f"Saved top {rank+1} model to {model_path} with loss {loss:.4f}")
