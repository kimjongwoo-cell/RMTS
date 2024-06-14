import os
import argparse
import torch as th
from utils import *
from modeling_t5 import T5ForConditionalGeneration
from transformers import T5Tokenizer, BartTokenizer
#from modeling_bart import BartForConditionalGeneration
import gc
import pickle
import warnings

# 모든 경고 숨기기
warnings.filterwarnings("ignore")
def main(args):
    
    set_seed(args)
    
    if not os.path.isdir(f"ckpts_{args.result_path}"):
        os.makedirs(f"ckpts_{args.result_path}")

    args.save_model_path = f"ckpts_{args.result_path}" 
    
    if args.test:
        args.load_checkpoint_path = f"ckpts_{args.result_path}"
    

    if th.cuda.is_available() and args.gpu != -1:
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'

    
    if 't5' in args.model_name:
        tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    elif 'bart' in args.model_name:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')     
    #add_tokens = ["@", "{", "}"]
    #add_tokens = ["@", "{", "}",'<essay>',"<rationale>","[overall]", "[content]", "[organization]", "[word choice]", "[sentence fluency]", "[conventions]","[prompt adherence]", "[language]", "[narrativity]", "[style]","[voice]"]
    add_tokens = ["@", "{", "}",'<essay>',"<rationale>"]
    tokenizer.add_tokens(add_tokens)
    

    
    dataset_kwargs = dict(
        max_source_length = args.max_source_length,
        max_target_length = args.max_target_length,
        random_seed = args.seed
    )
    best_fold_result_dict = dict()
    best_fold_pred_dict = dict()
    best_fold_true_dict = dict()
    sub_best_fold_result_dict = dict()
    sub_best_fold_pred_dict = dict()
    sub_best_fold_true_dict = dict()
    
    for fold in range(5):
        if 't5' in args.model_name:
            model = T5ForConditionalGeneration.from_pretrained(args.model_name)
        elif 'bart' in args.model_name:
            model = BartForConditionalGeneration.from_pretrained(args.model_name)
        model.resize_token_embeddings(len(tokenizer))
        save_model_fold_path = os.path.join(args.save_model_path, str(fold))
        if not os.path.isdir(save_model_fold_path):
            os.makedirs(save_model_fold_path)
        args.save_model_fold_path = save_model_fold_path
        
        
        TRAIN_DATA_PATH = f"./data/essay/fold_{fold}/train.csv"
        DEV_DATA_PATH = f"./data/essay/fold_{fold}/dev.csv"
        TEST_DATA_PATH = f"./data/essay/fold_{fold}/test.csv"

        train_data = read_data(TRAIN_DATA_PATH)
        dev_data = read_data(DEV_DATA_PATH)
        test_data = read_data(TEST_DATA_PATH)
        
        train_dataset = train_data.map(lambda x: preprocess_data(x, tokenizer,args), batched=True)
        dev_dataset = dev_data.map(lambda x: preprocess_data(x, tokenizer,args), batched=True)
        test_dataset = test_data.map(lambda x: preprocess_data(x, tokenizer,args), batched=True)
        #print(test_dataset["input_ids"])
        
        
        if not args.test:
            print(f"Model Training Fold : {fold}")
            model = train(model, tokenizer, train_dataset, dev_dataset, args)

            for filename in os.listdir(args.save_model_fold_path):
                if filename.startswith("checkpoint-1"):
                    best_model_path = os.path.join(args.save_model_fold_path, filename)
                    best_checkpoint = th.load(best_model_path)
                    # best_model_path = os.path.join(args.save_model_fold_path)
                    model.load_state_dict(best_checkpoint)
                    best_model = model.to(args.device)
                    # best_model = best_model.to(args.device)
        
                    best_result, best_pred_dic, best_true_dic = test(tokenizer, best_model, test_dataset, args)
                    best_model = best_model.cpu()
        
                    del best_model
                    th.cuda.empty_cache()
                    gc.collect()  # Trigger Python garbage collection
        
                elif filename.startswith("checkpoint-2"):
                    sub_best_model_path = os.path.join(args.save_model_fold_path, filename)
                    sub_best_checkpoint = th.load(sub_best_model_path)
                    # best_model_path = os.path.join(args.save_model_fold_path)
                    model.load_state_dict(sub_best_checkpoint)
                    sub_best_model = model.to(args.device)
                    # best_model = best_model.to(args.device)
                    sub_best_result, sub_best_pred_dic, sub_best_true_dic = test(tokenizer, sub_best_model, test_dataset, args)
                    sub_best_model = sub_best_model.cpu()
                    
                    del sub_best_model
                    th.cuda.empty_cache()
                    gc.collect()  # Trigger Python garbage collection

        elif args.test:
            print(f"Model Test Fold : {fold}")
            for filename in os.listdir(args.save_model_fold_path):
                if filename.startswith("checkpoint-1"):
                    best_model_path = os.path.join(args.save_model_fold_path, filename)
                    best_checkpoint = th.load(best_model_path)
                    # best_model_path = os.path.join(args.save_model_fold_path)
                    model.load_state_dict(best_checkpoint)
                    best_model = model.to(args.device)
                    # best_model = best_model.to(args.device)
        
                    best_result, best_pred_dic, best_true_dic = test(tokenizer, best_model, test_dataset, args)
                    best_model = best_model.cpu()
        
                    del best_model
                    th.cuda.empty_cache()
                    gc.collect()  # Trigger Python garbage collection
        
                elif filename.startswith("checkpoint-2"):
                    sub_best_model_path = os.path.join(args.save_model_fold_path, filename)
                    sub_best_checkpoint = th.load(sub_best_model_path)
                    # best_model_path = os.path.join(args.save_model_fold_path)
                    model.load_state_dict(sub_best_checkpoint)
                    sub_best_model = model.to(args.device)
                    # best_model = best_model.to(args.device)
                    sub_best_result, sub_best_pred_dic, sub_best_true_dic = test(tokenizer, sub_best_model, test_dataset, args)
                    sub_best_model = sub_best_model.cpu()
                    
                    del sub_best_model
                    th.cuda.empty_cache()
                    gc.collect()  # Trigger Python garbage collection

        best_fold_result_dict[fold] = best_result
        best_fold_pred_dict[fold] = best_pred_dic
        best_fold_true_dict[fold] = best_true_dic
        
        
        sub_best_fold_result_dict[fold] = sub_best_result
        sub_best_fold_pred_dict[fold] = sub_best_pred_dic
        sub_best_fold_true_dict[fold] = sub_best_true_dic
        
        with open(f"./results_{args.result_path}/best_result_dict.pkl", "wb") as f:
            pickle.dump(best_fold_result_dict, f)
        with open(f"./results_{args.result_path}/best_pred_dict.pkl", "wb") as f:
            pickle.dump(best_fold_pred_dict, f)
        with open(f"./results_{args.result_path}/best_true_dict.pkl", "wb") as f:
            pickle.dump(best_fold_true_dict, f)
        with open(f"./results_{args.result_path}/sub_best_result_dict.pkl", "wb") as f:
            pickle.dump(sub_best_fold_result_dict, f)
        with open(f"./results_{args.result_path}/sub_best_pred_dict.pkl", "wb") as f:
            pickle.dump(sub_best_fold_pred_dict, f)
        with open(f"./results_{args.result_path}/sub_best_true_dict.pkl", "wb") as f:
            pickle.dump(sub_best_fold_true_dict, f)
        
        
    
    # return best_fold_result_dict, best_fold_pred_dict, best_fold_true_dict
    return best_fold_result_dict, best_fold_pred_dict, best_fold_true_dict, \
        sub_best_fold_result_dict, sub_best_fold_pred_dict, sub_best_fold_true_dict
        



    
    


if __name__ == "__main__":


        parser = argparse.ArgumentParser('Essay Scoring')
        parser.add_argument('--gpu', '-g', type=int, default=0, help='which gpu to use, specify -1 to use CPU')
        parser.add_argument('--train_batch_size', '-trb', type=int, default=4, help='batch_size')
        parser.add_argument('--test_batch_size', '-teb', type=int, default=4, help='test_batch_size')
        parser.add_argument('--seed', '-s', type=int, default=42, help='random seed')
        parser.add_argument('--patience', '-p', type=int, default=10, help='number of patience for early stopping')
        parser.add_argument('--max_source_length', '-sl', type=int, default=512, help='max length of source essay')
        parser.add_argument('--max_target_length', '-tl', type=int, default=256, help='max length of target essay')
        parser.add_argument("--train_epochs", type=int, default=15)
        parser.add_argument("--save_checkpoint_path", type=str, default=None)
        parser.add_argument("--test", type=bool, default=False)
        parser.add_argument("--num_beams", type=int, default=1)

        parser.add_argument("--result_path", type=str, default=f"t5_criteria")
        parser.add_argument("--reverse", type=bool, default=False)
        parser.add_argument("--criteria", type=bool, default=True)
        parser.add_argument("--aggreagate", type=str, default='linear')
        parser.add_argument('--model_name', '-m', type=str, default='t5-base', help='name of the t5 model')
        parser.add_argument('--remove',type=bool, default=False)
        #"overall", "content", "organization", "word choice", "sentence fluency", "conventions", "prompt adherence", "language", "narrativity"
        #parser.add_argument('--exclued_trait',type=str, default=[f'{item}'])
        args = parser.parse_args()

        best_fold_result_dict, best_fold_pred_dict, best_fold_true_dict, \
            sub_best_fold_result_dict, sub_best_fold_pred_dict, sub_best_fold_true_dict = main(args)

        with open(f"./results_{args.result_path}/final_best_result_dict.pkl", "wb") as f:
            pickle.dump(best_fold_result_dict, f)
        with open(f"./results_{args.result_path}/final_best_pred_dict.pkl", "wb") as f:
            pickle.dump(best_fold_pred_dict, f)
        with open(f"./results_{args.result_path}/final_best_true_dict.pkl", "wb") as f:
            pickle.dump(best_fold_true_dict, f)
        with open(f"./results_{args.result_path}/final_sub_best_result_dict.pkl", "wb") as f:
            
            pickle.dump(sub_best_fold_result_dict, f)
        with open(f"./results_{args.result_path}/final_sub_best_pred_dict.pkl", "wb") as f:
            pickle.dump(sub_best_fold_pred_dict, f)
        with open(f"./results_{args.result_path}/final_sub_best_true_dict.pkl", "wb") as f:
            pickle.dump(sub_best_fold_true_dict, f)

# PowerShell에서 디렉토리 확인

