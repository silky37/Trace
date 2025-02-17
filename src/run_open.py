import json
import random
import time
import torch
import argparse
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import numpy as np
import string
import pandas as pd
from huggingface_hub import login

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
os.environ["HF_TOKEN"] = 'Your_Token'
login(os.environ["HF_TOKEN"])

def process_output(pred, choices):
    try:
        pred = pred.lower().replace("（", "(").replace("）", ")").replace(".", "")
        choices = [
            choice.replace(" & ", " and " if lang == "en" else "和")
            for choice in choices
        ]
        lines = pred.split("\n")
        for j in range(len(lines)):
            output = lines[len(lines) - 1 - j]
            if output:
                alphabets = {
                    "normal": [
                        f"({letters[i]})" for i in range(4)
                    ],
                    "paranthese": [
                        f"[{letters[i]}]" for i in range(4)
                    ],
                    "dot": [f": {letters[i]}" for i in range(4)],
                    "option": [
                        f"option {letters[i]}" for i in range(4)
                    ],
                    "option1": [
                        f"option ({letters[i]})"
                        for i in range(4)
                    ],
                    "choice": [
                        f"choice {letters[i]}" for i in range(4)
                    ],
                    "choice1": [
                        f"choice ({letters[i]})"
                        for i in range(4)
                    ],
                    "选项": [
                        f"选项 {letters[i]}" for i in range(4)
                    ],
                    "选项1": [
                        f"选项 ({letters[i]})" for i in range(4)
                    ],
                }

                for v in alphabets.values():
                    for a in v:
                        if a in output:
                            return v.index(a)
                for c in choices:
                    if c.lower() in output:
                        return choices.index(c)
                if len(output) == 1 and output in letters[:4]:
                    return letters.index(output)
                if output[0] in letters[:4] and output[1] in [
                    "<",
                    "[",
                    "(",
                    ")",
                    ":",
                ]:
                    return letters.index(output[0])
    except Exception as e:
        print("Error in processing output", type(e).__name__, "–", e)

    return -1
    
def format_prompt_4(d, prompt_key):
    emo1, emo2, emo3, emo4 = d['emotion1'], d['emotion2'], d['emotion3'], d['emotion4']
    e_c = [emo1, emo2, emo3, emo4]
    answer = d['answer']
    random.shuffle(e_c)
    scenario = d['scenario']
    subject = d['main_character']
    e_str = "\n".join([f"({letters[j]}) {c.strip()}" for j, c in enumerate(e_c)])
    e_pt = prompt[task]["Emotion"][lang].format(scenario=scenario, subject=subject, choices=e_str) + (prompt[prompt_key][lang])

    return e_pt, answer, e_c

def compute_metric(pred, ans):
    acc_p = np.mean([(pred[i]==ans[i] and pred[i+1]==ans[i+1]) for i in range(0,len(pred),2)]) * 100
    acc_q = np.mean([pred[i]==ans[i] for i in range(len(pred))]) * 100
    return acc_p, acc_q

def arrange_metric(prediction_data, col_name, data_path):
    time_data = prediction_data[(prediction_data['context_type']=='time')]
    place_data = prediction_data[(prediction_data['context_type']=='place')]
    agent_data = prediction_data[(prediction_data['context_type']=='agent')]
    flipped_data = prediction_data[(prediction_data['flipped']=='Y')]
    constant_data = prediction_data[(prediction_data['flipped']=='N')]

    acc_p_full, acc_q_full = compute_metric(prediction_data[col_name].tolist(), prediction_data['answer'].tolist())
    acc_p_time, acc_q_time = compute_metric(time_data[col_name].tolist(), time_data['answer'].tolist())
    acc_p_place, acc_q_place = compute_metric(place_data[col_name].tolist(), place_data['answer'].tolist())
    acc_p_agent, acc_q_agent = compute_metric(agent_data[col_name].tolist(), agent_data['answer'].tolist())
    acc_p_flipped, acc_q_flipped = compute_metric(flipped_data[col_name].tolist(), flipped_data['answer'].tolist())
    acc_p_constant, acc_q_constant = compute_metric(constant_data[col_name].tolist(), constant_data['answer'].tolist())

    with open(os.path.join(data_path, 'metrics_open.txt'), "a") as f:
        f.write(col_name + '\n')
        f.write(f'Time    : {round(acc_p_time,1)} / {round(acc_q_time,1)}' + '\n')
        f.write(f'Place   : {round(acc_p_place,1)} / {round(acc_q_place,1)}' + '\n')
        f.write(f'Agent   : {round(acc_p_agent,1)} / {round(acc_q_agent,1)}' + '\n')
        f.write(f'Flipped : {round(acc_p_flipped,1)} / {round(acc_q_flipped,1)}' + '\n')
        f.write(f'Constant: {round(acc_p_constant,1)} / {round(acc_q_constant,1)}' + '\n')
        f.write(f'Overall : {round(acc_p_full,1)} / {round(acc_q_full,1)}' + '\n')
        f.write('\n')

    print('Results for ' + col_name)
    print(f'Time    : {round(acc_p_time,1)} / {round(acc_q_time,1)}')
    print(f'Place   : {round(acc_p_place,1)} / {round(acc_q_place,1)}')
    print(f'Agent   : {round(acc_p_agent,1)} / {round(acc_q_agent,1)}')
    print(f'Flipped : {round(acc_p_flipped,1)} / {round(acc_q_flipped,1)}')
    print(f'Constant: {round(acc_p_constant,1)} / {round(acc_q_constant,1)}')
    print(f'Overall : {round(acc_p_full,1)} / {round(acc_q_full,1)}')
    print()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, default='./data', help='path to data'
    )
    parser.add_argument(
        '--model_id', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='specific name of the model'
    )
    parser.add_argument(
        '--cot', action='store_true', default=False, help='whether to use cot'
    )
    parser.add_argument(
        '--ps', action='store_true', default=False, help='whether to use plan-and-solve'
    )
    parser.add_argument(
        '--selfask', action='store_true', default=False, help='whether to use plan-and-solve'
    )
    parser.add_argument(
        '--n', type=int, default=-1, help='CoT to sample'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    data_path = args.data_path
    model_id = args.model_id
    cot = args.cot or (args.n > -1)

    if args.cot or (args.n > -1):
        prompt_key = 'cot'
    elif args.ps:
        prompt_key = 'ps'
    elif args.selfask:
        prompt_key = 'selfask'
    else:
        prompt_key = 'base'

    print('## Cofig ##')
    print('Model ID: ' + model_id)
    print('Prompt: ' + prompt_key)
    eval_data = pd.read_csv(os.path.join(data_path, "final_batch_head.csv"))
    data_size = eval_data.shape[0]
    prediction_data = eval_data.copy()

    val_dict = json.load(open("src/dicts.json", "r"))
    prompt = val_dict["Prompts"]
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    task,lang = 'EU','en'
    letters = string.ascii_lowercase
    system_pt = prompt["System" + (f'_{prompt_key}' if prompt_key!='base' else "")][lang]

    if '9b' in model_id:
        pipe = pipeline(
            "text-generation", 
            model=model_id, 
            model_kwargs={"torch_dtype": torch.bfloat16},  
            device=0
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    predictions = []
    for idx in tqdm(range(data_size), desc='Generating...'):
        d = eval_data.loc[idx]
        e_pt, answer, e_c = format_prompt_4(d, prompt_key)

        if 'gemma' in model_id:
            e_pt = [
                        {"role": "user", "content": system_pt + '\n\n' + e_pt},
                    ]
        else:
            e_pt = [
                        {"role": "system", "content": system_pt},
                        {"role": "user", "content": e_pt},
                    ]

        if '9b' in model_id:
            output = pipe(e_pt, max_new_tokens=500 if model_id!='base' else 75, 
                            pad_token_id=pipe.tokenizer.eos_token_id)
            e_response = output[0]['generated_text'][-1]['content'].strip()
        else:
            tokenized_chat = tokenizer.apply_chat_template(e_pt, tokenize=True, add_generation_prompt=True, 
                                                        return_tensors="pt", return_dict=True).to(device)
            
            output = model.generate(**tokenized_chat, pad_token_id=tokenizer.eos_token_id, 
                                    max_new_tokens=500 if model_id!='base' else 75)
            e_response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        e_r = process_output(e_response, e_c)

        # print('## output ##')
        # print(system_pt)
        # print('#'*10)
        # print(e_pt[-1]['content'])
        # print('#'*10)
        # print(e_response)
        # print('#'*10)
        # print(e_r)

        pred_emotion = e_c[e_r]
        predictions.append(pred_emotion)

    # Whether the emotion is flipped
    if 'flipped' not in prediction_data.columns:
        flipped = []
        for i in range(prediction_data.shape[0]):
            if i % 2 == 0:
                c = "N" if prediction_data.loc[i]['answer']==prediction_data.loc[i+1]['answer'] else "Y"
            else:
                c = "N" if prediction_data.loc[i]['answer']==prediction_data.loc[i-1]['answer'] else "Y"
            flipped.append(c)
        prediction_data.insert(9, 'flipped', flipped)

    # Add predictions
    predictions = predictions + ['']*(data_size-len(predictions))
    col_name = model_id + (f'-{prompt_key}' if prompt_key!='base' else '') + (str(args.n) if args.n > -1 else '')
    if col_name in prediction_data.columns:
        prediction_data = prediction_data.drop(columns=[col_name])

    prediction_data.insert(len(prediction_data.columns), col_name, predictions)
    prediction_data.to_csv(os.path.join(data_path, "final_batch_head.csv"), index=False, sep=',', na_rep='')
    arrange_metric(prediction_data, col_name, data_path)