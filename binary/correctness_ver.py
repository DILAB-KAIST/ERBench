import pandas as pd
import argparse
import random
from openai import AzureOpenAI
import os
from nameparser import HumanName
import time
def parse_args():
    '''
    Config
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices = ['movie', 'soccer', 'airport', 'music', 'book', 'soccer_olympic', 'soccer_key', 'movie_foreign_year', 'multimodal_movie','multimodal_soccer'], default = 'movie')
    parser.add_argument('--demo', type=str, default='null')
    parser.add_argument('--model', type=str, choices=['gpt35', 'gpt4', 'llama', 'mistral', 'gemini', 'gemini_vision', 'gpt_finetune', 'gpt_finetune_2', 'claude'], default='gpt35')
    parser.add_argument('--rag', type=str, default='null')
    parser.add_argument('--index', type = int, default = 0)
    args = parser.parse_args()
    
    return args





def run_gpt4(prompt):
    randomness = random.randint(0,1)
    if randomness ==0:
        client = AzureOpenAI( 
        api_version = "2024-02-15-preview",
        azure_endpoint=  #writeyour endpoint,
        api_key=#write your api key
        )
    else:
        client = AzureOpenAI( 
        api_version = "2024-02-15-preview",
        azure_endpoint = #write your endpoint, 
        api_key=#write your api key
        )


   # system_prompt = 'Answer only looking at A1 and A2 without your knowledge. If some properties are missing in A1 assume that the corresponding properties are the same as A2. Answer in yes or no.'
    msg = [
       #{"role": "system", "content": system_prompt},
        {"role": "user", "content":prompt},
    ]
       
    completion = client.chat.completions.create(
        model=#write you model name
        messages = msg, 
        temperature = 0
    )
    return completion.choices[0].message.content
def retrieve_qa(task, demo, model, rag):
    df = pd.DataFrame(columns =['entity_idx', 'prompt_idx', 'question', 'rationale', 'model_gold_entity', 'gold_entity'])

    if demo !='null':
        log_directory = f'./results/{model}/{task}_{demo}.log'
    else:
        if rag =='null':
            log_directory = f'./results/{model}/{task}.log'
        else:
            log_directory = f'./results/{model}/{task}_rag.log'
    with open(log_directory,"r") as f:
        while(True):

            line = f.readline()
            if len(line)==0:
                break
            if "th question" in line:
                two_idx = line.split("th question")[0]
                movie_idx = int(two_idx.split('-')[0])
                prompt_idx = int(two_idx.split('-')[1])
                line =f.readline()
                while('Q: ' not in line):
                    line =f.readline()
                question = line.split(":")[1].replace('\n','')
                line = f.readline()
                rationale = ""
                
                while ('Gold Answer:' not in line):
                    
                    if 'A:' in line:
                        rationale = line.split('A:')[1].strip()
                        line=f.readline()
                        continue
                    if line.strip() =="":
                        line = f.readline()
                        continue
                    rationale = rationale + line.strip()
                    line = f.readline()

                ##GOLD ANSWER: in line
                if "Gold Answer:" not in line:
                    raise ValueError("GOLD ANSWER ERROR")
                line = f.readline()
                if "Gold Entity:" not in line:
                    raise ValueError("GOLD ENTITY ERROR")
                
                gold_entity = line.split("Gold Entity: ")[1].strip().lower()
                gold_orig_entity  = line.split("Gold Entity: ")[1].strip().lower()
                if task=='soccer_olympic':
                    gold_entity = gold_entity.split(',')
                    if len(gold_entity[-1].strip())==0:
                        gold_entity = gold_entity[:len(gold_entity)-1]
                    if len(gold_entity)>=3:
                        tmp_entity = ["/".join(gold_entity[2:])]
                        gold_entity = gold_entity[:2]+tmp_entity
                    gold_entity = ','.join(gold_entity)
                if task=='soccer' or task=='soccer_key' or task=='multimodal_soccer':
                    name = HumanName(gold_entity)
                    first_name = name.first
                    last_name = name.last
                    if last_name =='' or 'jr' in last_name:
                        gold_entity = first_name
                    else:
                        gold_entity = last_name
                if prompt_idx !=2:
                    df.loc[len(df.index)] = [movie_idx, prompt_idx, question, rationale, gold_entity, gold_orig_entity]



    if demo !='null':
        res_dir = f'./correctness_verification/data/{model}/{task}_{demo}.csv'
    else:
        if rag =='null':
            res_dir = f'./correctness_verification/data/{model}/{task}.csv'
        else:
            res_dir = f'./correctness_verification/data/{model}/{task}_rag.csv'
    df.to_csv(res_dir)
    return df
def run_gpt_judge(task, df,model, demo, rag, index):
    if task=='movie':
        task_keywords = ['Is there a movie, ', 'Is it true that there are no movies, ']
        attributes = "released year, star, director"
    elif task =='soccer':
        task_keywords = ['Is there a soccer player ', 'Is it true that there are no soccer players ']
        attributes = "uniform number, associdated club, nationality (in 2019)"
    elif task == 'airport':
        task_keywords = ['Is there an airport ', 'Is it true that there are no airports ']
        attributes = "longitude, latitude"
    elif task=='music':
        task_keywords = ["Is there an artist or group", "Is it true that no artists nor groups "]
        attributes = "song title, released year"
    elif task=='book':
        task_keywords = ["Is there a book ", "Is it true that there are no books"]
        attributes = "author, published date"
    human_df = pd.read_csv(f'./correctness_verification/human_data/{model}/{task}_correct.csv')
    for i in range(len(df)): #len(df)
        entity_idx = df.loc[i,'entity_idx']
        if(index>entity_idx):
            i+=1
            continue
        prompt_idx = df.loc[i,'prompt_idx']
        question = df.loc[i,'question']
        rationale = df.loc[i, 'rationale']
        gold_entity = df.loc[i,'model_gold_entity']
        gold_orig_entity = df.loc[i,'gold_entity']
        for task_keyword in task_keywords:
            if task_keyword in question:
                gold_answer = question.split(task_keyword)[1]
                
        if task =='movie':
            prior = f"The movie, {gold_orig_entity}, was "
            entity = 'movie'
        elif task == 'soccer':
            prior = f"The soccer player, {gold_orig_entity}, was "
            entity = "soccer player"
        elif task =='airport':
            prior = f"The airport, {gold_orig_entity}, is "
            entity = "airport"
        elif task=='music':
            prior = f"The artist or group, {gold_orig_entity}, "
            entity = 'artist'
        elif task=='book':
            prior = f"The book, {gold_orig_entity}, is "
            entity = 'book'

        
        gold_answer = prior + gold_answer
        gold_answer = gold_answer[:-1] + '.'
        prompt = f'Answer in yes or no. A1: {rationale}\nA2: {gold_answer} \n\nAre the two answers,A1 and A2, referring to the same {entity}, {gold_orig_entity}, with the same properties, {attributes}? If there is no {entity} names mentioned in both A1 and A2, output yes. If only one of A1 and A2 mention a {entity} name, output no. If both A1 and A2 mention {entity} name, answer only looking at A1 and A2 without your knowledge. \n'
        time.sleep(0.2)

        if not (model=='claude' and task =='music' and entity_idx==974):
            output = run_gpt4(prompt)

            time.sleep(0.2)
            if gold_entity.lower() not in rationale.lower():
                gold_output = "no"
            else:
                gold_output = 'yes'
            if "yes" in output.lower():
                if gold_output=='yes':
                    judge = "Correct w/ ERBench"
                else:
                    judge = "Wrong w/ ERBench"
            elif "no" in output.lower():
                if gold_output =='yes':

                    judge = 'Wrong w/ ERBench'
                else:
                    judge =  "Correct w/ ERBench"
                
            else:
                judge = output
        else:
            output = 'filtered, hence manually inspected'
            judge = 'Correct w/ ERBench'
        if task=='soccer':
            soccer_idx = [69, 73, 119, 131, 183, 199, 217, 296, 299, 420, 457, 458, 531, 535, 562, 570, 572, 587, 604, 608, 634, 657, 683, 691, 703, 716, 725, 771, 799, 867, 870, 913, 956, 959, 972, 1012, 1050, 1055, 1127, 1137, 1183, 1231, 1255, 1291, 1338, 1348, 1423, 1449, 1476, 1483]
            if entity_idx in soccer_idx:
                desired_row = human_df[(human_df['entity_idx'] == entity_idx) & (human_df['question_idx'] == prompt_idx)]
                human_res = 'no' if int(desired_row.iloc[0]['reasoning_manual'])==0 else 'yes'
                if ('yes' in output.lower() and human_res =='yes') or ('no' in output.lower() and human_res =='no'):
                    judge_human = 'Correct w/ human'
                else:
                    judge_human = 'Wrong w/ human'
            else:
                judge_human = "DID NOT COMPUTE"

        else:
            if i<100:
                human_res = 'no' if int(float(human_df.loc[i, 'reasoning_manual']))==0 else 'yes'
                if ('yes' in output.lower() and human_res =='yes') or ('no' in output.lower() and human_res =='no'):
                    judge_human = 'Correct w/ human'
                else:
                    judge_human = 'Wrong w/ human'
            else:
                judge_human = "DID NOT COMPUTE"
            
        if demo !='null':
            log_directory = f'./correctness_verification/results/{model}/{task}_{demo}.log'
        else:
            if rag =='null':
                log_directory = f'./correctness_verification/results/{model}/{task}.log'
            else:
                
                log_directory = f'./correctness_verification/results/{model}/{task}_rag.log'
        
        with open(log_directory,'a')as f:
            f.write(f"{entity_idx}-{prompt_idx}th question\n")
            f.write(f"Model Result: {rationale.strip()}\n")
            f.write(f"Gold Answer: {gold_answer.strip()}\n")
            f.write(f"GPT-Judge Output: {output.strip()}\n")
            f.write(f"GPT-Judge Result vs ERBench: {judge}\n")
            f.write(f"GPT-Judge Result vs human: {judge_human}\n\n")


            



def main():
    args = parse_args()
    task = args.task.strip()
    demo = args.demo
    model = args.model
    rag = args.rag
    index = args.index
    demo_print = '' if demo =='null' else f'demo {demo} '
    print(f"Running to Analyze {task}, {model}, {demo_print}analysis for correctness_verification!!")

    if demo !='null':
        res_dir = f'./correctness_verification/data/{model}/{task}_{demo}.csv'
    else:
        if rag =='null':
            res_dir = f'./correctness_verification/data/{task}.csv'
        else:
            res_dir = f'./correctness_verification/data/{task}_rag.csv'
    if not os.path.exists(res_dir):
        df = retrieve_qa(task, demo, model, rag)
    else:
        df = pd.read_csv(res_dir)
    
    run_gpt_judge(task, df, model, demo, rag,index)
    return
            
    

if __name__  =="__main__":
    main()