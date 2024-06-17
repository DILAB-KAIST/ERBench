import pandas as pd
import numpy as np
import argparse
import time
import pickle
import utils
import math


def parse_args():
    '''
    Config
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices = ['movie', 'soccer', 'airport', 'music','book', 'soccer_olympic', 'soccer_key','multimodal_movie', 'movie_foreign_year', 'movie_foreign_city', 'multimodal_soccer'], default = '', required = True)
    parser.add_argument('--tasktype', type=str, choices=['test', 'validate'], default= 'test')
    parser.add_argument("--index", type=int, default = 0)
    parser.add_argument("--demo", type=str, default='null')
    parser.add_argument("--rag", type =str, default='null')
    parser.add_argument('--model',type=str, default = 'gpt' )
    args = parser.parse_args()
    
    return args


def get_prompt(task, entity1, common_entity="N/A", entity2="N/A", entity3 = "N/A", entity4 = "N/A"):
    prompts = []
        
    if task == 'movie':
        #ENTITY1: Star, Entity2: Movie Director, Entity3: Year
        prompts.append( f"Is there a movie, released in {int(entity3)}, starring {entity1} where {entity2} is the director?") #YES
        prompts.append( f"Is it true that there are no movies, released in {int(entity3)}, starring {entity1} where {entity2} is the director?")
        #prompts.append( f"Is it true that there is a movie, released in {int(entity3)}, starring {entity1} where {entity2} is the director?")
    if task =='soccer':
        #  fifa_df = fifa_df[['short_name', 'club_name','club_jersey_number', 'nationality_name']]
   # fifa_df.columns = ['common_entity', 'entity1', 'entity3', 'entity2']
        prompts.append(f"Is there a soccer player from {entity2} who played for {entity1} with uniform number {int(entity3)} in {entity1} in 2019?")
        prompts.append(f"Is it true that there are no soccer players from {entity2} who played for {entity1} with uniform number {int(entity3)} in {entity1} in 2019?")
        #prompts.append(f"Is it true that there is a soccer player from {entity2} who played for {entity1} with uniform number {int(entity3)} in {entity1} in 2019?")
    if task == 'airport':
        prompts.append(f"Is there an airport located at latitude {entity1} and longitude {entity2}?")
        prompts.append(f'Is it true that there are no airports located at latitude {entity1} and longitude {entity2}?')
        #prompts.append(f'Is it true that there is an airport located at latitude {entity1} and longitude {entity2}?')
    if task == 'music':
        prompts.append(f"Is there an artist or group who sang a song titled {entity1} in {entity2}?")
        prompts.append(f"Is it true that no artists nor groups sang a song titled {entity1} in {entity2}?")
        #prompts.append(f"Is it true that there is an artist or group who sang a song titled {entity1} in {entity2}?")
    if task =='book':
        prompts.append(f"Is there a book written by {entity1} that was published in {entity2}?")
        prompts.append(f"Is it true that there are no books written by {entity1} that were published in {entity2}?")
        #prompts.append(f"Is it true that there is a book written by {entity1} that was published in {entity2}?")
    if task =='soccer_olympic':
        #,entity1,entity2,entity3,gold_entity
        #0,L. Messi,FC Barcelona,barcelona,"1992,"
        prompts.append(f"Did the city, where the soccer club, {entity1} played for in 2019, is located in, hosted the Summer Olympics?")
        prompts.append(f"Is it true that the city, where the soccer club, {entity1} played for in 2019, is located in, never hosted the Summer Olympics?")
        #prompts.append(f"Is it true that the city, where the soccer club, {entity1} played for in 2019, is located in, hosted the Summer Olympics?")
    if task=='soccer_key':
        prompts.append(f"Is the player who wore jersey number {entity2} in soccer club {entity1} at 2019 the same player as the player who wore jersey number {entity4} for {entity3} national team in 2019?")
        prompts.append(f"Are the players who wore jersey number {entity2} in soccer club {entity1} at 2019 the same players as the players who wore jersey number {entity4} for {entity3} national team in 2019?")
        #prompts.append(f"Is it true that the player who wore jersey number {entity2} in soccer club {entity1} at 2019 a different player as the player who wore jersey number {entity4} for {entity3} national team in 2019?")
    
    if task=='movie_foreign_city':
        prompts.append(f"Was the director who directed the movie titled {common_entity} that was released at {entity3} born in {entity4}?")
        prompts.append(f"Is it true that the director who directed the movie titled {common_entity} that was released at {entity3} was not born in {entity4}?")
        #prompts.append(f"Is it true that the director who directed the movie titled {common_entity} that was released at {entity3} was born in {entity4}?")


    if task=='movie_foreign_year':
        prompts.append(f"Was the director who directed the movie titled {common_entity} that was released in {int(entity3)} born in the {int(entity4 // 10 * 10)}s?")
        prompts.append(f"Is it true that the director who directed the movie titled {common_entity} that was released in {int(entity3)} was not born in the {int(entity4 // 10 * 10)}s?")
        #prompts.append(f"Is it true that the director who directed the movie titled {common_entity} that was released in {int(entity3)} was born in the {int(entity4 // 10 * 10)}s?")
    if task=='multimodal_soccer':
        prompts.append(f"Is there a soccer player from {entity2} who played for the club in the image with uniform number {int(entity3)} in the club in the image in 2019?")
        prompts.append(f"Is it true that there are no soccer players from {entity2} who played for the club in the image with uniform number {int(entity3)} in the club in the image in 2019?")
       # prompts.append(f"Is it true that there is a soccer player from {entity2} who played for the club in the image with uniform number {int(entity3)} in the club in the image in 2019?")
    if task=='multimodal_movie':
        prompts.append(f"Is the movie, released in {int(entity3)}, starring {entity1} where {entity2} is the director the same movie as the movie with the movie poster as the given image?")
        #prompts.append(f"Is it true that there are no soccer players from {entity2} who played for the club in the image with uniform number {int(entity3)} in the club in the image in 2019?")
        prompts.append(f"Is the movie, released in {int(entity3)}, starring {entity1} where {entity2} is the director a different movie as the movie with the movie poster as the given image?")
    
    return prompts

def is_float(s):
    try:
        float_value = float(s)
        return True
    except ValueError:
        return False

def run_target(task, df, demo_type, rag_str, model = 'gpt35', index = 0):
    def is_float(string):
        try:
            float(string)
            return True
        except ValueError:
            return False
    entity1, entity2, gold_entity, entity3,entity4 = "N/A","N/A","N/A", "N/A","N/A"

    
    cot = False
    rag = True if rag_str != 'null' else False
    if rag:
        rag_exclude_idx= set()
    if demo_type !='null':
        cot_q = []
        cot_a = []
        with open(f'./demo_prompt/{task}_{demo_type}.txt') as f:

            cot_prompt = f.readlines()
            for line in cot_prompt:
                if line =='\n' or line=='':
                    continue
                if line.startswith('Q: '):
                    line = line.split('Q: ')[1]
                    cot_q.append(line)
                elif line.startswith('A: '):
                    line = line.split('A: ')[1]
                    cot_a.append(line)
        cot = True

    for i in range(len(df)):
        if(index>i):
            continue
        if (task=='multimodal_soccer') and (i==761):
            continue
        if task=='multimodal_movie' and model=='gpt4v' and (i==426 or i==711 or i==1219):
            continue
        if (task =='soccer' or task =='soccer_olympic') and (i==1069 or i==609 or i==1067):
        #    #MANUAL EXCLUSION
            continue
        if task=='music' and (i==960 or i==810):
            continue
        if task =='book' and (i==47 or i==232):
            continue
        if task =='multimodal_movie' and (i==288 or i==289 or i==461 or i==755 or i==842 or i==879 or i==936 or  i==1050 or i==1072 or i==1086 or i==1339):
            continue
        if task =='movie' and (i==1016):
            continue
        if task =='movie' and rag and (i==677 or i==1104 or i==1328 or i==1420):
            continue
        if task =='book' and model=='gpt_finetune' and (i==123):
            continue
        if task =='soccer' and model =='gpt_finetune' and (i==463):
            continue
        if task =='music' and model =='gpt_finetune' and (i==307 or i==733 or i==1042):
            continue
        if task =='music' and model =='claude' and (i==204 or i==244 or i==684 or i==948 or i==1014 or i==250 or i==1100): #from 250 due to demo
            continue
        if task =='book' and model =='gpt_finetune_2' and (i==239 or i==1055):
            continue

        entity1 = df.loc[i,'entity1']
        entity2 = df.loc[i,'entity2']
        if 'entity3' in df.columns:
            entity3 = df.loc[i,'entity3']
            if task=='soccer' or task =='movie' or task =='multimodal_soccer':
                if float(entity3).is_integer():
                    entity3 = int(entity3)
                else:
                    entity3 = -1
        if 'entity4'in df.columns:
            entity4 = df.loc[i,'entity4']
        #SOCCER: ENTITY1: Club, entity3 uniform number, entity2: nationality
        if 'foreign' in task and 'movie' in task:
            gold_entity = f"{df.loc[i,'entity2']}, {int(df.loc[i,'entity4'])}"
        elif task == 'soccer_olympic':
            gold_entity = f"{entity2}, {df.loc[i, 'common_entity']}, {entity3}"
        else:
            gold_entity=df.loc[i, 'common_entity']
        common_entity = df.loc[i, 'common_entity']
        
        if task=='soccer_key':
            if float(entity2).is_integer():
                entity2 = int(entity2)
            else:
                print(f"Entity2 shouldn't happen for {i}th player")
                entity2 = -1
            if float(entity4).is_integer():
                entity4 = int(entity4)
                
            else:
                print(f"Entity4 shouldn't happen for {i}th player")
                entity4 = -1
#        if entity1=="N/A" or entity2 =="N/A" or gold_entity =='N/A':
 #           raise ValueError("Value of Entity is Strange")
        prompts =   get_prompt(task = task, entity1 = entity1, entity2 =  entity2,entity3=entity3, entity4=entity4, common_entity=common_entity)
        if task =='soccer' and rag:
            queries = [f'Football Club {entity1.strip()} in 2019', f'{entity2.strip()} National Football Team in 2019']
            rag_text = utils.retrieval_augmented_text(queries, task, entity1.strip(), entity2.strip())
            if rag_text.strip()=='':
                rag_exclude_idx.add(i)
        elif task =='movie' and rag:
            queries = [f'actor {entity1.strip()}', f'director {entity2.strip()}']
            rag_text = utils.retrieval_augmented_text(queries,task,entity1.strip(), entity2.strip())
            if rag_text.strip()=='':
                rag_exclude_idx.add(i)
        else:
            rag_text = ""
        id =0
        err_cnt=0
        while(id < len(prompts)):
            if err_cnt==5:
                return
            prompt_id = id
            prompt = prompts[prompt_id]
            qprompt = rag_text + prompt
            idx = str(i)+'-'+str(prompt_id)
            id+=1
            try:
                if cot:
                    
                    response = utils.run_llm(prompt = qprompt, cot_q = cot_q, cot_a = cot_a, cot=cot, model=model)

    
                    
                else:
                    if model =='gemini_vision':
                        if task=='multimodal_soccer':
                            img_keyword = entity1
                            if is_float(img_keyword) and np.isnan(img_keyword):
                                img_keyword = 'Undefined Club'
                        elif task=='multimodal_movie':
                            img_keyword = f'movie {gold_entity} poster'
                        response= utils.run_llm(prompt = qprompt, img_keyword=img_keyword, model = model)
                    elif model =='gpt4v':
                        if task=='multimodal_soccer':
                            img_keyword = entity1
                            if is_float(img_keyword) and np.isnan(img_keyword):
                                img_keyword = 'Undefined Club'
                        elif task=='multimodal_movie':
                            img_keyword = f'movie {gold_entity} poster'
                        response= utils.run_llm(prompt = qprompt, img_keyword=img_keyword, model=model)
                    else:
                        response = utils.run_llm(prompt = qprompt, model=model)
            except Exception as error:
                id -=1
                err_cnt +=1
                if model=='gpt4v':
                    time.sleep(10)
                print(f"{idx}th error: {error}")
                continue
            if 'gemini' in model:
                time.sleep(2)
            elif model=='gpt4v':
                time.sleep(2)
            elif model =='claude':
                time.sleep(1)
            else:
                time.sleep(0.5)
  

          

            if not task =='soccer_olympic':            
                gold_answer = "no" if prompt_id==1 else "yes" ##MODIFIED!!!!
            else:
                if len(entity3.strip())==0:
                    gold_answer = 'yes' if prompt_id==1 else "no" 
                else:
                    gold_answer = "no" if prompt_id==1 else "yes" 
            
            write_to_file(task, prompt,rag_text, idx, gold_answer, gold_entity,response, demo_type=demo_type, model=model, rag=rag)
            
        if i==1500:
            break
        #break
    if rag:
        with open(f'./rag_data/{task}_exc_idx.pkl','wb') as f:
            pickle.dump(   rag_exclude_idx, f)
        
def validation_step(task, df,model='gpt35', index = 0):
    entity1, entity2, gold_entity, entity3,entity4 = "N/A","N/A","N/A", "N/A","N/A"

    cnt=0
    i=0
    while(i<len(df)):
        if cnt>100:
            cnt=0
            continue
        if(index>i):
            i+=1
            continue
        entity1 = df.loc[i,'entity1']
        entity2 = df.loc[i,'entity2']

        gold_entity = df.loc[i, 'common_entity']
        if 'entity3' in df.columns:
            entity3 = df.loc[i, 'entity3']
        if 'entity4' in df.columns:
            entity4 = df.loc[i,'entity4']
        if entity1=="N/A" or entity2 =="N/A" or gold_entity =='N/A':
            raise ValueError("Value of Entity is Strange")
        if task == 'movie':
            prompt = f"Do you know the movie {gold_entity} released in {int(entity3)}? If yes, did {entity1} star in the movie {gold_entity}? If yes, did {entity2} direct the movie {gold_entity}?"

                

        elif task =='soccer' :
            if float(entity3).is_integer():
                prompt = f"Do you know the soccer player {gold_entity}? If yes, was {gold_entity} born in {entity2}? If yes, did {gold_entity} play for {entity1} with back number {int(entity3)} in 2019?"
            #prompt = f"Do you know the soccer player {gold_entity}? If yes, was {gold_entity} born in {entity2}? If yes, was {gold_entity} the only player to play for {entity1} with back number {int(entity3)} in 2019?"
            else:
                response = 'no'
                prompt = f"Do you know the soccer player {gold_entity}? If yes, was {gold_entity} born in {entity2}? If yes, did {gold_entity} play for {entity1} with back number {entity3} in 2019?"
                write_to_validate_file(task, prompt, str(i), response, model=model)

                continue
        elif task=='soccer_olympic':
            if entity3 ==' ':
                prompt = f"Did {entity1} play for {entity2}? If yes, is {entity2} located in the city {gold_entity}?"
            else:
                prompt = f"Did {entity1} play for {entity2}? If yes, is {entity2} located in the city {gold_entity}? If yes did {gold_entity} host the summer olympics at {entity3}?"
        elif task=='soccer_key':
            
            prompt = f"Do you know the soccer player {gold_entity}? If yes, did {gold_entity} play for {entity1} with back number {int(entity2)}? If yes, did  {gold_entity} play for {entity3} with back number {int(entity4)}?"
        
        elif task=='airport':
            prompt = f"Is the {gold_entity} Airport located at latitude {entity1}, longitude {entity2}?"
        elif task =='book':
            #prompt = f"Do you know the book, {gold_entity}? If yes, is the book {gold_entity} written by {entity1}? If yes, is the book {gold_entity} published on {entity2}?"
            prompt = f"Do you know the book, {gold_entity} written by {entity1}? If yes, is the book {gold_entity} written by {entity1} published in {entity2}?"
        elif task =='music':
            prompt = f"Do you know the music {entity1} by {gold_entity}? If yes, was the {entity1} by {gold_entity} released in {entity2}?"
        elif task=='movie_foreign_directorcity':
            prompt = f"Do you know the movie {gold_entity} released at {entity3}? If yes, did {entity2} direct the movie {gold_entity}? Was {entity2} born in {entity4}?"
        elif task =='movie_foreign_year':
            prompt =f"Do you know the movie {gold_entity} released in {int(entity3)}? If yes, did {entity2} direct the movie {gold_entity}? Was {entity2} born in year {int(entity4)}?"
        elif task=='multimodal_soccer':
            if float(entity3).is_integer():
                prompt = f"Do you know the soccer player {gold_entity}? If yes, was {gold_entity} born in {entity2}? If yes, did {gold_entity} play for the club in the image with back number {int(entity3)} in 2019?"
            else:
                response = 'no'
                prompt = f"Do you know the soccer player {gold_entity}? If yes, was {gold_entity} born in {entity2}? If yes, did {gold_entity} play for the club in the image with back number {int(entity3)} in 2019?"
                write_to_validate_file(task, prompt, str(i), response, model=model)

                continue
        elif task=='multimodal_movie':
            prompt = f"Do you know the movie {gold_entity} released in {int(entity3)}? If yes, did {entity1} star in the movie {gold_entity}? If yes, did {entity2} direct the movie {gold_entity}? If yes, is the movie poster in the given image the movie poster of this movie, {gold_entity}?"
        else:
            raise ValueError('Strange Task')
        try:
            if model =='gpt35':
                if (task=='music') and (i==739 or i==1337):
                    response = 'no'
                elif task=='soccer_olympic' and i==609:
                    response = 'no'
                else:
                    response = utils.run_llm(prompt, tasktype='validate', model = model)
            elif model =='gpt4':
                if (task=='soccer' and i==609) or (task=='book' and i==608):
                    response = 'no'
                elif task=='soccer_olympic' and i==609:
                    response = 'no'
                elif (task=='music') and (i==739 or i==1337):
                    response = 'no'
                else:
                    response = utils.run_llm(prompt, tasktype='validate', model=model)
            elif model=='gemini_vision':
                if task=='multimodal_soccer':
                    img_keyword = entity1
                elif task=='multimodal_movie':
                    img_keyword = f'movie {gold_entity} poster'
                if task =='multimodal_movie' and (i==288 or i==289 or i==461 or i==755 or i==842 or i==879 or i==936 or i==1050 or i==1072 or i==1086):
                    response = 'no'
                else:
                    response = utils.run_llm(prompt,img_keyword=img_keyword, tasktype='validate', model = model)
            elif model=='gpt4v':
                if task=='multimodal_soccer':
                    img_keyword = entity1
                elif task=='multimodal_movie':
                    img_keyword = f'movie {gold_entity} poster'
                if task =='multimodal_movie' and (i==426 or i==936 or i==1219):
                    response = 'no'
                else:
                    response = utils.run_llm(prompt,img_keyword=img_keyword, tasktype='validate', model =model)
            elif model=='gpt_finetune':
                if (task=='music') and (i==864):
                    response = 'no'
                elif( task=='soccer') and (i==609):

                    response='no'
                else:
  
                    response = utils.run_llm(prompt, tasktype = 'validate', model = model)
            elif model=='gpt_finetune_2':
                if task =='soccer' and (i==609):
                    response = 'no'
                elif task=='music' and (i==1337):
                    response = 'no'
                else:
                    response = utils.run_llm(prompt, tasktype='validate', model = model)
            elif model=='claude':
                if task =='music' and i==1100:
                    response = 'no'
                else:
                    response = utils.run_llm(prompt, tasktype='validate', model = model)
        except Exception as error:
            #i -=1
            cnt+=1
            print(f"{i}th question error for {model}, {error}")
            continue
        if 'gemini' in model:
            time.sleep(2)
        else:
            time.sleep(0.5)
        cnt=0
        write_to_validate_file(task, prompt, str(i), response, model = model)
        i+=1

        #break
        
        


def main():
    args = parse_args()
    task = args.task.strip()
    idx = args.index
    tasktype = args.tasktype
    demo_type = args.demo
    model = args.model
    rag_str = args.rag
    print(f"Running {task}!!")
    utils.preprocess(task)
    if task =='movie'or task=='multimodal_movie':
        df = pd.read_csv(f'./movie_data/crafted/movie_df.csv')
    elif task =='soccer' or task=='multimodal_soccer':
        if rag_str =='null':
            df = pd.read_csv('./fifa_data/crafted/fifa_df.csv')
        else: 
            df = pd.read_csv('./fifa_data/crafted/fifa_df_2.csv')
    elif task =='airport':
        df = pd.read_csv('./airport_data/crafted/airport_df.csv')
    elif task =='music':
        df = pd.read_csv('./music_data/crafted/music_df.csv')
    elif task=='book':
        df = pd.read_csv('./book_data/crafted/book_df.csv')
    elif task=='soccer_olympic':
        df = pd.read_csv('./soccer_olympic/crafted/total_df.csv')
    elif task=='soccer_key':
        df = pd.read_csv('./fifa_data/crafted/fifa_key_df.csv')
    elif task=='movie_foreign_city':
        df = pd.read_csv('./movie_data/crafted/movie_city_final.csv')
    elif task =='movie_foreign_year':
        df = pd.read_csv('./movie_data/crafted/movie_year_final.csv')
    if tasktype =='validate':
        print(f'VALIDATING task : {task}!!')
        validation_step(task= task, df=df, index=idx, model=model)
    #columns = get_columns(task)
    else:
        print(f"Starting Experiment! {task}")
        run_target(task=task, df = df, index=idx, demo_type =demo_type, rag_str = rag_str, model = model)
    
    return



        


def write_to_validate_file(task, prompt, idx, response, model):
    folder ='results'
    directory = f'./{folder}/{model}/validate_{task}.log'

    with open(directory,'a') as f:
        f.write(f"{idx}th question\n")
        f.write("Q: ")
        f.write(prompt)
        f.write("\nA:")
        f.write(response)
        f.write('\n\n')


def write_to_file(task, qprompt,rag_text, index, answer, entity, response, demo_type, model,rag):

    folder ='results'
    if demo_type!='null':
        directory = f'./{folder}/{model}/{task}_{demo_type}.log'
    else:
        if rag:
            directory = f'./{folder}/{model}/{task}_rag.log'
        else:
            directory = f'./{folder}/{model}/{task}.log'

    with open(directory,'a') as f:
        f.write(f"{index}th question\n")
        f.write(rag_text)
        f.write("Q: ")
        f.write(qprompt)
        f.write("\nA:")
        f.write(response)
        f.write('\nGold Answer: ')
        f.write(str(answer))
        f.write('\nGold Entity: ')
        f.write(str(entity))
        f.write('\n\n')

    return

if __name__ == "__main__":
    main()