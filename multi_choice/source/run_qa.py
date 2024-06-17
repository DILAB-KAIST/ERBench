import pandas as pd
import argparse
import time
import os
import random
import ast
from langchain_community.retrievers import WikipediaRetriever
from utils import *
import re


def preprocess(task_to_dbpath, task, dpath, random_seed):
    '''
    Given task, preprocess database to get entity/relations.

    Notes:
    We found a minor bug in reproducing from the original dataset for music/airport/book datasets.
    Please refer binary QA how to craft the other datasets.
    '''
    
    # read original database
    if os.path.exists(task_to_dbpath[task]):
        df = pd.read_csv(task_to_dbpath[task])
    else:
        raise ValueError(f"Error: Original database path ({task_to_dbpath[task]}) does not exit.")


    # select entity attributes (i.e. database columns) w.r.t. tasks
    if task == 'soccer':
            
        # select columns and drop na
        column_names = ['long_name', 'short_name', 'club_name','club_jersey_number', 'nationality_name', 'league_name']
        df = df[column_names] 
        df.dropna(inplace=True)

        # convert jersey number to int
        df['club_jersey_number'] = df['club_jersey_number'].astype(int)
        
        # sample 1500 rows from the first (to account famous players)
        # df = df.iloc[0:1500]

    
    elif task == 'movie':

        # select columns and drop na
        column_names = ['movie_title','title_year', 'director_name', 'country', 'genres']
        df = df[column_names] 
        df.dropna(inplace=True)

        # strip spaces
        df['movie_title'] = df['movie_title'].str.strip()
        df['title_year'] = df['title_year'].astype(int)
        df['genres'] = df['genres'].apply(lambda x: 'animation' if 'Animation' in x else 'non-animation')

        # sample 1500 rows from the first (to account famous movies)
        df = df.iloc[0:1500]

        # drop duplicates
        df.drop_duplicates(subset=['movie_title'], inplace=True)


        
    else:
        # For other music/airport/book datasets, we find a minor bug in reproducing from the original dataset.
        # Please use the crafted dataset for the reproducibility of the results in the paper.
        # You can refer to preprocess_music.py, preprocess_airport.py, and preprocess_book.py included in the code for binary QA 
        # for the crafted dataset.
        raise ValueError("ERROR: Preprocessing function of given dataset is not yet defined.")



    # reset index and save
    df = df.reset_index(drop=True)
    df.to_csv(dpath)
    return 

def preprocess_rag(task, dpath):
    '''
    Preprocess RAG dataset for given task.
    We use WikipediaRetriever to get related documents for each entity.
    '''
    print("Preprocessing RAG dataset...\n")

    if os.path.exists(dpath):
        print(f"Find existing RAG dataset: {dpath}\n")
        return pd.read_csv(dpath)

    retriever = WikipediaRetriever()

    df = pd.read_csv(f'../dataset/crafted/{task}.csv')
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df['rag_text'] = pd.NA


    def get_related_docs(retriever, entity):
        entity_doc = retriever.get_relevant_documents(entity)
        if len(entity_doc) > 0:
            return entity_doc[0].metadata['summary']
        else: 
            return 'None'


    cnt = 0
    if task == 'movie':
        for row in df.iterrows():
            doc = get_related_docs(retriever, f"{row[1]['movie_title']} {row[1]['title_year']}")
            if row[1]['director_name'].lower() in doc.lower():
                df.loc[row[0], 'rag_text'] = doc
                cnt += 1
            print("RAG idx: {}\t cnt: {}".format(row[0], cnt))

            
    elif task == 'soccer':
        for row in df.iterrows():
            doc = get_related_docs(retriever, f"{row[1]['long_name']}")
            if row[1]['club_name'].lower() in doc.lower():
                df.loc[row[0], 'rag_text'] = doc
                cnt += 1
            print("RAG idx: {}\t cnt: {}".format(row[0], cnt))

            if int(row[0]) > 1500:
                break
    else: 
        raise ValueError(f'Error: RAG mode for given task {task} is not yet defined.')


    df.to_csv(dpath)
    return df

def get_validation_prompt(task, entity_list, multimodal):
    '''
    Given task and entity, apply question templates to construct validation question prompts.
    The question templates should be predefined w.r.t. FD of each dataset.
    '''


    if task == 'movie':

        # Common entity: Movie Name
        # Entity 1: Released Year
        base_q = f"Do you know the movie {entity_list[0]} released in year {entity_list[1]}?"
        # Entity 2: Director
        e1_q = f"Is the movie {entity_list[0]} released in year {entity_list[1]} directed by {entity_list[2]}?"
        # Entity 3: Country of origin
        e2_q = f"Is the movie {entity_list[0]} released in year {entity_list[1]} produced in {entity_list[3]}?"
        # Entity 4: Genre (animation/non-animation)
        e3_q = f"Is the movie {entity_list[0]} released in year {entity_list[1]} {entity_list[4]} movie?"
        # for multimodal
        e4_q = f"Is the movie poster in the given image the movie poster of the movie {entity_list[0]} released in year {entity_list[1]}?"

        entity_q = [e1_q, e2_q, e3_q]

        if multimodal:
            entity_q.append(e4_q)


    elif task == 'soccer':
        # for soccer, entity_list[0] is player long name, where common entity is player short name (entity_list[1])

        # Common entity: Player Name
        base_q = f"Do you know the soccer player {entity_list[1]}?"
        # Entity 1: Club Name
        e1_q = f"Did the soccer player {entity_list[1]} play for {entity_list[2]} in 2019?"
        # Entity 2: Jersey Number
        e2_q = f"Did the soccer player {entity_list[1]} wear jersey number {entity_list[3]} in 2019?"
        # Entity 3: Nationality
        e3_q = f"Was the soccer player {entity_list[1]} born in {entity_list[4]}?"
        # Entity 4: League Name
        e4_q = f"Did the soccer player {entity_list[1]} participate in leauge named {entity_list[5]} during the year 2019?"
        # for multimodal
        e5_q = f"Did the soccer player {entity_list[1]} play for the club in the image?"

        entity_q = [e1_q, e2_q, e3_q, e4_q]

        if multimodal:
            entity_q.append(e5_q)


    elif task == 'airport':
        # Common entity: Airport Name
        base_q = f"Do you know the {entity_list[0]} airport?"
        # Entity 1: Airport ICAO location indicator
        e1_q = f"Is the ICAO location indicator of the {entity_list[0]} airport {entity_list[1]}?"
        # Entity 2: Airport Latitude
        e2_q = f"Is the latitude of the {entity_list[0]} airport {entity_list[2]}?"
        # Entity 3: Airport Longitude
        e3_q = f"Is the longitude of the {entity_list[0]} airport {entity_list[3]}?"
        # Entity 4: Airport Location
        e4_q = f"Is the country code of {entity_list[0]} airport {entity_list[4]}?"

        entity_q = [e1_q, e2_q, e3_q, e4_q]


    elif task == 'music':
        # Common entity: Track Name
        # Entity 1: Artist Name
        base_q = f'Do you know the song "{entity_list[0]}" by {entity_list[1]}?'
        # Entity 2: Release Date
        e1_q = f'Is the song titled "{entity_list[0]}" by {entity_list[1]} released in {entity_list[2]}?'
        # Entity 3: Genre
        e2_q = f'Is the genre of the song titled "{entity_list[0]}" by {entity_list[1]} {entity_list[3]}?'
        
        entity_q = [e1_q, e2_q]



    elif task == 'book':
        # Common entity: Book Title
        # Entity 1: Author
        base_q = f'Do you know the book "{entity_list[0]}" written by {entity_list[1]}?'
        # Entity 2: Published Month
        e1_q = f'Is the published month of the book titled "{entity_list[0]}" written by {entity_list[1]} {entity_list[2]}?'
        # Entity 3: Published Year
        e2_q = f'Is the published year of the book titled "{entity_list[0]}" written by {entity_list[1]} {entity_list[3]}?'
        # Entity 4: Publisher
        e3_q = f'Is the book titled "{entity_list[0]}" written by {entity_list[1]} published by the publisher named {entity_list[4]}?'

        entity_q = [e1_q, e2_q, e3_q]


    else:
        raise ValueError(f"Error: Validation prompt template for given task {task} is not yet defined.")
    

    # aggregate all questions
    all_q = [base_q] + entity_q
    all_q = list(map(lambda x: "Q: " + x + '\nA:', all_q))

    return all_q

def run_validation(save_dir, task, df, index=0, model=''):
    '''
    Run validation task for given model and task, and save LLM responses to log file.
    '''

    # make log file to save model responses
    log_file_path = get_savename(save_dir, task, model=model)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)


    # get responses by iterating df
    for entity_id in range(len(df)):

        # skip until given index# skip until given index
        # this is useful as API calls can crash and we can resume from the last index
        if(index > entity_id):
            print(f"Skipping idx: {entity_id}...")
            continue
        else:
            print(f"Running idx: {entity_id}...")


        # get entities 
        entity_list = df.iloc[entity_id].tolist()[1:]
        if None in entity_list:
            raise ValueError("Value of Entity is Strange")
        

        # get prompts w.r.t. task and model
        prompts = get_validation_prompt(task, entity_list, multimodal = (model == 'gemini_v') or (model == 'gpt_v'))


        # call model APIs and save responses
        # if API fails, we will retry for 5 times and then skip the entity if it still fails
        id = 0
        fail_cnt = 0
        while (id < len(prompts)):
            prompt_id = id
            prompt = prompts[prompt_id]
            idx = str(entity_id)+'-'+str(prompt_id)
            id +=1

            try:
                if model != 'gemini_v' and model != 'gpt_v':
                    response = run_llm(prompt, tasktype='validate', model=model)
                    
                    if model == 'claude':
                        if 'content' in response:
                            response = response['content'][0]['text']
                        elif 'error' in response:
                            if response['error']['type'] == 'rate_limit_error':
                                print('Rate Limit. Waiting for 1 hour...')
                                time.sleep(3600)
                                id -= 1 
                                continue
                            else:
                                raise ValueError


                else: # multimodal QA
                    if task == 'soccer':
                        img_keyword = entity_list[2] # club image
                    elif task == 'movie': 
                        img_keyword = f'movie {entity_list[0]} poster' # movie poster
                    else:
                        raise ValueError()
                    
                    response = run_llm(prompt, tasktype='validate', model=model, task=task, img_keyword=img_keyword)
                    
                    if model == 'gpt_v':
                        if 'choices' in response:
                            response = response['choices'][0]['message']['content']
                        elif 'error' in response:
                            error_msg = response['error']['message']

                            wait_time = re.findall(r'(\d+)\sseconds', error_msg)[0]
                            print(f"Waiting for {wait_time} seconds...")
                            time.sleep(int(wait_time))
                            id -= 1
                            continue

                        else:
                            raise ValueError 
                        

                    
                    
                    

            except Exception as error:
                fail_cnt +=1
                print(f"{model} ERROR with idx: {idx} (fail_cnt: {fail_cnt}): {error}")

                if fail_cnt > 5:
                    print(f"Too many errors. Skipping idx: {idx}")
                    response = "No. (RESPONSE ERROR)" # This entity will be considered as invalid

                else:
                    id -=1
                    continue

            # this is to avoid API rate limit
            if model =='gemini':
                time.sleep(2)
            else:
                time.sleep(1)

            # save responses to log file
            write_to_file(log_file_path, idx, prompt, response)

            # reset fail count
            fail_cnt = 0
            
        # task-specific break
        if task=='soccer' and entity_id>=1499:
            break
    return
    

def get_qa_prompt(task, entity_list, num_choices, rag=False, multimodal=False):
    '''
    Given task and entity, apply question templates to construct QA question prompts.
    The question templates should be predefined w.r.t. FD of each dataset.
    '''


    if task =='soccer':
        # Common entity: Player Name
        base_q1 = f"Q: What's the inaccurate option about soccer player {entity_list[0]}? Provide an explanation.\n"
        base_q2 = f"Q: What is the false option about soccer player named {entity_list[0]}? Provide an explanation.\n"
        base_q3 = f"Q: What is the wrong option regarding the soccer player {entity_list[0]}? Provide an explanation.\n"
        base_q = [base_q1, base_q2, base_q3]

        # Entity 1: Club Name
        e1_q1 = f"Option 1: Played for {entity_list[1]} in 2019.\n"
        e1_q2 = f"Option 1: He played for {entity_list[1]} in 2019.\n"
        e1_q3 = f"Option 1: He participated in {entity_list[1]} during the year 2019.\n"
        
        if multimodal:
            e1_q1 = f"Option 1: Played for the club in the image in 2019.\n"
            e1_q2 = f"Option 1: He played for the club in the image in 2019.\n"
            e1_q3 = f"Option 1: He participated in the club in the image during the year 2019.\n"

        e1_q = [e1_q1, e1_q2, e1_q3]

        # Entity 2: Jersey Number
        e2_q1 = f"Option 2: Wore jersey number {entity_list[2]} in 2019.\n"
        e2_q2 = f"Option 2: His uniform number was {entity_list[2]} in 2019.\n"
        e2_q3 = f"Option 2: His jersey number during 2019 was {entity_list[2]}.\n"
        e2_q = [e2_q1, e2_q2, e2_q3]
        
        # Entity 3: Nationality
        e3_q1 = f"Option 3: Born in {entity_list[3]}.\n"
        e3_q2 = f"Option 3: He was born in {entity_list[3]}.\n"
        e3_q3 = f"Option 3: His birthplace is {entity_list[3]}.\n"
        e3_q = [e3_q1, e3_q2, e3_q3]

        # Entity 4: League Name
        e4_q1 = f"Option 4: Participated in leauge named {entity_list[4]} during the year 2019.\n"
        e4_q2 = f"Option 4: He played in {entity_list[4]} during the year 2019.\n"
        e4_q3 = f"Option 4: He participated in {entity_list[4]} during the year 2019.\n"
        e4_q = [e4_q1, e4_q2, e4_q3]


        # Aggregate all entity questions w.r.t. number of choices
        entity_q = [e1_q, e2_q, e3_q, e4_q]




    elif task =='movie':
        # Common entity: Movie Name
        # Entity 1: Released Year
        base_q1 = f"Q: What's the inaccurate option about the movie {entity_list[0]} released in year {entity_list[1]}? Provide an explanation.\n"
        base_q2 = f"Q: What is the false option about the movie {entity_list[0]} released in year {entity_list[1]}? Provide an explanation.\n"
        base_q3 = f"Q: What is the wrong option regarding the movie {entity_list[0]} released in year {entity_list[1]}? Provide an explanation.\n"
        
        if multimodal:
            base_q1 = f"Q: What's the inaccurate option about the movie with the movie poster as the given image? Provide an explanation.\n"
            base_q2 = f"Q: What is the false option about the movie with the movie poster as the given image? Provide an explanation.\n"
            base_q3 = f"Q: What is the wrong option regarding the movie with the movie poster as the given image? Provide an explanation.\n"
        
        base_q = [base_q1, base_q2, base_q3]    
        
        # Entity 2: Director
        e1_q1 = f"Option 1: Directed by {entity_list[2]}.\n"
        e1_q2 = f"Option 1: It was directed by {entity_list[2]}.\n"
        e1_q3 = f"Option 1: The name of the Director is {entity_list[2]}.\n"
        e1_q = [e1_q1, e1_q2, e1_q3]

        # Entity 3: Country of origin
        e2_q1 = f"Option 2: Produced in the country {entity_list[3]}.\n"
        e2_q2 = f"Option 2: It was produced in the country {entity_list[3]}.\n"
        e2_q3 = f"Option 2: The movie was produced in the country {entity_list[3]}.\n"
        e2_q = [e2_q1, e2_q2, e2_q3]

        # Entity 4: Genre (animation/non-animation)
        prefix = 'an' if entity_list[4] == 'animation' else 'a'
        e3_q1 = f"Option 3: Has the genre of {entity_list[4]} movie.\n"
        e3_q2 = f"Option 3: It is {prefix} {entity_list[4]} movie.\n"
        e3_q3 = f"Option 3: The movie is {prefix} {entity_list[4]} movie.\n"
        e3_q = [e3_q1, e3_q2, e3_q3]

        entity_q = [e1_q, e2_q, e3_q]


    elif task =='airport':
        # Common entity: Airport Name
        base_q1 = f"Q: What's the inaccurate option about the airport {entity_list[0]}? Provide an explanation.\n"
        base_q2 = f"Q: What is the false option about the airport {entity_list[0]}? Provide an explanation.\n"
        base_q3 = f"Q: What is the wrong option regarding the airport {entity_list[0]}? Provide an explanation.\n"
        base_q = [base_q1, base_q2, base_q3]

        # Entity 1: Airport Shortcode
        e1_q1 = f"Option 1: ICAO Shortcode of the airport is {entity_list[1]}.\n"
        e1_q2 = f"Option 1: The abbreviated form (ICAO) for the airport is {entity_list[1]}.\n"
        e1_q3 = f"Option 1: The ICAO shortcode for the airport is the same with {entity_list[1]}.\n"
        e1_q = [e1_q1, e1_q2, e1_q3]

        # Entity 2: Airport Latitude
        e2_q1 = f"Option 2: Latitude of the airport is {entity_list[2]}.\n"
        e2_q2 = f"Option 2: The latitude of the airport is {entity_list[2]}.\n"
        e2_q3 = f"Option 2: The airport is located at {entity_list[2]} latitude.\n"
        e2_q = [e2_q1, e2_q2, e2_q3]

        # Entity 3: Airport Longitude
        e3_q1 = f"Option 3: Longitude of the airport is {entity_list[3]}.\n"
        e3_q2 = f"Option 3: The longitude of the airport is {entity_list[3]}.\n"
        e3_q3 = f"Option 3: The airport is located at {entity_list[3]} longitude.\n"
        e3_q = [e3_q1, e3_q2, e3_q3]

        # Entity 4: Airport Location
        e4_q1 = f"Option 4: Country code of the airport is {entity_list[4]}.\n"
        e4_q2 = f"Option 4: The country code of the airport is {entity_list[4]}.\n"
        e4_q3 = f"Option 4: The airport has a country code of {entity_list[4]}.\n"
        e4_q = [e4_q1, e4_q2, e4_q3]

        entity_q = [e1_q, e2_q, e3_q, e4_q]


    elif task =='music':
        # Common entity: Track Name
        # Entity 1: Artist Name
        base_q1 = f"Q: What's the inaccurate option about the song {entity_list[0]} of the artist {entity_list[1]}? Provide an explanation.\n"
        base_q2 = f"Q: What is the false option about the song {entity_list[0]} of the artist {entity_list[1]}? Provide an explanation.\n"
        base_q3 = f"Q: What is the wrong option regarding the song {entity_list[0]} of the artist {entity_list[1]}? Provide an explanation.\n"
        base_q = [base_q1, base_q2, base_q3]

        # Entity 2: Release Date
        e1_q1 = f"Option 1: The song was released in {entity_list[2]}.\n"
        e1_q2 = f"Option 1: The song was released in the year {entity_list[2]}.\n"
        e1_q3 = f"Option 1: The released year of the song is {entity_list[2]}.\n"
        e1_q = [e1_q1, e1_q2, e1_q3]

        # Entity 3: Genre
        e2_q1 = f"Option 2: The genre of the song is {entity_list[3]}.\n"
        e2_q2 = f"Option 2: The song belongs to {entity_list[3]} genre.\n"
        e2_q3 = f"Option 2: The song is categorized as {entity_list[3]} genre.\n"
        e2_q = [e2_q1, e2_q2, e2_q3]

        entity_q = [e1_q, e2_q]


    elif task =='book':
        # Common entity: Book Title
        # Entity 1: Author
        base_q1 = f"Q: What's the inaccurate option about the book titled {entity_list[0]}, written by an author named {entity_list[1]}? Provide an explanation.\n"
        base_q2 = f"Q: What is the false option about the book titled {entity_list[0]}, written by an author named {entity_list[1]}? Provide an explanation.\n"
        base_q3 = f"Q: What is the wrong option regarding the book titled {entity_list[0]}, written by an author named {entity_list[1]}? Provide an explanation.\n"
        base_q = [base_q1, base_q2, base_q3]

        # Entity 2: Published Month
        e1_q1 = f"Option 1: Published month of the book is {entity_list[2]}.\n"
        e1_q2 = f"Option 1: The book was published in the month {entity_list[2]}.\n"
        e1_q3 = f"Option 1: The published month of the book is {entity_list[2]}.\n"
        e1_q = [e1_q1, e1_q2, e1_q3]

        # Entity 3: Published Year
        e2_q1 = f"Option 2: Published year of the book is {entity_list[3]}.\n"
        e2_q2 = f"Option 2: The book was published in the year {entity_list[3]}.\n"
        e2_q3 = f"Option 2: The published year of the book is {entity_list[3]}.\n"
        e2_q = [e2_q1, e2_q2, e2_q3]

        # Entity 4: Publisher Name
        e3_q1 = f"Option 3: Published by the publisher named {entity_list[4]}.\n"
        e3_q2 = f"Option 3: The book was published by the publisher named {entity_list[4]}.\n"
        e3_q3 = f"Option 3: The publisher of the book is the publisher named {entity_list[4]}.\n"
        e3_q = [e3_q1, e3_q2, e3_q3]

        entity_q = [e1_q, e2_q, e3_q]

    else:
        raise ValueError("Error: QA prompt template for given task is not yet defined.")


    # truncate entity questions w.r.t. number of choices
    entity_q = entity_q[:num_choices]



    # Aggregate all questions
    for i, q in enumerate(base_q):
        for e_q in entity_q:
            q += e_q[i]
        q += '\nA: '
        base_q[i] = q


    # if rag, add rag prompt
    # rag prompt should be the last value of entity_list
    if rag:
        base_q = list(map(lambda x: entity_list[-1] + '\n\n' + x, base_q))


    return base_q

def run_qa(save_dir, task, df, num_choices, mixed, demo=False, rag=False, index = 0, model = ''):
    '''
    Run QA task for given model and task, and save LLM responses to log file.

    Args:
    - mixed: float, the probability of mixed question (i.e. all true question)
    - demo: bool, whether to use pre-defined demos for few-shot setting
    - rag: bool, whether to run RAG setting
    - index: int, the entity index to start running QA task.
    '''

    # get log file path
    log_file_path = get_savename(save_dir= save_dir, task = task, num_choices= num_choices, mixed = mixed, demo = demo, rag = rag, model = model)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # load original dataframe for rag
    # we need this to precisely compare non-rag vs rag
    if rag:
        orig_df = pd.read_csv(log_file_path.replace('_rag.log', '.csv'))

    # load pre-defined demos for few-shot
    if demo:
        demo_path = f'../dataset/demo/{task}.txt'
        if not os.path.exists(demo_path):
            raise ValueError(f"Error: Given demo path ({demo_path}) does not exist.")

        with open(demo_path) as f:
            demo_prompt = f.read()


    # iterate over dataframe
    for i in df.index:

        # skip until given index
        # this is useful as API calls can crash and we can resume from the last index
        if(index > i):
            print(f"Skipping idx: {i}...")
            continue
        else:
            print(f"Running idx: {i}...")
            mixed_all_true_flag = False
            if mixed > 0:
                mixed_prob = random.random()
                if mixed_prob <= mixed:  
                    mixed_all_true_flag = True


        if rag: # as rag can fail, we only compare entities in originalDB which successfully have rag in ragDB
            prompts = []
            # get the same prompts w.r.t. entity index in originalDB
            for rows in orig_df[orig_df['entity_idx'] == i].iterrows():
                gold_answer = rows[1]['gold_answer']
                gold_entity = rows[1]['gold_entity']
                prompt = df.loc[i].tolist()[-1] + '\n\n' + rows[1]['question']
                for option in ast.literal_eval(rows[1]['choices']):
                    prompt += '\n{}'.format(option)
                prompt += '\n\nA: '
                prompts.append(prompt)
        else:
            # get entities, where the first element is the common entity  
            entity_list = df.iloc[i].tolist()[1:]
            if task == 'soccer': 
                # for soccer, we use first entity (long name) only for rag
                entity_list = entity_list[1:]
            if None in entity_list:
                raise ValueError("Value of Entity is Strange")
            

            # generate answer
            if mixed_all_true_flag:
                gold_entity = 'NONE'
                gold_answer = "There is no answer"

                           
            else:
                # random select answer from num_choices
                gold_answer = random.randint(1, num_choices)


                # assign gold entity 
                if task in ['movie', 'music', 'book']: 
                    # for these tasks, option1 matches with entity2
                    gold_entity = entity_list[gold_answer+1]
                else:
                    # for others, option1 matches with entity1
                    gold_entity = entity_list[gold_answer]

                # get false value
                while True:

                    if task == 'airport':
                        false_value = random.choice(df.iloc[:, gold_answer+1])
                    
                    else: # other tasks matches option1 with entity2
                        
                        # since this answer is binarized, save time for movie task
                        if task == 'movie' and gold_answer == 3:
                            false_value = 'non_animation' if entity_list[4] == 'animation' else 'animation'
                        else:
                            false_value = random.choice(df.iloc[:, gold_answer+2].tolist())
                    


                    # assert false value is not the same with gold entity
                    if false_value != gold_entity:
                        break

                # replace the entity list with false value
                if task in ['movie', 'music', 'book']: 
                    # for these tasks, option1 matches with entity2
                    entity_list[gold_answer+1] = false_value       
                else:
                    entity_list[gold_answer] = false_value
                
                # for airport task, gold answer 4 is for country code
                # we consider country name (entity_list[5]) as another valid gold entity
                if task == 'airport' and gold_answer in [4]:
                    gold_entity = gold_entity + ', ' + entity_list[5]
            
            # get prompts
            prompts = get_qa_prompt(task, entity_list, num_choices, rag, multimodal=(model=='gemini_v' or model=='gpt_v'))

        # if few-shot setting, add manually generated demos
        if demo:
            for j, prompt in enumerate(prompts):
                prompts[j] = demo_prompt + '\n\n\n' + prompt

        # run gpt-3.5 and gpt-4
        id = 0
        fail_cnt = 0
        while (id < len(prompts)):
            prompt_id = id
            prompt = prompts[prompt_id]
            idx = str(i)+'-'+str(prompt_id)
            id +=1 

            try:
                if mixed > 0:
                    response = run_llm(prompt, tasktype='multiqa_hint', model=model)
                elif rag:
                    response = run_llm(prompt, tasktype='multiqa_rag', model=model)
                else:
                    if model != 'gemini_v' and model != 'gpt_v':
                        response = run_llm(prompt, tasktype='multiqa', model=model)

                        if model == 'claude':
                            if 'content' in response:
                                response = response['content'][0]['text']
                            elif 'error' in response:
                                if response['error']['type'] == 'rate_limit_error':
                                    print('Rate Limit. Waiting for 1 hour...')
                                    time.sleep(3600)
                                    id -= 1 
                                    continue
                                else:
                                    raise ValueError
                        
                    else:
                        if task == 'soccer':
                            img_keyword = entity_list[1] # club image
                        elif task == 'movie': 
                            img_keyword = f'movie {entity_list[0]} poster' # movie poster
                        else:
                            raise ValueError()
                        
                        response = run_llm(prompt, task = task, img_keyword=img_keyword, tasktype='multiqa', model = model)

                        if model == 'gpt_v':
                            if 'choices' in response:
                                response = response['choices'][0]['message']['content']
                            elif 'error' in response:
                                error_msg = response['error']['message']

                                wait_time = re.findall(r'(\d+)\sseconds', error_msg)[0]
                                print(f"Waiting for {wait_time} seconds...")
                                time.sleep(int(wait_time))
                                id -= 1
                                continue

                            else:
                                raise ValueError

                        # return response.json()['choices'][0]['message']['content']

            except Exception as error:
                fail_cnt +=1
                print(f"{model} ERROR with idx: {idx} (fail_cnt: {fail_cnt}) : {error}")

                if fail_cnt > 5:
                    print(f"Too many errors. Skipping idx: {idx}")
                    response = "RESPONSE ERROR"
 
                else:
                    id -=1
                    continue
    
            if model =='gemini':
                time.sleep(2)
            else:
                time.sleep(1)

            write_to_file(log_file_path, idx, prompt, response, gold_answer, gold_entity)
            fail_cnt = 0 
            

        # task-specific break
        if task=='soccer' and i==1499:
            break






def parse_args():
    '''
    Config
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices = ['movie', 'soccer', 'airport', 'music', 'book'], required = True)
    parser.add_argument('--tasktype', "-t", type=str, choices=['validate', 'multiqa'], default= 'multiqa')
    parser.add_argument("--index", type=int, default = 0)
    parser.add_argument("--mixed", type=float, default=0.0, help='if set to [0, 1), mix all_true option for given probability')
    parser.add_argument("--random_seed", "-s", type=int, default=1116)
    parser.add_argument("--model", type=str, choices = ['gpt35', 'gpt4', 'mistral', 'llama', 'gemini', 'gemini_v', 'claude', 'gpt_v'], required=True)
    parser.add_argument("--rag", action='store_true', default=False)
    parser.add_argument("--demo", action="store_true", default=False)
    
    args = parser.parse_args()
    
    return args

def main():
    '''
    Main function to run QA task
    '''

    # config
    print("================= Config ==========================")
    args = parse_args()
    task = args.task.strip()
    tasktype = args.tasktype
    idx = args.index
    mixed = args.mixed
    random_seed = args.random_seed
    model = args.model
    rag = args.rag
    demo = args.demo
    for k, v in args.__dict__.items():
        print(f"{k}: {v}")
    print("===================================================")
    data_dir = '../dataset'
    save_dir = '../results'

    # set random seed
    random.seed(random_seed)



    # set maximum number of choices
    MAX_CHOICES = {'movie': 3, 'soccer': 4, 'airport': 4, 'music': 2, 'book': 3}
    num_choices = MAX_CHOICES[task]


    # Pre-processed database name
    processed_db_path = os.path.join(data_dir, 'crafted', f'{task}.csv')
    print(f"Checking pre-processed database: {processed_db_path}...\n")

    # if not found, pre-process the database
    if not os.path.exists(processed_db_path):

        # the processed database will be stored in {data_dir}/crafted  
        os.makedirs(os.path.join(data_dir, 'crafted'), exist_ok=True)

        # original database (pre-defined)
        # we use airport data via python "airportsdata" package
        task_to_dbpath = {'movie': f'{data_dir}/original/movie_data.csv',
                        'soccer': f'{data_dir}/original/players_20.csv',  
                        'music': f'{data_dir}/original/music.csv',
                        'book': f'{data_dir}/original/BooksDataset.csv'}
        
        # preprocess database to get entity/relations
        print(f"Not found. Pre-processing database: {task_to_dbpath[task]}...\n")
        preprocess(task_to_dbpath, task, processed_db_path, random_seed)


    # load preprocessed database
    print(f"Found pre-processed dataset. Loading database!!")
    df = pd.read_csv(processed_db_path)

    # if rag, preprocess ragDB
    if rag: 
        processed_db_path = processed_db_path.replace(".csv", "_rag.csv")
        df = preprocess_rag(task, processed_db_path)
        df.dropna(inplace=True)

    # run validation or QA task 
    if tasktype =='validate':
        validated_dir = os.path.join(data_dir, 'validated')
        os.makedirs(validated_dir, exist_ok=True)
        print(f'Starting Validation : {task}!!')
        run_validation(save_dir=validated_dir, task=task, df = df, index=idx, model=model)

    else:
        print(f"Starting Experiment! {task}")
        run_qa(save_dir=save_dir, task=task, df = df, num_choices = num_choices, mixed = mixed, index=idx, model=model, rag = rag, demo = demo)

    return





if __name__ == "__main__":
    main()