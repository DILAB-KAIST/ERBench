import pandas as pd
import numpy as np
import os
import base64

from mimetypes import guess_type
from subprocess import call
import pickle
import re
import requests
import airportsdata as ap
import PIL.Image
from urllib.parse import urlparse, parse_qs
import time
#import nltk
#from nltk import ne_chunk, word_tokenize, pos_tag
import unicodedata
import google.generativeai as genai
from google.ai import generativelanguage as glm
from bs4 import BeautifulSoup
import langid
from langchain_community.retrievers import WikipediaRetriever
import wikipediaapi
from openai import AzureOpenAI
import anthropic


'''
FOR OPENAI VERSION 0.28.1
import openai
'''

def replace_unlatin_characters(text):
    return ''.join(
        unicodedata.normalize('NFKD', char).encode('ASCII', 'ignore').decode('utf-8')
        for char in text
    )
def remove_special_characters(input_string):
    # Define a regex pattern to match any character that is not alphanumeric or whitespace
    pattern = re.compile(r'[^a-zA-Z0-9\s]')
    
    # Use the sub() method to replace matched patterns with an empty string
    result_string = pattern.sub('', input_string)
    result_string = result_string.replace('  ', ' ')
    return result_string
def is_english(text):
    lang, confidence = langid.classify(text)
    return lang=='en' and confidence >0.5
def preprocess_soccer():
    if os.path.exists('./fifa_data/crafted/fifa_df.csv'):
        return
    fifa_df = pd.read_csv('./fifa_data/players_20.csv')
    fifa_df = fifa_df[['short_name', 'club_name','club_jersey_number', 'nationality_name']]
    fifa_df.columns = ['common_entity', 'entity1','entity3',  'entity2']
    fifa_df.dropna(inplace=True)
    fifa_df= fifa_df.iloc[0:1500]
    fifa_df = fifa_df.reset_index()
    fifa_df.to_csv('./fifa_data/crafted/fifa_df.csv')
def preprocess_movie_foreign(subtask):
    if not os.path.exists(f'./movie_data/crafted/movie_{subtask}.csv'):
        df= pd.read_csv('./movie_data/crafted/movie_df.csv', index_col=0)
        df.drop('index',axis=1, inplace=True)
        new_df = pd.DataFrame(columns=['common_entity','entity3','entity2','entity1', 'entity4'])
        exclude_list = []
        for i in range(len(df)):
            director = df.loc[i, 'entity2']
            dir_split = director.split(' ')
            joined_name = '_'.join(dir_split)
            url = f'https://en.wikipedia.org/wiki/{joined_name}'

            # Fetch the HTML content of the page
            response = requests.get(url)
            html_content = response.content

            # Parse the HTML content
            soup = BeautifulSoup(html_content, 'html.parser')

            # Find the specific table cell (td) containing the birth date and place
            infobox = soup.find('table', {'class': 'infobox'})
            if infobox:
                birth_data = infobox.find('span', {'class': 'bday'})
                birthplace_data = infobox.find('div', {'class': 'birthplace'})

                # Extracting text if found
                birth_date = birth_data.text if birth_data else 'Birth date not found'
                birthplace = birthplace_data.text.strip() if birthplace_data else 'Birthplace not found'

                #print(f"Birth Date: {birth_date}")
                #print(f"Birthplace: {birthplace}")
                birth_year = birth_date.split('-')[0]
                birth_city = birthplace.split(',')[0]
                #print(f"Birth Date: {birth_year}")
                #print(f"Birthplace: {birth_city}")
                x = df.loc[i].to_list()
                if subtask=='city':
                    x.append(birth_city)
                elif subtask=='year':
                    x.append( birth_year)
                else:
                    raise ValueError("UNDEFINED SUBTASK")
                new_df.loc[len(new_df)] = x
                #print(new_df)
                time.sleep(0.5)

            else:
                #print('skip')
                x = df.loc[i].to_list()
                x.append('nan')
                new_df.loc[len(new_df)] = x
                exclude_list.append(i)
        new_df.to_csv(f'./movie_data/crafted/movie_{subtask}.csv')
    df = pd.read_csv(f'./movie_data/crafted/movie_{subtask}.csv', index_col=0)
    df = df[df['entity4'] != 'Birth date not found']
    df.dropna(inplace=True)
    df.to_csv(f'./movie_data/crafted/movie_{subtask}_final.csv')
    

def preprocess_soccer_key():
    if os.path.exists('./fifa_data/crafted/fifa_key_df.csv'):
        return
    fifa_df = pd.read_csv('./fifa_data/players_20.csv')
    fifa_df = fifa_df[['short_name', 'club_name', 'club_jersey_number', 'nationality_name', 'nation_jersey_number']]
    fifa_df.columns = ['common_entity', 'entity1','entity2',  'entity3', 'entity4']
    fifa_df.dropna(inplace=True)
    fifa_df.reset_index(inplace=True)
    fifa_df.to_csv('./fifa_data/crafted/fifa_key_df.csv')

def preprocess_multihop_soccer_olympic():
    if os.path.exists('./soccer_olympic/crafted/total_df.csv'):
        return
    if not os.path.exists('./soccer_olympic/crafted/olympic.csv'):

        olympic_df = pd.read_csv('./soccer_olympic/olympic_df.csv')
        
        ol_df = olympic_df[['game_slug', 'game_location', 'game_year', 'game_season']]
        ol_df.columns = ['city', 'country', 'year', 'season']
        ol_df = ol_df[ol_df['season']=='Summer']
        ol_df = ol_df.reset_index()
        for i in range(len(ol_df)):

            ol_df.loc[i, 'city'] = ol_df.loc[i, 'city'].split(str(ol_df.loc[i, 'year']))[0].replace('-','')
        ol_df = ol_df.groupby('city')['year'].agg(list).reset_index()
        
    
        for i in range(len(ol_df)):
            t_l = ol_df.loc[i, 'year']

            year=''
            for value in t_l:
                year = year+str(value)+','
            ol_df.loc[i,'year'] = year
        ol_df.to_csv('./soccer_olympic/crafted/olympic.csv')
    soccer = pd.read_csv('./soccer_olympic/crafted/fifa_df.csv', index_col=0)
    soccer = soccer.drop(['entity3', 'entity2', 'index'], axis=1)
    city_club_df = pd.read_csv('./soccer_olympic/crafted/club_city_debug.csv',index_col=0)
    soccer.columns = ['player_name', 'club_name']
    for i in range(len(soccer)):
        soccer.loc[i,'city'] = city_club_df.loc[i,'city_name'].lower()
    olympic_year = pd.read_csv('./soccer_olympic/crafted/olympic.csv', index_col=0)
    soccer = soccer.merge(olympic_year, how='left', on='city')
    #player_name,club,city,year
    soccer.fillna(' ',inplace=True)
    soccer.columns = ['entity1', 'entity2', 'common_entity', 'entity3']
    soccer.to_csv('./soccer_olympic/crafted/total_df.csv')

    


def preprocess_airport():
    if os.path.exists('./airport_data/crafted/airport_df.csv'):
        return
    airports = ap.load()
    airport_df = pd.DataFrame.from_dict(airports).transpose()
    airport_df = airport_df[['name', 'lat', 'lon']]
    airport_df.columns = ['common_entity', 'entity1', 'entity2']
    #airport_df['common_entity'] = airport_df['common_entity'].str.replace('Airport','').strip()
    airport_df['common_entity'] = airport_df['common_entity'].apply(lambda x: x.replace('Airport','').strip())
    random_seed=1116
    airport_df = airport_df.sample(frac=1, random_state=random_seed)
    airport_df = airport_df.iloc[:1500]
    airport_df.to_csv('./airport_data/crafted/airport_df.csv')

def preprocess_book():
    if os.path.exists('./book_data/crafted/book_df.csv'):
        return
    df = pd.read_csv('./book_data/book.csv')
    df['Authors'] = df['Authors'].str.replace('By ', '')
    df['Authors'] = df['Authors'].str.replace('By', '')
    df['Title'] = df['Title'].str.replace('"', '')
    df.dropna(subset=['Authors'], inplace=True)
    df.dropna(subset=['Title'], inplace=True)
    df.dropna(subset=['Publish Date'],inplace=True)
    df['tmp'] = df['Publish Date'].str.split(',')
    df['MonthDate'] = df['tmp'].apply(lambda x: x[1].strip().split(' ')[0] if len(x) > 1 else None)
    df['Year'] = df['tmp'].apply(lambda x: int(x[2]) if len(x) > 2 else None)
    df =df[df['Year']<2022]
    df['Publish Date'] = df['tmp'].apply(lambda x: f"{x[1].strip().split(' ')[0].strip()}, {x[2].strip()}")
    df = df[['Title', 'Authors', 'Publish Date']]
    special_characters_pattern = re.compile(r'[!@#$%^&*()_+{}\[\];,<>.~\\/-]')
    included_substrings = ['\(', 'Edition', 'edition']
    exclude_pattern = f'{special_characters_pattern.pattern}|{"|".join(map(re.escape, included_substrings))}'
    df = df[~df['Title'].str.contains(exclude_pattern, regex=True)]
    df.columns = ['common_entity', 'entity1', 'entity2']
    random_seed=1116
    df = df.sample(frac=1, random_state=random_seed)
    df = df.iloc[:1500]
    df.to_csv('./book_data/crafted/book_df.csv')
def preprocess_beer():
    if os.path.exists('./beer_data/crafted/beer_df.csv'):
        return
    beer_df = pd.read_csv('./beer_data/beer.csv')
    beer_df = beer_df[['beer_name', 'brewery_name', 'beer_style']]
    beer_df.columns = ['common_entity', 'entity1', 'entity2']
    beer_df.dropna(inplace=True)
    beer_df= beer_df[~beer_df.duplicated(keep=False)]
    beer_df = beer_df.sample(frac=1)
    beer_df = beer_df.iloc[:1500]
    beer_df = beer_df.reset_index()
    beer_df.to_csv('./beer_data/crafted/beer_df.csv')
def preprocess_music():
    if os.path.exists('./music_data/crafted/music_df.csv'):
        return
    random_seed= 1146
    music_df = pd.read_csv('./music_data/music.csv')
    music_df = music_df[['artist_name', 'track_name', 'release_date']]
    music_df.dropna(inplace=True)
    music_df = music_df[music_df['artist_name']!='x']
    music_df = music_df[music_df['track_name']!='x']
    music_df.columns = ['common_entity', 'entity1', 'entity2']
    music_df = music_df.sample(frac=1, random_state= random_seed)
    music_df = music_df.iloc[:1500]
    music_df = music_df.reset_index()
    music_df.to_csv('./music_data/crafted/music_df.csv')



def preprocess_movie(type='normal'):
    if type =='wikidata':
        if os.path.exists('./movie_data/crafted/movie_df_wikidata.csv'):
            return
        if not os.path.exists('./movie_data/crafted/movie_list.pkl'):
            print('1'+ str(os.path.exists('./movie_data/crafted/movie_list.pkl')))
            call(["python", "movie_crawl.py"])
        if not os.path.exists('./movie_data/crafted/targeted_dict_list'):
            print('2'+ str(os.path.exists('./movie_data/crafted/targeted_dict_list')))
            call(["python", "crawl_test.py"])
        with open('./movie_data/crafted/targeted_dict_list','rb') as f:
            movie_list =  pickle.load(f)
                    
        movie_df = pd.DataFrame(columns = ['common_entity', 'entity1','entity2'])

        for movie in movie_list:
            if movie['screenwriters'] =='not_found' or movie['casts'] =='not_found':
                continue
            if 'film' not in movie['description']:
                continue
            movie_df.loc[len(movie_df.index)] = [movie['title'], movie['casts'][0], movie['screenwriters'][0]]

        movie_df.to_csv('./movie_data/crafted/movie_df_wikidata.csv')
    elif type =='normal':
        if os.path.exists('./movie_data/crafted/movie_df.csv'):
            return
        movie_df = pd.read_csv('./movie_data/movie.csv')
        movie_df = movie_df[['movie_title','title_year', 'director_name', 'actor_1_name']]
        movie_df.columns = ['common_entity', 'entity3' ,'entity2', 'entity1']
        movie_df.dropna(inplace=True)
        movie_df = movie_df.iloc[:1500]
        movie_df['common_entity'] = movie_df['common_entity'].str.strip()
        movie_df = movie_df.reset_index()
        movie_df.to_csv('./movie_data/crafted/movie_df.csv')

def preprocess(task):
    if task=="movie" or task=='multimodal_movie':
        preprocess_movie()
    elif task =='soccer':
        preprocess_soccer()
    elif task =='airport':
        preprocess_airport()
    elif task =='music':
        preprocess_music()
    elif task =='soccer_olympic':
        preprocess_multihop_soccer_olympic()
    elif task=='soccer_key':
        preprocess_soccer_key()
    elif task=='movie_foreign_city':
        preprocess_movie_foreign('city')
    elif task=='movie_foreign_year':
        preprocess_movie_foreign('year')
    elif task=='multimodal_soccer':
        preprocess_soccer()
    elif task=='book':
        preprocess_book()
    else:
        raise ValueError("NOT YET DEFINED!!")


def get_data_url(image_path):
    # Function to encode a local image into data URL 
    def local_image_to_data_url(image_path):
        # Guess the MIME type of the image based on the file extension
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'  # Default MIME type if none is found

        # Read and encode the image file
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

        # Construct the data URL
        return f"data:{mime_type};base64,{base64_encoded_data}"

    # Example usage
    #image_path = f'./images/{}'
    data_url = local_image_to_data_url(image_path)
    return data_url

def run_gpt35(prompt, tasktype = 'normal',cot_q=[], cot_a=[], cot=False, nonbinary=False):
    '''
    if openai version ==0.28.1
    openai.api_type = "azure"
    openai.api_base =''
    openai.api_version = "2023-05-15"
    openai.api_key =''
    '''
    #NEED TO MODIFY THIS FOR RELEASE
    client = AzureOpenAI( 
        api_version = "2023-05-15",
        azure_endpoint= "",
        api_key=""
    )
    if tasktype=='normal':
        system_msg = "Answer the following question in yes or no, and then explain why. Say 'unsure' if you don't know and then explain why."
        #system_msg = "The first few passages are hints, that may not contain all relevant information. Answer the following question with your own knowledge getting help from the first few passages if possible. Answer the question in the last sentence in yes or no, and then explain why. Say 'unsure' if you don't know and then explain why."
    elif tasktype =='validate':
        system_msg = "Answer the following question in yes or no. Be concise"
    if cot: 
        assert(len(cot_q)==len(cot_a))
        msg = [
            {"role": "system", "content": system_msg}
        ]
        for q,a in zip(cot_q, cot_a):
            msg.append({'role': 'user', 'content': q})
            msg.append({'role': 'assistant', 'content': a})
        msg.append({"role": "user", "content":prompt})

    else:
        if nonbinary:
            system_msg = "Answer the question in one or two words and then explain why. Say 'unsure' if you don't know and then explain why."
            msg = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content":prompt},
            ]
        else:
            msg = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content":prompt},
                ]
    '''
    If openai version is 0.28.1 use this module
    response = openai.ChatCompletion.create(
        engine="gpt3_5_2",
        temperature = 0,
        messages= msg
    )

    return response['choices'][0]['message']['content']
    '''
    completion = client.chat.completions.create(
        model="gpt3_5_2",
        messages = msg,
        temperature = 0)
    return completion.choices[0].message.content
    


def run_gpt4(prompt, tasktype='normal',cot_q=[], cot_a=[], cot=False, nonbinary=False):
    '''
        if openai version ==0.28.1
        openai.api_type = "azure"
        openai.api_base =''
        openai.api_version = "2023-05-15"
        openai.api_key =''
    '''
    #NEED TO MODIFY THIS PART BEFORE RELEASE
    client = AzureOpenAI( 
        api_version = "", #WRITE THE CORRECT VERSION
        azure_endpoint=  "", #WRITE YOUR ENDPOINT
        api_key="" #WRITE YOUR KEY
    )
    if tasktype=='normal':
        system_msg = "Answer the following question in yes or no, and then explain why. Say 'unsure' if you don't know and then explain why."
        #system_msg = "The first few passages are hints, that may not contain all relevant information. Answer the following question with your own knowledge getiing help from the first few passages if possible. Answer the question in the last sentence in yes or no, and then explain why. Say 'unsure' if you don't know and then explain why."
    elif tasktype =='validate':
        system_msg = "Answer the following question in yes or no. Be concise"
    if cot: 
        assert(len(cot_q)==len(cot_a))
        msg = [
            {"role": "system", "content": system_msg}
        ]
        for q,a in zip(cot_q, cot_a):
            msg.append({'role': 'user', 'content': q})
            msg.append({'role': 'assistant', 'content': a})
        msg.append({"role": "user", "content":prompt})
    else:
        if nonbinary:
            system_msg = "Answer the question in one or two words and then explain why. Say 'unsure' if you don't know and then explain why."
            msg = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content":prompt},
            ]
        msg = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content":prompt},
            ]
    '''
        if openai version ==0.28.1
    response = openai.ChatCompletion.create(
        engine="", #WRITE THE ENGINE
        temperature = 0,
        messages = msg,
    )
    #print(msg)


    return response['choices'][0]['message']['content']
    '''
    completion = client.chat.completions.create(
        model="", #WRITE THE ENGINE
        messages = msg, 
        temperature = 0
    )
    return completion.choices[0].message.content

def run_finetuned_gpt35(prompt, tasktype='normal'):
    '''
    if openai version ==0.28.1
    openai.api_type = "azure"
    openai.api_base = ''
    openai.api_version = "2023-05-15"
    openai.api_key = ''
    '''


    client = AzureOpenAI( 
        api_version = "2023-05-15",
        azure_endpoint= '',
        api_key=''
    )
    if tasktype=='normal':
        system_msg = "Answer the following question in yes or no, and then explain why. Say 'unsure' if you don't know and then explain why."
    elif tasktype =='validate':
        system_msg = "Answer the following question in yes or no. Be concise"


    msg = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content":prompt},
        ]
    '''
    
    response = openai.ChatCompletion.create(
        engine="", #WRITE THE ENGINE
        temperature = 0,
        messages = msg,
    )
    #print(msg)

    
    return response['choices'][0]['message']['content']
    '''
    completion = client.chat.completions.create(
        model="", #WRITE THE ENGINE
        messages=msg,
        temperature=0
    )
    return completion.choices[0].message.content

def run_finetuned_gpt35_2(prompt, tasktype='normal'):
    '''
    if openai version ==0.28.1
    openai.api_type = "azure"
    openai.api_base = ''
    openai.api_version = "2023-05-15"
    openai.api_key = ''
    '''
    if tasktype=='normal':
        system_msg = "Answer the following question in yes or no, and then explain why. Say 'unsure' if you don't know and then explain why."
    elif tasktype =='validate':
        system_msg = "Answer the following question in yes or no. Be concise"


    msg = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content":prompt},
        ]
    '''
    OPENAI VERSION==0.28.1
    response = openai.ChatCompletion.create(
    engine="gpt_ft_soccer",
    temperature = 0,
    messages = msg,
    )
    #print(msg)


    return response['choices'][0]['message']['content']
    '''
    completion = client.chat.completions.create(
            model="gpt_ft_soccer",
            messages=msg,
            temperature=0
        )
    return completion.choices[0].message.content

def check_validation_file(task, model):


    directory = f'./results/{model}/validate_{task}.log'
    if os.path.exists(directory):
        with open(directory,'r') as f:
            corpus = f.read()
            pattern = r'(\d+[a-z]{2}\squestion)\n(Q:.*?)(A:.*?)(?=\n\d|$)'
            matches = re.findall(pattern, corpus, re.DOTALL)

            result = []
            for match in matches:
                index_question = match[0]
                question = match[1].strip()
                answers =  match[2].strip()
                result.append([index_question, question, answers])

        cnt_idx=[]
        for idx_text, q, a in result:
            pattern_no = r'\bno\b'
            if re.search(pattern_no, a, re.IGNORECASE):
                pattern_movie = r'Do you know the movie (.*?)\?'

                match = re.search(pattern_movie, q)
                if match:
                    movie_title = match.group(1)
                    if re.search(pattern_no, movie_title, re.IGNORECASE):
                        a = a.replace(movie_title, '')
                
                if re.search(pattern_no, a, re.IGNORECASE):
                    idx = int(idx_text.split('th question')[0])
                    cnt_idx.append(idx)
        return cnt_idx
    else:
        raise ValueError('Path Not Found')
    
def find_intersection():
    models = ['gpt35', 'gpt4', 'llama', 'gemini', 'claude','mistral']
    files = ['soccer', 'music', 'airport', 'book', 'movie']
    tot =0
    fin_index = {}
    def union(lst1, lst2):
        final_list = list(set(lst1) | set(lst2))
        return final_list
    for file in files:
        cnt_idx_for_task = []
        for model in models:
            index = []
            result = []
            with open(f'./results/{model}/validate_{file}.log', 'r') as f:

                corpus = f.read()
                pattern = r'(\d+[a-z]{2}\squestion)\n(Q:.*?)(A:.*?)(?=\n\d|$)'
                matches = re.findall(pattern, corpus, re.DOTALL)


                for match in matches:
                    index_question = match[0]
                    question = match[1].strip()
                    answers =  match[2].strip()
                    result.append([index_question, question, answers])

            cnt_idx=[]
            for idx_text, q, a in result:
                pattern_no = r'\bno\b'
                if re.search(pattern_no, a, re.IGNORECASE):
                    pattern_movie = r'Do you know the movie (.*?)\?'

                    match = re.search(pattern_movie, q)
                    if match:
                        movie_title = match.group(1)
                        if re.search(pattern_no, movie_title, re.IGNORECASE):
                            a = a.replace(movie_title, '')
                    
                    if re.search(pattern_no, a, re.IGNORECASE):
                        idx = int(idx_text.split('th question')[0])
                        cnt_idx.append(idx)
            cnt_idx_for_task = union(cnt_idx_for_task,cnt_idx)
            if file=='movie':
                cnt_idx_for_task = union(cnt_idx_for_task, [ 1100, 385, 1074, 97, 581, 144, 781, 391, 1307, 136, 83, 185, 645, 652, 1123])
        fin_index[file] = cnt_idx_for_task
    return fin_index
def check_purity(file_path):
    df = pd.read_csv(file_path, index_col=0)
    tot_len = 887#int(len(df)/3)
    print(tot_len)
    for i in range(tot_len):
        tmp_df = df[df['entity_idx']==i]
        if len(tmp_df) !=3:
            print (i)
def check_proxy_cardinality(task, entity1="NULL", entity2='NULL', entity3='NULL', entity4='NULL', gold_entity='NULL'):
    total_cardinality = 0
    if task == 'movie':
        file_path = './movie_data/movie.csv'
        df = pd.read_csv(file_path, index_col=0)
        if entity1 != 'NULL':
            value_counts = df['Star1'].value_counts()
            total_cardinality += value_counts.get(entity1, 0)
        if entity2 != 'NULL':
            value_counts = df['Director'].value_counts()
            total_cardinality += value_counts.get(entity2, 0)
        if entity3 != 'NULL':
            value_counts = df['Released_Year'].value_counts()
            total_cardinality += value_counts.get(entity3, 0)
    if task=='soccer':
        file_path = './fifa_data/players_20.csv'
        df = pd.read_csv(file_path, index_col=0)
        if entity1 != 'NULL':
            value_counts = df['club_name'].value_counts()
            total_cardinality += value_counts.get(entity1, 0)
        if entity2 != 'NULL':
            value_counts = df['nationality_name'].value_counts()
            total_cardinality += value_counts.get(entity2, 0)
        if entity3 != 'NULL':
            value_counts = df['club_jersey_number'].value_counts()
            total_cardinality += value_counts.get(entity3, 0)
    return total_cardinality

def compute_cardinality(task, model):
    if task =='movie':
        df = pd.read_csv('./movie_data/crafted/movie_df.csv',index_col=0)
    if task== 'soccer':
        df = pd.read_csv('./fifa_data/crafted/fifa_df.csv',index_col=0)
    exclude_list =   check_validation_file(task, model)
    total_cardinality =0
    cnt=0
    for i in range(len(df)):
        if i in exclude_list:
            continue
        cnt+=1
        if 'entity1' in df.columns:
            entity1 = df.loc[i,'entity1']
        else:
            entity1 = 'NULL'
        if 'entity2' in df.columns:
            entity2 = df.loc[i,'entity2']
        else:
            entity2 = 'NULL'
        if 'entity3' in df.columns:
            entity3 = df.loc[i,'entity3']
        else:
            entity3 = 'NULL'
        if 'entity4' in df.columns:
            entity4 = df.loc[i,'entity4']
        else:
            entity4 = 'NULL'

        total_cardinality += check_proxy_cardinality(task, entity1, entity2, entity3, entity4)
    return total_cardinality/cnt
    
def download_image(url, filename):
    response = requests.get(url)

    if response.status_code == 200:
 
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"Image downloaded as {filename}")
    else:
        print("Failed to download image")




def fetch_google_images(queries, folder_name='./images', num_images=5):
    for query in queries:
        new_folder_path = os.path.join(folder_name, query)
        if os.path.exists(new_folder_path):
            continue

        search_url = f"https://www.google.com/search?q={query}&tbm=isch"
        print(search_url)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }

        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        img_urls = []
        for img in soup.find_all('img', limit=num_images):
            if img.has_attr('src'):
                img_url = img['src']
                img_urls.append(img_url)
        filename = f"{folder_name}/{query}/image.jpg"
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
        for img_url in img_urls:
            try:
                download_image(img_url, filename)
                print('SUCCESSFULLY DOWNLOADED')
                break
            except:
                print(f"Failed: {img_url}")
                continue

def run_gemini_vision(prompt, img_keyword, tasktype='normal',cot_q=[], cot_a=[], cot=False):
    if img_keyword == 'Undefined Club': 
        return 'Unsure. This is an invalid question'
    fetch_google_images([img_keyword])
    img = PIL.Image.open(f'./images/{img_keyword}/image.jpg')
    if tasktype=='normal':
        system_msg = "Answer the following question in yes or no, and then explain why. Say 'unsure' if you don't know and then explain why."
    elif tasktype =='validate':
        system_msg = "Answer the following question in yes or no. Be concise."
    if cot: 
        assert(len(cot_q)==len(cot_a))
        msg = [
            {"role": "system", "content": system_msg}
        ]
        for q,a in zip(cot_q, cot_a):
            msg.append({'role': 'user', 'content': q})
            msg.append({'role': 'assistant', 'content': a})
        msg.append({"role": "user", "content":prompt})
    else:
        msg = [
                system_msg, prompt, img
            ]


    genai.configure(api_key='') #KEY

    model = genai.GenerativeModel('gemini-pro-vision')

    config = {"temperature" : 0}
    safety_settings = {
            glm.HarmCategory.HARM_CATEGORY_HARASSMENT: glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            glm.HarmCategory.HARM_CATEGORY_HATE_SPEECH: glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            glm.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            glm.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        }



    response = model.generate_content(msg, generation_config = config, safety_settings = safety_settings)
    #print(response.prompt_feedback, prompt)


    return response.text



def run_claude(prompt, tasktype='normal',cot_q=[], cot_a=[], cot=False):
    client = anthropic.Anthropic(
        api_key="",  # WRITE YOUR KEY
        
    )
    if tasktype=='normal':
        system_msg = "Answer the following question in yes or no, and then explain why. Say 'unsure' if you don't know and then explain why."
    elif tasktype =='validate':
        system_msg = "Answer the following question in yes or no. Be concise"
    if cot: 
        assert(len(cot_q)==len(cot_a))
        msg = []
        for q,a in zip(cot_q, cot_a):
            msg.append({'role': 'user', 'content': q})
            msg.append({'role': 'assistant', 'content': a})
        msg.append({"role": "user", "content":prompt})
    else:
        msg = [
            {"role": "user", "content": prompt}
        ]
    message = client.messages.create(
        model="claude-3-sonnet-20240229", #CLAUDE VERSION
        temperature=0.0,
        max_tokens = 1024,
        system=system_msg,
        messages= msg
    )
    return message.content[0].text

def run_mistral(prompt, tasktype='normal',cot_q=[], cot_a=[], cot=False):
    data = {
        "input_data": {
            "input_string": [],
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0,
                "return_full_text": True
            }
        }
    }
    if tasktype=='normal':
        system_msg = "Answer the following question in yes or no, and then explain why. Say 'unsure' if you don't know and then explain why."
    elif tasktype =='validate':
        system_msg = "Answer the following question in yes or no. Be concise."
    if cot: 
        assert(len(cot_q)==len(cot_a))
        """msg = [
            {"role": "system", "content": system_msg}
        ]"""
        msg = []
        for q,a in zip(cot_q, cot_a):
            msg.append({'role': 'user', 'content': q})
            msg.append({'role': 'assistant', 'content': a})
        msg.append({"role": "user", "content": system_msg + " " + prompt})
    else:
        msg = [
                #{"role": "system", "content": system_msg},
                {"role": "user", "content":system_msg + ' ' + prompt},
            ]
    data['input_data']["input_string"] = msg
    body = str.encode(json.dumps(data))

    url = ''
    # Replace this with the primary/secondary key or AMLToken for the endpoint
    api_key = ''
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    # The azureml-model-deployment header will force the request to go to a specific deployment.
    # Remove this header to have the request observe the endpoint traffic rules
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        print(prompt)
        print(result)
        result_dict = eval(result.decode('utf8'))[0]
        last_answer = result_dict[str(len(result_dict)-1)]
        
        return last_answer.split('\'assistant\'')[1].split('\'content\': ')[1].split("\}")[0].strip("\"\}\{\'")

    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))

def run_llama(prompt, tasktype='normal',cot_q=[], cot_a=[], cot=False):
    data = {
        "input_data": {
            "input_string": [],
            "parameters": {
                "temperature": 0,
                "max_new_tokens": 200
            }
        }
    }
    if tasktype=='normal':
        system_msg = "Answer the following question in yes or no, and then explain why. Say 'unsure' if you don't know and then explain why."
    elif tasktype =='validate':
        system_msg = "Answer the following question in yes or no. Be concise."
    if cot: 
        assert(len(cot_q)==len(cot_a))
        msg = [
            {"role": "system", "content": system_msg}
        ]
        for q,a in zip(cot_q, cot_a):
            msg.append({'role': 'user', 'content': q})
            msg.append({'role': 'assistant', 'content': a})
        msg.append({"role": "user", "content":prompt})
    else:
        msg = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content":prompt},
            ]
    data['input_data']["input_string"] = msg
    body = str.encode(json.dumps(data))

    url = ''
    # Replace this with the primary/secondary key or AMLToken for the endpoint
    api_key = ''
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    # The azureml-model-deployment header will force the request to go to a specific deployment.
    # Remove this header to have the request observe the endpoint traffic rules
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key) }

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        
        return eval(result.decode('utf8'))["output"]
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))

def run_gemini(prompt, tasktype='normal',cot_q=[], cot_a=[], cot=False):

    if tasktype=='normal':
        system_msg = "Answer the following question in yes or no, and then explain why. Say 'unsure' if you don't know and then explain why."
    elif tasktype =='validate':
        system_msg = "Answer the following question in yes or no. Be concise."
    if cot: 
        assert(len(cot_q)==len(cot_a))
        msg = [
            system_msg
        ]
        for q,a in zip(cot_q, cot_a):
            msg.append(q)
            msg.append(a)
        msg.append(prompt)
    else:
        msg = [
                system_msg, prompt
            ]

    genai.configure(api_key='')

    model = genai.GenerativeModel('gemini-pro')

    config = {"temperature" : 0, "max_output_tokens" : 200}
    safety_settings = {
        glm.HarmCategory.HARM_CATEGORY_HARASSMENT: glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        glm.HarmCategory.HARM_CATEGORY_HATE_SPEECH: glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        glm.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        glm.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    }
    response = model.generate_content(msg, generation_config = config, safety_settings = safety_settings)

    return response.text

def run_gemini_vision(prompt, img_keyword, tasktype='normal',cot_q=[], cot_a=[], cot=False):
    if img_keyword == 'Undefined Club': 
        return 'Unsure. This is an invalid question'
    fetch_google_images([img_keyword])
    img = PIL.Image.open(f'./images/{img_keyword}/image.jpg')
    if tasktype=='normal':
        system_msg = "Answer the following question in yes or no, and then explain why. Say 'unsure' if you don't know and then explain why."
    elif tasktype =='validate':
        system_msg = "Answer the following question in yes or no. Be concise."
    if cot: 
        assert(len(cot_q)==len(cot_a))
        msg = [
            {"role": "system", "content": system_msg}
        ]
        for q,a in zip(cot_q, cot_a):
            msg.append({'role': 'user', 'content': q})
            msg.append({'role': 'assistant', 'content': a})
        msg.append({"role": "user", "content":prompt})
    else:
        msg = [
                system_msg, prompt, img
            ]


    genai.configure(api_key='')

    model = genai.GenerativeModel('gemini-pro-vision')

    config = {"temperature" : 0}
    safety_settings = {
            glm.HarmCategory.HARM_CATEGORY_HARASSMENT: glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            glm.HarmCategory.HARM_CATEGORY_HATE_SPEECH: glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            glm.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            glm.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        }



    response = model.generate_content(msg, generation_config = config, safety_settings = safety_settings)
    #print(response.prompt_feedback, prompt)


    return response.text

def run_llm(prompt, tasktype='normal',cot_q=[], cot_a=[], cot=False, model='', img_keyword = ''):
    if model== 'gpt35':
        return run_gpt35(prompt, tasktype,cot_q, cot_a, cot)
    elif model == 'gpt4':
        return run_gpt4(prompt, tasktype,cot_q, cot_a, cot)
    elif model == 'mistral':
        return run_mistral(prompt, tasktype,cot_q, cot_a, cot)
    elif model == 'llama':
        return run_llama(prompt, tasktype,cot_q, cot_a, cot)
    elif model == 'gemini':
        return run_gemini(prompt, tasktype,cot_q, cot_a, cot)
    elif model =='claude':
        return run_claude(prompt, tasktype, cot_q, cot_a, cot)
    elif model == 'gemini_vision':
        return run_gemini_vision(prompt, img_keyword, tasktype,cot_q, cot_a, cot)
    elif model =='gpt_finetune':
        return run_finetuned_gpt35(prompt, tasktype,cot_q, cot_a, cot)
    elif model =='gpt_finetune_2':
        return run_finetuned_gpt35_2(prompt, tasktype,cot_q, cot_a, cot)
    elif model=='gpt4v':
        return run_gpt_vision(prompt, img_keyword, tasktype)
    else:
        raise ValueError("MODEL NOT YET DEFINED!!")

def gpt4V_encode(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def run_gpt_vision(prompt, img_keyword, tasktype='normal'):
    if img_keyword == 'Undefined Club': 
        return 'Unsure. This is an invalid question'
    fetch_google_images([img_keyword])
    data_url =gpt4V_encode(f'./images/{img_keyword}/image.jpg')
    end_point = "" #GPT VISION ENEDPOINT
    api_key = '' #GPT VISION KEY

    headers = {
    "Content-Type": "application/json",
    "api-key": api_key,
    }   
    

    if tasktype=='normal':
        system_msg = "Answer the following question in yes or no, and then explain why. Say 'unsure' if you don't know and then explain why."
    elif tasktype =='validate':
        system_msg = "Answer the following question in yes or no. Be concise."
  
    
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
            "role": "system",
            "content": [
            {
                "type": "text",
                "text": system_msg
            },

            ]
        },
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{data_url}"
                }
            }
            ]
        }
        ],
        'max_tokens': 4096,
        'temperature':0
    }
    response = requests.post(end_point, headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']
    
    
    
    
    
def search_wiki(keyword):
    common_entity = keyword
    #max_length = 256
    # wikipedia client
    user_agent = "ssoy_openai"
    wiki_client = wikipediaapi.Wikipedia(user_agent, 'en')
    page = wiki_client.page(common_entity)
    if page.exists():
        summary = page.summary
        if len(summary)<=0:
            print("Page: {} - has no summary".format(common_entity))
        else:
            print(summary)
    else:
        print("Page: {} - does not exist".format(common_entity))

def gold_entity_soccer_club_heuristic(gold_entity):
    gold_entity_list = []
    gold_entity_list.append(gold_entity)
    gold_entity_list.append(gold_entity.replace('CF', ''))
    gold_entity_list.append(gold_entity.replace('FK', ''))
    gold_entity_list.append(gold_entity.replace('FC ', ''))
    gold_entity_list.append(gold_entity.replace('AC ', ''))
    gold_entity_list.append(gold_entity.replace('RC ', ''))
    gold_entity_list.append(gold_entity.replace('U.C. ', 'U.C.'))
    gold_entity_list.append(gold_entity.replace('de', ' '))
    gold_entity_list.append(gold_entity.replace('Al', ''))
    gold_entity_list.append(gold_entity.replace('SK', ''))
    gold_entity_list.append(gold_entity.replace('SL', ''))
    gold_entity_list.append(gold_entity.replace('SV', ''))
    #gold_entity_list.append(gold_entity.replace('City', ''))
    gold_entity_list.append(gold_entity.replace('1.', ''))
    if "Bayern München" in gold_entity:
        gold_entity_list.append("Bayern Munich")
    if "Bayern Munchen" in gold_entity:
        gold_entity_list.append("Bayern Munich")
    if 'Leverkusen' in gold_entity:
        gold_entity_list.append('Bayer Leverkusen')
    if 'Real Betis' in gold_entity:
        gold_entity_list.append('Real Betis')
    if "Hoffenheim" in gold_entity:
        gold_entity_list.append("Hoffenheim")
    if "Bilbao" in gold_entity:
        gold_entity_list.append("Bilbao")
    #if "Madrid" in gold_entity:
     #   gold_entity_list.append("Madrid")
    if "Tigres" in gold_entity:
        gold_entity_list.append("Tigres")
    if "Montpellier" in gold_entity:
        gold_entity_list.append("Montpellier")
    if "Lille" in gold_entity:
        gold_entity_list.append("Lille")
    for i in range(len(gold_entity_list)):
        gold_entity_list[i] = gold_entity_list[i].strip()
        gold_entity_list[i] = gold_entity_list[i].replace('   ', ' ')
        gold_entity_list[i] = gold_entity_list[i].replace('  ', ' ')
    return gold_entity_list

def check_soccer_olympic_entity(gold_entity, model_reasoning):
    gold_entities = gold_entity.split(',')
    if not (len(gold_entities) ==3 or len(gold_entities)==2):
        raise ValueError("STRANGE PARSING PROBABLY IN GOLD ENTITIES")
    gold_entity1 = gold_entities[0]
    gold_entity2 = gold_entities[1]
    gold_entity3 = 'NEVER GONNA!_!__!_!_!EXIST' if len(gold_entities)==2 else gold_entities[2]
    val=0
    gold_entity1_list = gold_entity_soccer_club_heuristic(gold_entity1)
    for gval in gold_entity1_list:
        if gval.lower().strip() in model_reasoning.lower().strip():
            val+=1
            break
    if gold_entity2.lower().strip() in model_reasoning.lower().strip():
        val +=1
    if len(gold_entities)==2:
        return val/2
    else:
        sub_g_3 = gold_entity3.strip().split('/')
        sval=0
        for yearval in sub_g_3:
            if yearval.strip() in model_reasoning.lower().strip():
                sval+=1
        sval /=len(sub_g_3)
        val += sval
        return val/3

def make_rag_df(task):
    retriever = WikipediaRetriever()

    if task == 'soccer':
        df = pd.read_csv('./fifa_data/crafted/fifa_df.csv')
        new_df = pd.DataFrame(columns=['entity1','entity2', 'content'])
        for i in range(len(df)):
            docs = ''
            tmp = []
            entity1 = df.loc[i,'entity1']
            entity2 = df.loc[i,'entity2']
            queries = [f'Football Club {entity1.strip()} in 2019', f'{entity2.strip()} National Football Team in 2019']

            for query in queries:
                entity_doc = retriever.get_relevant_documents(query)
                if len(entity_doc) > 0:
                    docs += entity_doc[0].metadata['summary']
                    docs+= '\n\n'
            tmp.append(entity1.strip())
            tmp.append(entity2.strip())
            tmp.append(docs)
            new_df.loc[i] = tmp
            time.sleep(1)
    if task == 'movie':
        df = pd.read_csv('./movie_data/crafted/movie_df.csv')
        new_df = pd.DataFrame(columns=['entity1','entity2', 'content'])
        for i in range(len(df)):
            docs = ''
            tmp = []
            entity1 = df.loc[i,'entity1']
            entity2 = df.loc[i,'entity2']
            queries = [f'actor {entity1.strip()}', f'director {entity2.strip()}']

            for query in queries:
                entity_doc = retriever.get_relevant_documents(query)
                if len(entity_doc) > 0:
                    docs += entity_doc[0].metadata['summary']
                    docs+= '\n\n'
            tmp.append(entity1.strip())
            tmp.append(entity2.strip())
            tmp.append(docs)
            new_df.loc[i] = tmp
        time.sleep(1)
    new_df.to_csv(f'./rag_data/{task}.csv')
    
def retrieval_augmented_text(queries, task, entity1, entity2):
    
    retriever = WikipediaRetriever()
    rag_df = pd.read_csv(f'./rag_data/{task}.csv')
    match = rag_df[(rag_df['entity1'] == entity1)& (rag_df['entity2']==entity2)]
    if len(match)==0:
        print('No match')
        docs = ''
        for query in queries:
            entity_doc = retriever.get_relevant_documents(query)
            if len(entity_doc) > 0:
                docs += entity_doc[0].metadata['summary']
                docs+= '\n\n'
        return docs
    else:
        return match.iloc[0]['content']
def check_director_name_year(gold_entity, model_reasoning):

    name, year = gold_entity.split(',')
    name = name.strip()
    year = year.strip()
    tot_val = 0
    first_name = name.split(' ')[0]
    last_name = name.split(' ')[-1]
    if first_name.strip() in model_reasoning or last_name.strip() in model_reasoning:
        tot_val +=1
    if str(year).strip() in model_reasoning:
        tot_val +=1
    tot_val  = tot_val /2
    return tot_val