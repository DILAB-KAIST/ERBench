import pandas as pd
#import airportsdata as ap
import re
import numpy as np
import argparse
import json
import pycountry
import calendar
import random
def get_month_number(month_name):
    # Get a mapping of month names to month numbers
    month_name_to_number = {name: num for num, name in enumerate(calendar.month_name)}



    # Retrieve the month number
    month_number = month_name_to_number.get(month_name)

    return month_number
def get_qa_prompt(task, df_row, candidate):
    '''
        If candidate ==0 => Existence Prompt with Answer Yes
        If candidate ==1 => Existence Prompt with Answer No
        If candidate ==2 => Negation Prompt with Answer No
        If candidate ==3 => Negation Prompt with Answer Yes
    '''
    answer_random = np.random.randint(0,2)
    if task =='movie':
        common_entity, entity3, entity2, entity1 = df_row['common_entity'].strip(), df_row['entity3'], df_row['entity2'].strip(), df_row['entity1'].strip()
        
        if candidate ==0:
            user_question = f"Is there a movie, released in {int(entity3)}, starring {entity1} where {entity2} is the director?"#YES
            
            if answer_random==0:
                assistant_answer= f'Yes, the movie, among various movies {entity1} starred in {int(entity3)}, {entity2} directed the movie "{common_entity}".'
            else:
                assistant_answer = f'Yes, the movie, among various movies that {entity2} directed in {int(entity3)}, {entity1} starred in the movie, "{common_entity}". '
        elif candidate ==1:
            random_number = np.random.randint(-2,3)
            while(random_number ==0):
                random_number = np.random.randint(-2,3)
            user_question = f"Is there a movie, released in {int(entity3) +random_number}, starring {entity1} where {entity2} is the director?"#NO
            if answer_random ==0:
                assistant_answer = f' No, however there is a movie, "{common_entity}" that {entity2} directed, where {entity1} starred in, which was released in {int(entity3)}.'
            else:
                assistant_answer = f'No, however there is movie, "{common_entity}" that {entity1} starred in, where {entity2} is the director, which was released in {int(entity3)}.'
        elif candidate ==2:
            user_question = f"Is it true that there are no movies, released in {int(entity3)}, starring {entity1} where {entity2} is the director?"
            if answer_random ==0:
                assistant_answer =f'No, it is not true. The movie, "{common_entity}" that {entity2} directed, where {entity1} appeared in the movie was released in {int(entity3)}.'
            else:
                assistant_answer = f' No, it is not true. The movie "{common_entity}", which was released in 2019 was directed by {entity2}, where {entity1} starred in the movie.'
        elif candidate ==3:
            random_number = np.random.randint(-2,3)
            while(random_number ==0):
                random_number = np.random.randint(-2,3)
            user_question = f"Is it true that there are no movies, released in {int(entity3)+ random_number}, starring {entity1} where {entity2} is the director?"
            if answer_random==0:
                assistant_answer = f'Yes, it is true. However, there is a movie, "{common_entity}" that {entity2} directed and {entity1} starred in, which was released in {int(entity3)}.'
            else:
                assistant_answer = f'Yes, it is true. However, there is a movie, "{common_entity}" that {entity2} directed and {entity1} starred in, which was released in {int(entity3)}.'
        else:
            raise ValueError("STRANGE CANDIDATE")
    elif task =='soccer':
        common_entity, entity3, entity2, entity1 = df_row['common_entity'].strip(), df_row['entity3'], df_row['entity2'].strip(), df_row['entity1'].strip()
        if candidate==0:
            user_question = f"Is there a soccer player from {entity2} who played for {entity1} with uniform number {int(entity3)} in {entity1} in 2019?"

            assistant_answer = f'Yes, {common_entity}, a {entity2} footballer, was a member of {entity1} and wore the uniform number {int(entity3)} while playing for {entity1} in 2019.'

        elif candidate ==1:
            country_names = [country.name for country in pycountry.countries]

    # Get a random country name
            while(True):
                random_country = random.choice(country_names)
                if random_country.strip() !=entity2.strip():
                    break
            user_question = f"Is there a soccer player from {random_country} who played for {entity1} with uniform number {int(entity3)} in {entity1} in 2019?"
            assistant_answer = f'No. {common_entity}, a {entity2} soccer player, was a member of {entity1} and wore the uniform number {int(entity3)} while playing for {entity1} in 2019. He is from {entity2}, not {random_country}.'
        elif candidate ==2:
            user_question = f'Is it true that there are no soccer players from {entity2} who played for {entity1} with uniform number {int(entity3)} in {entity1} in 2019?'
            assistant_answer = f'No, it is not true.  There is a {entity2} soccer player, {common_entity}, who was a member of {entity1} and wore the uniform number {int(entity3)} while playing for {entity1} in 2019.'
        elif candidate ==3:
            country_names = [country.name for country in pycountry.countries]

    # Get a random country name
            while(True):
                random_country = random.choice(country_names)
                if random_country.strip() !=entity2.strip():
                    break
            user_question = f'Is it true that there are no soccer players from {random_country} who played for {entity1} with uniform number {int(entity3)} in {entity1} in 2019?'
            assistant_answer = f'Yes, it is true. The soccer player, who was a member of {entity1} wearing the uniform number {int(entity3)} while playing for {entity1} in 2019 was {common_entity}. He is from {entity2}, not {random_country}.'
        else:
            raise ValueError("STRONG CANDIDATE")
    
    elif task =='music':
        common_entity, entity2, entity1 = df_row['common_entity'].strip(), df_row['entity2'], df_row['entity1'].strip()
        if candidate ==0:      
            user_question = f'Is there an artist or group who sang a song titled {entity1} in {int(entity2)}?'
            assistant_answer = f'Yes, there is an artist who sang a song titled "{entity1}" in {int(entity2)}. The song is by the artist/group {common_entity}.'
        elif candidate==1:
            random_number = np.random.randint(-2,3)
            while(random_number ==0):
                random_number = np.random.randint(-2,3)
            user_question =  f'Is there an artist or group who sang a song titled {entity1} in {int(entity2)+random_number}?'
            assistant_answer = f'No, there is no such artist or group. However, there is an artist who sang a song titled "{entity1}" in {int(entity2)}. The song is by the artist {common_entity}.'
        elif candidate ==2:
            user_question = f"Is it true that no artists nor groups sang a song titled {entity1} in {int(entity2)}?"
            assistant_answer = f'No, it is not true. {common_entity} sang a song titled "{entity1}" in {int(entity2)}.'
        elif candidate==3:
            random_number = np.random.randint(-2,3)
            while(random_number ==0):
                random_number = np.random.randint(-2,3)
            user_question = f"Is it true that no artists nor groups sang a song titled {entity1} in {int(entity2)+random_number}?"
            assistant_answer = f'Yes, it is true. However, there is an artist who sang a song titled "{entity1}" in {int(entity2)}. The song is by the artist {common_entity}.'
        else:
            raise ValueError("STRONG CANDIDATE")
    elif task =='book':
        common_entity, entity2, entity1 = df_row['common_entity'].strip(), df_row['entity2'].strip(), df_row['entity1'].strip()
        if candidate ==0:
            user_question = f"Is there a book written by {entity1} that was published in {entity2}?"
            assistant_answer = f'Yes, there is a book written by {entity1} that was published in {entity2}. The book is titled "{common_entity}".'
        elif candidate ==1:
            #January, 2009
            original_month, original_year  = entity2.split(',')
            original_year = int(original_year.strip())
            original_month = original_month.strip()
            random_choice = np.random.randint(0,2)
            random_number = np.random.randint(-2,3)
            while(random_number ==0):
                random_number = np.random.randint(-2,3)
            if random_choice==0:

                mnum = get_month_number(original_month)


                new_month_num = mnum+ random_number
                if new_month_num>=12:
                    new_month_num = 12
                if new_month_num <=1:
                    new_month_num=1
                if mnum ==new_month_num:
                    new_month_num = 13-new_month_num
                new_month = calendar.month_name[new_month_num]
                new_year = original_year
            elif random_choice ==1:
                new_year = original_year+random_number
                new_month = original_month 
            else:
                raise ValueError('np.random wrong')
            
            random_date = f'{new_month}, {new_year}' #TODO NEED TO CHANGE MONTH OR YEAR
            user_question = f"Is there a book written by {entity1} that was published in {random_date}?"
            assistant_answer = f'No, there are no books written by {entity1} that was published in {random_date}. There is a book, written by {entity1} that was published in a similar date, {entity2}. The book is titled "{common_entity}".'

        elif candidate ==2:
            user_question = f"Is it true that there are no books written by {entity1} that were published in {entity2}?"
            assistant_answer = f'No, it is not true. There is a book written by {entity1} that was published in {entity2}. The book is titled "{common_entity}".'
        elif candidate==3:
            original_month, original_year  = entity2.split(',')
            original_year = int(original_year.strip())
            original_month = original_month.strip()
            random_choice = np.random.randint(0,2)
            random_number = np.random.randint(-2,3)
            while(random_number ==0):
                random_number = np.random.randint(-2,3)
            if random_choice==0:

                mnum = get_month_number(original_month)


                new_month_num = mnum+ random_number
                if new_month_num>=12:
                    new_month_num = 12
                if new_month_num <=1:
                    new_month_num=1
                if mnum ==new_month_num:
                    new_month_num = 13-new_month_num
                new_month = calendar.month_name[new_month_num]
                new_year = original_year
            elif random_choice ==1:
                new_year = original_year+random_number
                new_month = original_month 
            else:
                raise ValueError('np.random wrong')
            
            random_date = f'{new_month}, {new_year}' #TODO NEED TO CHANGE MONTH OR YEAR
            user_question = f"Is it true that there are no books written by {entity1} that were published in {random_date}?"
            assistant_answer = f'Yes, it is true. There are no books written by {entity1} that were published in {random_date}. There is a book , written by {entity1} that was published in a similar date, {entity2}. The book is titled "{common_entity}".'

        else:
            raise ValueError("STRONG CANDIDATE")
    else:
        raise ValueError("YET UNDEFINED")
    return user_question, assistant_answer
    

def parse_args():
    '''
    Config
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default = 3000)
    args = parser.parse_args()
    
    return args


def random_split(input_list, num_splits):
    # Get the length of the input list
    list_length = len(input_list)

    # Ensure the number of splits is valid
    if num_splits < 1 or num_splits > list_length:
        raise ValueError("Invalid number of splits")

    # Calculate the approximate size of each split
    split_size = list_length // num_splits

    # Randomly shuffle the indices
    shuffled_indices = random.sample(range(list_length), list_length)

    # Split the shuffled indices into num_splits parts
    splits = [shuffled_indices[i * split_size:(i + 1) * split_size] for i in range(num_splits - 1)]
    splits.append(shuffled_indices[(num_splits - 1) * split_size:])

    # Use the indices to extract the corresponding elements for each split
    result_splits = [[input_list[i] for i in split] for split in splits]

    return result_splits



def define_hyperparameters(n, datasets = ['movie', 'soccer', 'book', 'music']):

    hyperparameters = dict()
    hyperparameters['datasets'] = datasets
    hyperparameters['N'] = n  #number of entities per dataset / HYPERPARAMETER

    ##NEEDED FOR FINETUNING NOT NOW
    hyperparameters['n_epochs'] = 2
    hyperparameters['batch_size'] = 1
    hyperparameters["learning_rate_multiplier"]=1

    '''model_hyperparameters= {
    "n_epochs" : 2,
    "batch_size”:1,
    “learning_rate_multiplier”: 1
    }
    '''
    return hyperparameters

def preprocess_df(task, hyperparameters):
    if task =='movie':
        df = pd.read_csv('./movie_data/movie.csv')
        df = df[['movie_title','title_year', 'director_name', 'actor_1_name']]
        df.columns = ['common_entity', 'entity3' ,'entity2', 'entity1']
        df.dropna(inplace=True)
        df = df.iloc[1500:1500+hyperparameters['N']]
        df['common_entity'] = df['common_entity'].str.strip()
        df = df.reset_index()
    elif task =='soccer':
        
        df = pd.read_csv('./fifa_data/players_20.csv')
        df = df[['short_name', 'club_name','club_jersey_number', 'nationality_name']]
        df.columns = ['common_entity', 'entity1','entity3',  'entity2']
        df.dropna(inplace=True)
        df= df.iloc[1500:1500+hyperparameters['N']]
        df = df.reset_index()
        

    elif task =='book':
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
        df = df.iloc[1500:1500+hyperparameters['N']]
        df.reset_index(inplace=True)
    elif task=='music':
        random_seed= 1146
        df = pd.read_csv('./music_data/music.csv')
        df = df[['artist_name', 'track_name', 'release_date']]
        df.dropna(inplace=True)
        df = df[df['artist_name']!='x']
        df = df[df['track_name']!='x']
        df.columns = ['common_entity', 'entity1', 'entity2']
        df = df.sample(frac=1, random_state= random_seed)
        df = df.iloc[1500:1500+hyperparameters['N']]
        df = df.reset_index()
    else:
        raise ValueError("STRANGE DATASET FOR THE CURRENT EXPERIMENT")
    return df
def main():
    args = parse_args()
    N = args.n
    hyperparameters = define_hyperparameters(N, datasets=['soccer'])

    system_prompt = "Answer the following question in yes or no, and then explain why. Say 'unsure' if you don't know and then explain why."
    input_list = list(range(N))
    num_splits = 4
    candidate0, candidate1, candidate2, candidate3 = random_split(input_list, num_splits)
    json_data_list = []
    for dataset in hyperparameters['datasets']:

        df = preprocess_df(dataset, hyperparameters)
        for i in range(len(df)):
            row = df.loc[i]
            if i in candidate0:
                idx=0
            elif i in candidate1:
                idx=1
            elif i in candidate2:
                idx =2
            else:
                idx=3
            q, a = get_qa_prompt(dataset, row, idx)
            

            json_data = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ]
            }
            json_data_list.append(json_data)
    random.shuffle(json_data_list)
    output_file_path = f"./finetune_data/soccer_training_set.jsonl"
    with open(output_file_path, 'w') as jsonl_file:
        for json_data in json_data_list:
            # Convert each dictionary to a JSON string and write it as a line in the JSONL file
            jsonl_file.write(json.dumps(json_data) + '\n')
    print(f"FINISHED WRITING {output_file_path}")

if __name__ == "__main__":
    main()