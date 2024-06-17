'''
Code to anlayze Multiple-choice QA result.

Notes:
- This code should be run after running the main code (i.e., run_qa.py)

- To consider internal knowledge of LLM, we consider three types of analysis:
    (1) Use all QA pairs
    (2) Use only valid QA pairs of the model (i.e. entities which the given model can answer)
    (3) Use common valid QA pairs among the models

    By default, this code saves the result of (1) and (2) in the same directory as the QA log file.
    The validation mode among 1, 2, 3 can be controlled.
    
- When comparing non-rag (original multiple-choice QA) vs rag, run analysis of non-rag first and then run analysis of rag.
'''

import pandas as pd
import argparse
import os
import json
import re
import utils
from unidecode import unidecode
from utils import get_savename


def qa_split_claude(log_path, save=True):
    '''
    Given log file, return dataframe contains as follows:
    ['question', 'entity_idx', 'question_idx', 'model_answer', 'model_reasoning', 'gold_answer', 'gold_entity']
    '''
    # read log file and process to csv
    csv_path = log_path.replace('.log', '.csv')
    res_df = pd.DataFrame(columns=['entity_idx', 'question_idx', 'question', 'choices', 'gold_answer', 'gold_entity', 'model_answer', 'model_response_all'])     
    with open(log_path, 'r') as f:
        
        corpus = f.read()

        # log patterns
        pattern = r'(\d+)-(\d+[a-z]{2}\squestion)\n(<Question>.*?)(<Answer>.*?)(<Gold Answer>.*?)(<Gold Entity>.*?)(?=\n\n|$)'
        matches = re.findall(pattern, corpus, re.DOTALL)

    
        for match in matches:
            res = []

            # parse entity id and question(prompt) id
            entity_id = match[0]
            question_id = match[1].replace('th question', '')

            # parse question
            if 'rag' in log_path:
                question_pattern = r'<Question>\s*(.*?)\n\nA:'
            else:
                question_pattern = r'<Question>\s*Q: (.*?) Provide'
            question = re.findall(question_pattern, match[2], re.DOTALL)[0]
            
            # parse choices
            option_pattern = r'Option (\d): (.+)'
            options = re.findall(option_pattern, match[2], re.MULTILINE)
            num_options= len(options)
            options_joined = list(map(lambda x: f"Option {x[0]}: {x[1]}", options))


            # parse model response
            response = match[3].replace('<Answer>', '').replace("\n\n", "\n").replace("\n", " ").strip()
            response_sentences = match[3].replace('<Answer>', '').replace("\n\n", "\n").strip().split('\n')
            if response == 'RESPONSE ERROR':
                continue
            

            # parse model answer in first sentence
            explicit_answer = re.findall(r'Option (\d):', response.split('.')[0])
            if len(explicit_answer) == 1: # One answer, ideal case
                model_answer = explicit_answer[0]

                # check if the option number is valid
                if model_answer not in [str(i) for i in range(1, num_options+1)]:
                    model_answer = "BINDING ERROR"

            elif len(explicit_answer) >= 1: # Multiple answers
                
                # check if the same option number is repeated
                if len(set(explicit_answer)) == 1:
                    model_answer = explicit_answer[0]
                else:
                    model_answer = "MULTIPLE ANSWER ERROR"

                # check if the answer is "all correct"
                if 'no inaccurate' in response.lower() or 'no incorrect' in response.lower():
                    model_answer = "NONE OF ABOVE"

            elif len(explicit_answer) == 0: # No answer
                # Need manual check!
                model_answer = "NONE OF ABOVE"

                # check whther . is wrongly parsed
                explicit_answer = re.findall(r'option (\d):|Option (\d)|option is (\d)', response_sentences[0])
                if len(explicit_answer) >= 1:
                    for candidate in explicit_answer[0]:
                        if len(candidate) == 1:
                            model_answer = candidate[0]
                            break
                
                
                else:
                    # check the last sentence
                    explicit_answer = re.findall(r'Option (\d):|Option (\d)|option is (\d)', response.split('.')[-1], re.IGNORECASE)
                    if len(explicit_answer) >= 1:
                        for candidate in explicit_answer[0]:
                            if len(candidate) == 1:
                                model_answer = candidate[0]
                                break

                    # check "Therefore", .. etc
                    else: 
                        claude_words = ['therefore', 'based on the analysis', 'option is inaccurate']
                        sentences = response.split('.')
                        for sent in sentences:
                            if any([y in sent.lower() for y in claude_words]):
                                explicit_answer = re.findall(r'Option (\d):|Option (\d)|option is (\d)', sent)
                                if len(explicit_answer) >= 1:
                                    for candidate in explicit_answer[0]:
                                        if len(candidate) == 1:
                                            model_answer = candidate[0]
                                            break



            # parse gold answer 
            gold_answer = match[4].replace('<Gold Answer>', '').replace("\n", " ").strip()
            if gold_answer == "There is no answer":
                gold_answer = "NONE OF ABOVE"

            
            # parse gold entity
            gold_entity = match[5].replace('<Gold Entity>', '').replace("\n", " ").replace("=", '').strip()



            # add to dataframe
            res.append(int(entity_id))
            res.append(int(question_id))
            res.append(question)
            res.append(options_joined)
            res.append(str(gold_answer))
            res.append(str(gold_entity))
            res.append(str(model_answer))
            res.append(response)
            res_df.loc[len(res_df.index)] = res 

        if save:
            # save dataframe
            res_df.to_csv(csv_path)

    return res_df


def qa_split_gpt_gemini_v(log_path, save=True):
    '''
    Given log file, return dataframe contains as follows:
    ['question', 'entity_idx', 'question_idx', 'model_answer', 'model_reasoning', 'gold_answer', 'gold_entity']
    '''

    # read log file and process to csv
    csv_path = log_path.replace('.log', '.csv')
    res_df = pd.DataFrame(columns=['entity_idx', 'question_idx', 'question', 'choices', 'gold_answer', 'gold_entity', 'model_answer', 'model_response_all'])     
    with open(log_path, 'r') as f:
        
        corpus = f.read()

        # log patterns
        pattern = r'(\d+)-(\d+[a-z]{2}\squestion)\n(<Question>.*?)(<Answer>.*?)(<Gold Answer>.*?)(<Gold Entity>.*?)(?=\n\n|$)'
        matches = re.findall(pattern, corpus, re.DOTALL)

    
        for match in matches:
            res = []

            # parse entity id and question(prompt) id
            entity_id = match[0]
            question_id = match[1].replace('th question', '')

            # parse question
            if 'rag' in log_path:
                question_pattern = r'<Question>\s*(.*?)\n\nA:'
            else:
                question_pattern = r'<Question>\s*Q: (.*?) Provide'
            question = re.findall(question_pattern, match[2], re.DOTALL)[0]
            
            # parse choices
            option_pattern = r'Option (\d): (.+)'
            options = re.findall(option_pattern, match[2], re.MULTILINE)
            num_options= len(options)
            options_joined = list(map(lambda x: f"Option {x[0]}: {x[1]}", options))


            # parse model response
            response = match[3].replace('<Answer>', '').replace("\n\n", "\n").replace("\n", " ").strip()
            if response == 'RESPONSE ERROR':
                continue
            

            # parse model answer in first sentence
            explicit_answer = re.findall(r'Option (\d):', response.split('.')[0])
            if len(explicit_answer) == 1: # One answer, ideal case
                model_answer = explicit_answer[0]

                # check if the option number is valid
                if model_answer not in [str(i) for i in range(1, num_options+1)]:
                    model_answer = "BINDING ERROR"

            elif len(explicit_answer) >= 1: # Multiple answers
                
                # check if the same option number is repeated
                if len(set(explicit_answer)) == 1:
                    model_answer = explicit_answer[0]
                else:
                    model_answer = "MULTIPLE ANSWER ERROR"

                # check if the answer is "all correct"
                if 'no inaccurate' in response.lower() or 'no incorrect' in response.lower():
                    model_answer = "NONE OF ABOVE"

            elif len(explicit_answer) == 0: # No answer
                # Need manual check!
                model_answer = "NONE OF ABOVE"

                # check whther other pattern is included
                explicit_answer = re.findall(r'option (\d):|Option (\d)|option is (\d)', response.split('.')[0])
                if len(explicit_answer) >= 1:
                    for candidate in explicit_answer[0]:
                        if len(candidate) == 1:
                            model_answer = candidate[0]
                            break
                
                else:
                    # check answer in the last sentence
                    if 'gemini_v' in log_path:
                        explicit_answer = re.findall(r'Option (\d):|Option (\d)|option is (\d)', response.split('.')[-1], re.IGNORECASE)
                        if len(explicit_answer) >= 1:
                            for candidate in explicit_answer[0]:
                                if len(candidate) == 1:
                                    model_answer = candidate[0]
                                    break
                    # check answer in second sentence (due to '.' included in player's name)
                    elif 'gpt_v' in log_path:
                        response_sent = response.split('.')
                        if len(response_sent) >= 2:
                            explicit_answer = re.findall(r'Option (\d):|Option (\d)|option is (\d)', response.split('.')[1], re.IGNORECASE)
                            if len(explicit_answer) >= 1:
                                for candidate in explicit_answer[0]:
                                    if len(candidate) == 1:
                                        model_answer = candidate[0]
                                        break




            # parse gold answer 
            gold_answer = match[4].replace('<Gold Answer>', '').replace("\n", " ").strip()
            if gold_answer == "There is no answer":
                gold_answer = "NONE OF ABOVE"



            
            # parse gold entity
            gold_entity = match[5].replace('<Gold Entity>', '').replace("\n", " ").replace("=", '').strip()



            # add to dataframe
            res.append(int(entity_id))
            res.append(int(question_id))
            res.append(question)
            res.append(options_joined)
            res.append(str(gold_answer))
            res.append(str(gold_entity))
            res.append(str(model_answer))
            res.append(response)
            res_df.loc[len(res_df.index)] = res 

        if save:
            # save dataframe
            res_df.to_csv(csv_path)

    return res_df

def qa_split_gemini(log_path):
    '''
    Given log file, return dataframe contains as follows:
    ['question', 'entity_idx', 'question_idx', 'model_answer', 'model_reasoning', 'gold_answer', 'gold_entity']
    '''
    
    # check if log file exists
    if not os.path.exists(log_path):
        raise ValueError(f"Log path does not exist: {log_path}")


    # read log file and process to csv
    csv_path = log_path.replace('.log', '.csv')
    res_df = pd.DataFrame(columns=['entity_idx', 'question_idx', 'question', 'choices', 'gold_answer', 'gold_entity', 'model_answer', 'model_response_all'])     
    with open(log_path, 'r') as f:
        
        corpus = f.read()

        # log patterns
        # if all_true, there is no gold entity
        pattern = r'(\d+)-(\d+[a-z]{2}\squestion)\n(<Question>.*?)(<Answer>.*?)(<Gold Answer>.*?)(<Gold Entity>.*?)(?=\n\n|$)'
        matches = re.findall(pattern, corpus, re.DOTALL)

    
        for match in matches:
            res = []

            # parse entity id and question(prompt) id
            entity_id = match[0]
            question_id = match[1].replace('th question', '')

            # parse question
            if 'rag' in log_path:
                question_pattern = r'<Question>\s*(.*?)\n\nA:'
            else:
                question_pattern = r'<Question>\s*Q: (.*?) Provide'
            question = re.findall(question_pattern, match[2], re.DOTALL)[0]
            
            # parse choices
            option_pattern = r'Option (\d): (.+)'
            options = re.findall(option_pattern, match[2], re.MULTILINE)
            num_options= len(options)
            options_joined = list(map(lambda x: f"Option {x[0]}: {x[1]}", options))


             # parse model response
            response_1 = match[3].replace('<Answer>', '').replace("\n\n", "\n").replace("\n", " ").strip()
            if response_1 == 'RESPONSE ERROR':
                continue
            response = match[3].replace('<Answer>','').strip().replace("\n", "newline").strip()

            # parse model answer in first sentence
            response_list = response.split("newline")           

            # parse model answer in first sentence
            explicit_answer = re.findall(r'[Oo]ption (\d)', response_list[0])
            if len(explicit_answer) == 1: # One answer, ideal case
                model_answer = explicit_answer[0]

                # check if the option number is valid
                if model_answer not in [str(i) for i in range(1, num_options+1)]:
                    model_answer = "BINDING ERROR"

            elif len(explicit_answer) >= 1: # Multiple answers
                
                # check if the same option number is repeated
                if len(set(explicit_answer)) == 1:
                    model_answer = explicit_answer[0]
                else:
                    model_answer = "MULTIPLE ANSWER ERROR"

                # check if the answer is "all correct"
                if 'no inaccurate' in response_1.lower() or 'no incorrect' in response_1.lower():
                    model_answer = "NONE OF ABOVE"

            elif len(explicit_answer) == 0: # No answer
                # Need manual check!
                explicit_answer = re.findall(r'[\b\(](\d)[\.\b\)]', response_list[0])
                if len(explicit_answer) == 0: 
                    model_answer = "NONE OF ABOVE"
                else:
                    model_answer = explicit_answer[0]
                # # check whether option sentence is implicitly mentioned
                # for option_id, option_contents in options:
                #     if option_contents.lower() in response.lower():
                #         model_answer = option_id
                #         break   



            # parse gold answer 
            gold_answer = match[4].replace('<Gold Answer>', '').replace("\n", " ").strip()
            if gold_answer == "There is no answer":
                gold_answer = "NONE OF ABOVE"



            
            # parse gold entity
            if "all_true" in log_path:
                gold_entity = "NONE"
            else:
                gold_entity = match[5].replace('<Gold Entity>', '').replace("\n", " ").replace("=", '').strip()
                
              #  if 'airport' in log_path and gold_answer == '4':
                #    gold_entity = gold_entity.split(',')[0].strip()


            # add to dataframe
            res.append(int(entity_id))
            res.append(int(question_id))
            res.append(question)
            res.append(options_joined)
            res.append(gold_answer)
            res.append(gold_entity)
            res.append(model_answer)
            res.append(response_1)
            res_df.loc[len(res_df.index)] = res 

        # save dataframe
        res_df.to_csv(csv_path)

    return res_df

def qa_split_mistral_2(log_path):
    '''
    Given log file, return dataframe contains as follows:
    ['question', 'entity_idx', 'question_idx', 'model_answer', 'model_reasoning', 'gold_answer', 'gold_entity']
    '''
    
    # check if log file exists
    if not os.path.exists(log_path):
        raise ValueError(f"Log path does not exist: {log_path}")

   
    # if the log file already processed to csv, just read it
    csv_path = log_path.replace('.log', '.csv')
    '''if os.path.exists(csv_path):
        print("QA Log is already processed. Reading csv file...")
        return pd.read_csv(csv_path)'''
    

    # read log file and process to csv
    res_df = pd.DataFrame(columns=['entity_idx', 'question_idx', 'question', 'choices', 'gold_answer', 'gold_entity', 'model_answer', 'model_response_all'])     
    with open(log_path, 'r') as f:
        
        corpus = f.read()

        # log patterns
        # if all_true, there is no gold entity
        if "all_true" in log_path:
            pattern = r'(\d+)-(\d+[a-z]{2}\squestion)\n(<Question>.*?)(<Answer>.*?)(<Gold Answer>.*?)(?=\n\n|$)'
        else:
            pattern = r'(\d+)-(\d+[a-z]{2}\squestion)\n(<Question>.*?)(<Answer>.*?)(<Gold Answer>.*?)(<Gold Entity>.*?)(?=\n\n|$)'
        matches = re.findall(pattern, corpus, re.DOTALL)

    
        for match in matches:
            res = []

            # parse entity id and question(prompt) id
            entity_id = match[0]
            question_id = match[1].replace('th question', '')

            # parse question
            question_pattern = r'<Question>\s*Q: (.*?) Provide'
            question = re.findall(question_pattern, match[2], re.DOTALL)[0]
            
            # parse choices
            option_pattern = r'Option (\d): (.+)'
            options = re.findall(option_pattern, match[2], re.MULTILINE)
            num_options= len(options)
            options_joined = list(map(lambda x: f"Option {x[0]}: {x[1]}", options))


            # parse model response
            response_1 = match[3].replace('<Answer>', '').replace("\n\n", "\n").replace("\n", " ").strip()
            if response_1 == 'RESPONSE ERROR':
                continue
            response = match[3].replace('<Answer>','').strip().replace("\n", "newline").strip()

            # parse model answer in first sentence
            response_list = response.split("newline")
            #print(response_list)
            if len(response_list) == 1:
                if len(response_list[0].strip()) == 1 and utils.is_number(response_list[0].strip()):
                    model_answer = int(response_list[0].strip())
                else:
                    explicit_answer = re.findall(r'[oO]ption (\d)', response_list[0])
                    explicit_answer = list(dict.fromkeys(explicit_answer))
                    if len(explicit_answer) == 0:
                        explicit_answer = re.findall(r'\b(\w):', response_list[0])
                        explicit_answer_2 = re.findall(r'\b(\w)\.', response_list[0])
                        explicit_answer = explicit_answer + explicit_answer_2
                        if len(explicit_answer) == 0:
                            model_answer = "NONE OF ABOVE"
                        elif len(explicit_answer) == 1:
                            model_answer = explicit_answer[0]
                            if model_answer not in [str(i) for i in range(1, num_options+1)]:
                                model_answer = "BINDING ERROR" 
                        else:
                            model_answer = "MULTIPLE ANSWER ERROR"
                    elif len(explicit_answer) == 1:
                        model_answer = explicit_answer[0]
                        if model_answer not in [str(i) for i in range(1, num_options+1)]:
                            model_answer = "BINDING ERROR"
                    elif len(explicit_answer) > 1:
                        for j in range(len(explicit_answer)):
                            model_answer = explicit_answer[j]
                            if model_answer in [str(i) for i in range(1, num_options+1)]:
                                break
                        if model_answer not in [str(i) for i in range(1, num_options+1)]:
                            model_answer = "BINDING ERROR"        
            elif len(response_list) > 1:
                if response_list[1] == "":
                    explicit_answer = re.findall(r'[oO]ption (\d)', response_list[0])
                    explicit_answer = list(dict.fromkeys(explicit_answer))
                    if len(explicit_answer) == 0:
                        explicit_answer = re.findall(r'\b(\w)[:\.\)\b]', response_list[0])
                        #explicit_answer_2 = re.findall(r'\b(\w)\.', response_list[0])
                        #explicit_answer = explicit_answer + explicit_answer_2
                        if len(explicit_answer) == 0:
                            model_answer = "NONE OF ABOVE"
                        elif len(explicit_answer) == 1:
                            model_answer = explicit_answer[0]
                            if model_answer not in [str(i) for i in range(1, num_options+1)]:
                                model_answer = "BINDING ERROR" 
                        else:
                            model_answer = "MULTIPLE ANSWER ERROR"
                    elif len(explicit_answer) == 1:
                        model_answer = explicit_answer[0]
                        if model_answer not in [str(i) for i in range(1, num_options+1)]:
                            model_answer = "BINDING ERROR"
                    elif len(explicit_answer) > 1:
                        for j in range(len(explicit_answer)):
                            model_answer = explicit_answer[j]
                            if model_answer in [str(i) for i in range(1, num_options+1)]:
                                break
                        if model_answer not in [str(i) for i in range(1, num_options+1)]:
                            model_answer = "BINDING ERROR"
                else: # Answer shows multiple options
                    #print(response)
                    #print(response_list)
                    i = response_list.index("") + 1
                    if len(response_list) <= i:
                        model_answer = "NONE OF ABOVE"
                    else:
                        explicit_answer = re.findall(r'[oO]ption (\d)', response_list[i])
                        explicit_answer = list(dict.fromkeys(explicit_answer))
                        if len(explicit_answer) == 0:
                            explicit_answer = re.findall(r'\b(\w):', response_list[i])
                            explicit_answer_2 = re.findall(r'\b(\w)\.', response_list[i])
                            explicit_answer = explicit_answer + explicit_answer_2
                            if len(explicit_answer) == 0:
                                model_answer = "NONE OF ABOVE"
                            elif len(explicit_answer) == 1:
                                model_answer = explicit_answer[0]
                                if model_answer not in [str(i) for i in range(1, num_options+1)]:
                                    model_answer = "BINDING ERROR" 
                            else:
                                model_answer = "MULTIPLE ANSWER ERROR"
                        elif len(explicit_answer) == 1:
                            model_answer = explicit_answer[0]
                            if model_answer not in [str(i) for i in range(1, num_options+1)]:
                                model_answer = "BINDING ERROR"
                        elif len(explicit_answer) > 1:
                            for j in range(len(explicit_answer)):
                                model_answer = explicit_answer[j]
                                if model_answer in [str(i) for i in range(1, num_options+1)]:
                                    break
                            if model_answer not in [str(i) for i in range(1, num_options+1)]:
                                model_answer = "BINDING ERROR"
            else:
                model_answer = "NONE OF ABOVE"


            # parse gold answer 
            gold_answer = match[4].replace('<Gold Answer>', '').replace("\n", " ").strip()
            if gold_answer == "There is no answer":
                gold_answer = "NONE OF ABOVE"



            
            # parse gold entity
            if "all_true" in log_path:
                gold_entity = "NONE"
            else:
                gold_entity = match[5].replace('<Gold Entity>', '').replace("\n", " ").replace("=", '').strip()
                
            # add to dataframe
            res.append(int(entity_id))
            res.append(int(question_id))
            res.append(question)
            res.append(options_joined)
            res.append(str(gold_answer))
            res.append(str(gold_entity))
            res.append(str(model_answer))
            response = response.replace("newline", " ")
            res.append(response)
            res_df.loc[len(res_df.index)] = res 

        # save dataframe
        res_df.to_csv(csv_path)

    return res_df

def qa_split_mistral_34(log_path):
    '''
    Given log file, return dataframe contains as follows:
    ['question', 'entity_idx', 'question_idx', 'model_answer', 'model_reasoning', 'gold_answer', 'gold_entity']
    '''
    
    # check if log file exists
    if not os.path.exists(log_path):
        raise ValueError(f"Log path does not exist: {log_path}")

   
    # if the log file already processed to csv, just read it
    csv_path = log_path.replace('.log', '.csv')
    '''if os.path.exists(csv_path):
        print("QA Log is already processed. Reading csv file...")
        return pd.read_csv(csv_path)'''
    

    # read log file and process to csv
    res_df = pd.DataFrame(columns=['entity_idx', 'question_idx', 'question', 'choices', 'gold_answer', 'gold_entity', 'model_answer', 'model_response_all'])     
    with open(log_path, 'r') as f:
        
        corpus = f.read()

        # log patterns
        # if all_true, there is no gold entity
        if "all_true" in log_path:
            pattern = r'(\d+)-(\d+[a-z]{2}\squestion)\n(<Question>.*?)(<Answer>.*?)(<Gold Answer>.*?)(?=\n\n|$)'
        else:
            pattern = r'(\d+)-(\d+[a-z]{2}\squestion)\n(<Question>.*?)(<Answer>.*?)(<Gold Answer>.*?)(<Gold Entity>.*?)(?=\n\n|$)'
        matches = re.findall(pattern, corpus, re.DOTALL)

    
        for match in matches:
            res = []

            # parse entity id and question(prompt) id
            entity_id = match[0]
            question_id = match[1].replace('th question', '')

            # parse question
            question_pattern = r'<Question>\s*Q: (.*?) Provide'
            question = re.findall(question_pattern, match[2], re.DOTALL)[0]
            
            # parse choices
            option_pattern = r'Option (\d): (.+)'
            options = re.findall(option_pattern, match[2], re.MULTILINE)
            num_options= len(options)
            options_joined = list(map(lambda x: f"Option {x[0]}: {x[1]}", options))


            # parse model response
            response_1 = match[3].replace('<Answer>', '').replace("\n\n", "\n").replace("\n", " ").strip()
            if response_1 == 'RESPONSE ERROR':
                continue
            response = match[3].replace('<Answer>','').strip().replace("\n", "newline").strip()

            # parse model answer in first sentence
            response_list = response.split("newline")

            #print(response_list)
            # parse model answer in first sentence
            if len(response_list[0].strip()) == 1 and utils.is_number(response_list[0].strip()):
                model_answer = int(response_list[0].strip())
            else:
                explicit_answer = re.findall(r'\b(\d)[\.\b\)\:]', response_list[0])

                if len(explicit_answer) == 1: # One answer, ideal case
                    model_answer = explicit_answer[0]

                    # check if the option number is valid
                    if model_answer not in [str(i) for i in range(1, num_options+1)]:
                        model_answer = "BINDING ERROR"
                    

                elif len(explicit_answer) >= 1: # Multiple answers
                    
                    # check if the same option number is repeated

                    model_answer = explicit_answer[0]
                    if 'also incorrect' in response or 'also wrong' in response or 'also inaccurate' in response:
                        print("Multi")
                        model_answer = "MULTIPLE ANSWER ERROR"

                    # check if the answer is "all correct"
                    if 'no inaccurate' in response.lower() or 'no incorrect' in response.lower():
                        model_answer = "NONE OF ABOVE"

                elif len(explicit_answer) == 0: # No answer
                    # Need manual check!
                    if len(response_list) >=3:
                        if response_list[1] == "": #and 'expla' in response_list[2].lower():
                            explicit_answer = re.findall(r'[Oo]ption (\d)', response_list[2])
                            explicit_answer_1 = re.findall(r'[Oo]ption is (\d)', response_list[2])
                            explicit_answer = explicit_answer +explicit_answer_1
                    elif len(response_list) == 2:
                        #if 'expla' in response_list[1].lower():
                        explicit_answer = re.findall(r'[Oo]ption (\d)', response_list[1])
                        explicit_answer_1 = re.findall(r'[Oo]ption is (\d)', response_list[1])
                        explicit_answer = explicit_answer +explicit_answer_1

                        
                    if len(explicit_answer) == 1: # One answer, ideal case
                        model_answer = explicit_answer[0]

                        # check if the option number is valid
                        if model_answer not in [str(i) for i in range(1, num_options+1)]:
                            model_answer = "BINDING ERROR"
                        

                    elif len(explicit_answer) >= 1: # Multiple answers
                        
                        # check if the same option number is repeated
                        #if len(set(explicit_answer)) == 1:
                        model_answer = explicit_answer[0]
                        if 'also incorrect' in response or 'also wrong' in response or 'also inaccurate' in response:
                            #print("Multi")
                            model_answer = "MULTIPLE ANSWER ERROR"

                        # check if the answer is "all correct"
                        if 'no inaccurate' in response.lower() or 'no incorrect' in response.lower():
                            model_answer = "NONE OF ABOVE"

                    elif len(explicit_answer) == 0: # No answer
                        model_answer = "NONE OF ABOVE"



            # parse gold answer 
            gold_answer = match[4].replace('<Gold Answer>', '').replace("\n", " ").strip()
            if gold_answer == "There is no answer":
                gold_answer = "NONE OF ABOVE"



            
            # parse gold entity
            if "all_true" in log_path:
                gold_entity = "NONE"
            else:
                gold_entity = match[5].replace('<Gold Entity>', '').replace("\n", " ").replace("=", '').strip()
                
               # if 'airport' in log_path and gold_answer == '4':
                  #  gold_entity = gold_entity.split(',')[0].strip()


            # add to dataframe
            res.append(int(entity_id))
            res.append(int(question_id))
            res.append(question)
            res.append(options_joined)
            res.append(str(gold_answer))
            res.append(str(gold_entity))
            res.append(str(model_answer))
            res.append(response_1)
            res_df.loc[len(res_df.index)] = res 

        # save dataframe
        res_df.to_csv(csv_path)

    return res_df

def qa_split_mistral_demo(log_path):
    '''
    Given log file, return dataframe contains as follows:
    ['question', 'entity_idx', 'question_idx', 'model_answer', 'model_reasoning', 'gold_answer', 'gold_entity']
    '''
    
    # check if log file exists
    if not os.path.exists(log_path):
        raise ValueError(f"Log path does not exist: {log_path}")

   
    # if the log file already processed to csv, just read it
    csv_path = log_path.replace('.log', '.csv')
    '''if os.path.exists(csv_path):
        print("QA Log is already processed. Reading csv file...")
        return pd.read_csv(csv_path)'''
    

    # read log file and process to csv
    res_df = pd.DataFrame(columns=['entity_idx', 'question_idx', 'question', 'choices', 'gold_answer', 'gold_entity', 'model_answer', 'model_response_all'])     
    with open(log_path, 'r') as f:
        
        corpus = f.read()

        # log patterns
        # if all_true, there is no gold entity
        if "all_true" in log_path:
            pattern = r'(\d+)-(\d+[a-z]{2}\squestion)\n(<Question>.*?)(<Answer>.*?)(<Gold Answer>.*?)(?=\n\n|$)'
        else:
            pattern = r'(\d+)-(\d+[a-z]{2}\squestion)\n(<Question>.*?)(<Answer>.*?)(<Gold Answer>.*?)(<Gold Entity>.*?)(?=\n\n|$)'
        matches = re.findall(pattern, corpus, re.DOTALL)

    
        for match in matches:
            res = []

            # parse entity id and question(prompt) id
            entity_id = match[0]
            question_id = match[1].replace('th question', '')

            # parse question
            if 'rag' in log_path:
                question_pattern = r'<Question>\s*(.*?)\n\nA:'
            else:
                question_pattern = r'<Question>\s*Q: (.*?) Provide'
            question = re.findall(question_pattern, match[2], re.DOTALL)[0]
            
            # parse choices
            option_pattern = r'Option (\d): (.+)'
            options = re.findall(option_pattern, match[2], re.MULTILINE)
            num_options= len(options)
            options_joined = list(map(lambda x: f"Option {x[0]}: {x[1]}", options))


            # parse model response
            response = match[3].replace('<Answer>', '').replace("\n\n", "\n").replace("\n", " ").strip()
            if response == 'RESPONSE ERROR':
                continue
            

            # parse model answer in first sentence
            explicit_answer = re.findall(r'[Oo]ption (\d)', response.split("Q:")[0])
            if len(explicit_answer) == 1: # One answer, ideal case
                model_answer = explicit_answer[0]

                # check if the option number is valid
                if model_answer not in [str(i) for i in range(1, num_options+1)]:
                    model_answer = "BINDING ERROR"

            elif len(explicit_answer) >= 1: # Multiple answers
                
                # check if the same option number is repeated
                if len(set(explicit_answer)) == 1:
                    model_answer = explicit_answer[0]
                else:
                    model_answer = "MULTIPLE ANSWER ERROR"

                # check if the answer is "all correct"
                if 'no inaccurate' in response.lower() or 'no incorrect' in response.lower():
                    model_answer = "NONE OF ABOVE"

            elif len(explicit_answer) == 0: # No answer
                # Need manual check!
                model_answer = "NONE OF ABOVE"
                explicit_answer = re.findall(r'[\b\(](\d)[\.\b\)]', response.split("Q:")[0])

                # # check whether option sentence is implicitly mentioned
                # for option_id, option_contents in options:
                #     if option_contents.lower() in response.lower():
                #         model_answer = option_id
                #         break   



            # parse gold answer 
            gold_answer = match[4].replace('<Gold Answer>', '').replace("\n", " ").strip()
            if gold_answer == "There is no answer":
                gold_answer = "NONE OF ABOVE"



            
            # parse gold entity
            if "all_true" in log_path:
                gold_entity = "NONE"
            else:
                gold_entity = match[5].replace('<Gold Entity>', '').replace("\n", " ").replace("=", '').strip()
                
              #  if 'airport' in log_path and gold_answer == '4':
                #    gold_entity = gold_entity.split(',')[0].strip()


            # add to dataframe
            res.append(int(entity_id))
            res.append(int(question_id))
            res.append(question)
            res.append(options_joined)
            res.append(gold_answer)
            res.append(gold_entity)
            res.append(model_answer)
            res.append(response)
            res_df.loc[len(res_df.index)] = res 

        # save dataframe
        res_df.to_csv(csv_path)

    return res_df

def qa_split_llama(log_path):
    '''
    Given log file, return dataframe contains as follows:
    ['question', 'entity_idx', 'question_idx', 'model_answer', 'model_reasoning', 'gold_answer', 'gold_entity']
    '''
    
    # check if log file exists
    if not os.path.exists(log_path):
        raise ValueError(f"Log path does not exist: {log_path}")

   
    # if the log file already processed to csv, just read it
    csv_path = log_path.replace('.log', '.csv')
    '''if os.path.exists(csv_path):
        print("QA Log is already processed. Reading csv file...")
        return pd.read_csv(csv_path)
    '''

    # read log file and process to csv
    res_df = pd.DataFrame(columns=['entity_idx', 'question_idx', 'question', 'choices', 'gold_answer', 'gold_entity', 'model_answer', 'model_response_all'])     
    with open(log_path, 'r') as f:
        
        corpus = f.read()

        # log patterns
        # if all_true, there is no gold entity
        if "all_true" in log_path:
            pattern = r'(\d+)-(\d+[a-z]{2}\squestion)\n(<Question>.*?)(<Answer>.*?)(<Gold Answer>.*?)(?=\n\n|$)'
        else:
            pattern = r'(\d+)-(\d+[a-z]{2}\squestion)\n(<Question>.*?)(<Answer>.*?)(<Gold Answer>.*?)(<Gold Entity>.*?)(?=\n\n|$)'
        matches = re.findall(pattern, corpus, re.DOTALL)

    
        for match in matches:
            res = []

            # parse entity id and question(prompt) id
            entity_id = match[0]
            question_id = match[1].replace('th question', '')

            # parse question
            question_pattern = r'<Question>\s*Q: (.*?) Provide'
            question = re.findall(question_pattern, match[2], re.DOTALL)[0]
            
            # parse choices
            option_pattern = r'Option (\d): (.+)'
            options = re.findall(option_pattern, match[2], re.MULTILINE)
            num_options= len(options)
            options_joined = list(map(lambda x: f"Option {x[0]}: {x[1]}", options))


            # parse model response
            response_1 = match[3].replace('<Answer>', '').replace("\n\n", "\n").replace("\n", " ").strip()
            if response_1 == 'RESPONSE ERROR':
                continue
            response = match[3].replace('<Answer>','').strip().replace("\n", "newline").strip()

            # parse model answer in first sentence
            response_list = response.split("newline")
            #print(response_list)
            # parse model answer in first sentence

            if len(response_list) >=2:
                if response_list[1] =='':
                    explicit_answer = re.findall(r'[Oo]ption (\d)', response_list[0])
                    explicit_answer_1 = re.findall(r'[\b\(](\d)[\.\):\b]', response_list[0])
                    explicit_answer = explicit_answer + explicit_answer_1
                    if len(explicit_answer) == 0:
                        explicit_answer = re.findall(r'[Oo]ption (\d)', response_list[2])
                else:
                    explicit_answer = re.findall(r'[Oo]ption (\d)', response_list[0])
                    if len(explicit_answer) == 0:
                        explicit_answer = re.findall(r'[Oo]ption (\d)', response_list[1])
            else:
                explicit_answer = re.findall(r'[Oo]ption (\d)', response_list[0])
            #print(explicit_answer)
            """if 'Improving Your Memory' in response:
                print(response_list)
                print(explicit_answer)"""
            if len(explicit_answer) == 1: # One answer, ideal case
                model_answer = explicit_answer[0]

                # check if the option number is valid
                if model_answer not in [str(i) for i in range(1, num_options+1)]:
                    model_answer = "BINDING ERROR"

            elif len(explicit_answer) >= 1: # Multiple answers
                
                # check if the same option number is repeated
                if len(set(explicit_answer)) == 1:
                    model_answer = explicit_answer[0]
                else:
                    model_answer = "MULTIPLE ANSWER ERROR"

                # check if the answer is "all correct"
                if 'no inaccurate' in response.lower() or 'no incorrect' in response.lower():
                    model_answer = "NONE OF ABOVE"

            elif len(explicit_answer) == 0: # No answer
                # Need manual check!
                model_answer = "NONE OF ABOVE"

                # # check whether option sentence is implicitly mentioned
                # for option_id, option_contents in options:
                #     if option_contents.lower() in response.lower():
                #         model_answer = option_id
                #         break   



            # parse gold answer 
            gold_answer = match[4].replace('<Gold Answer>', '').replace("\n", " ").strip()
            if gold_answer == "There is no answer":
                gold_answer = "NONE OF ABOVE"



            
            # parse gold entity
            if "all_true" in log_path:
                gold_entity = "NONE"
            else:
                gold_entity = match[5].replace('<Gold Entity>', '').replace("\n", " ").replace("=", '').strip()
                
              #  if 'airport' in log_path and gold_answer == '4':
                 #   gold_entity = gold_entity.split(',')[0].strip()


            # add to dataframe
            res.append(int(entity_id))
            res.append(int(question_id))
            res.append(question)
            res.append(options_joined)
            res.append(gold_answer)
            res.append(gold_entity)
            res.append(model_answer)
            res.append(response_1)
            res_df.loc[len(res_df.index)] = res 

        # save dataframe
        res_df.to_csv(csv_path)

    return res_df
def qa_split_llama_demo(log_path):
    '''
    Given log file, return dataframe contains as follows:
    ['question', 'entity_idx', 'question_idx', 'model_answer', 'model_reasoning', 'gold_answer', 'gold_entity']
    '''
    
    # check if log file exists
    if not os.path.exists(log_path):
        raise ValueError(f"Log path does not exist: {log_path}")

   
    # if the log file already processed to csv, just read it
    csv_path = log_path.replace('.log', '.csv')
    '''if os.path.exists(csv_path):
        print("QA Log is already processed. Reading csv file...")
        return pd.read_csv(csv_path)
    '''

    # read log file and process to csv
    res_df = pd.DataFrame(columns=['entity_idx', 'question_idx', 'question', 'choices', 'gold_answer', 'gold_entity', 'model_answer', 'model_response_all'])     
    with open(log_path, 'r') as f:
        
        corpus = f.read()

        # log patterns
        # if all_true, there is no gold entity
        if "all_true" in log_path:
            pattern = r'(\d+)-(\d+[a-z]{2}\squestion)\n(<Question>.*?)(<Answer>.*?)(<Gold Answer>.*?)(?=\n\n|$)'
        else:
            pattern = r'(\d+)-(\d+[a-z]{2}\squestion)\n(<Question>.*?)(<Answer>.*?)(<Gold Answer>.*?)(<Gold Entity>.*?)(?=\n\n|$)'
        matches = re.findall(pattern, corpus, re.DOTALL)

    
        for match in matches:
            res = []

            # parse entity id and question(prompt) id
            entity_id = match[0]
            question_id = match[1].replace('th question', '')

            # parse question
            question_pattern = r'<Question>\s*Q: (.*?) Provide'
            question = re.findall(question_pattern, match[2], re.DOTALL)[0]
            
            # parse choices
            option_pattern = r'Option (\d): (.+)'
            options = re.findall(option_pattern, match[2], re.MULTILINE)
            num_options= len(options)
            options_joined = list(map(lambda x: f"Option {x[0]}: {x[1]}", options))


            # parse model response
            response_1 = match[3].replace('<Answer>', '').replace("\n\n", "\n").replace("\n", " ").strip()
            if response_1 == 'RESPONSE ERROR':
                continue
            response = match[3].replace('<Answer>','').strip().replace("\n", "newline").strip()

            # parse model answer in first sentence
            response_list = response.split("newline")
            #print(response_list)
            # parse model answer in first sentence
            #if 'sure' in response.lower():
                #print(response_list[0])
            if response_list[0].lower().startswith("sure") or 'happy to help' in response_list[0].lower():
                if len(response_1.split("Q:")) >=3:
                    response_list2 = response_1.split("Q:")[0] + response_1.split("Q")[1]
                    explicit_answer = re.findall(r'[Oo]ption (\d)', response_list2)
                else:
                    explicit_answer = re.findall(r'[Oo]ption (\d)', response_list[0])
                    explicit_answer_1 = re.findall(r'[\b\(](\d)[\.\):\b]', response_list[0])
                    explicit_answer = explicit_answer + explicit_answer_1
                    if len(explicit_answer) == 0:
                        if len(response_list) >=3:
                            if response_list[1] =='':
                                explicit_answer = re.findall(r'[Oo]ption (\d)', response_list[2])
                                if len(explicit_answer) == 0 and len(response_list) >=5 and response_list[3] == '':
                                    explicit_answer = re.findall(r'[Oo]ption (\d)', response_list[4])
                            else:
                                explicit_answer = re.findall(r'[Oo]ption (\d)', response_list[0])
                                if len(explicit_answer) == 0:
                                    explicit_answer = re.findall(r'[Oo]ption (\d)', response_list[1])
            else:
                if len(response_list) >=3:
                    if response_list[1] =='':
                        explicit_answer = re.findall(r'[Oo]ption (\d)', response_list[0])
                        explicit_answer_1 = re.findall(r'[\b\(](\d)[\.\):\b]', response_list[0])
                        explicit_answer = explicit_answer + explicit_answer_1
                        if len(explicit_answer) == 0:
                            explicit_answer = re.findall(r'[Oo]ption (\d)', response_list[2])
                    else:
                        explicit_answer = re.findall(r'[Oo]ption (\d)', response_list[0])
                        if len(explicit_answer) == 0:
                            explicit_answer = re.findall(r'[Oo]ption (\d)', response_list[1])
                else:
                    explicit_answer = re.findall(r'[Oo]ption (\d)', response_list[0])
            #print(explicit_answer)
             
            #print(response_list)
            #print(explicit_answer)
            if len(explicit_answer) == 1: # One answer, ideal case
                model_answer = explicit_answer[0]

                # check if the option number is valid
                if model_answer not in [str(i) for i in range(1, num_options+1)]:
                    model_answer = "BINDING ERROR"

            elif len(explicit_answer) >= 1: # Multiple answers
                
                # check if the same option number is repeated
                if len(set(explicit_answer)) == 1:
                    model_answer = explicit_answer[0]
                else:
                    model_answer = "MULTIPLE ANSWER ERROR"

                # check if the answer is "all correct"
                if 'no inaccurate' in response.lower() or 'no incorrect' in response.lower():
                    model_answer = "NONE OF ABOVE"

            elif len(explicit_answer) == 0: # No answer
                # Need manual check!
                model_answer = "NONE OF ABOVE"

                # # check whether option sentence is implicitly mentioned
                # for option_id, option_contents in options:
                #     if option_contents.lower() in response.lower():
                #         model_answer = option_id
                #         break   



            # parse gold answer 
            gold_answer = match[4].replace('<Gold Answer>', '').replace("\n", " ").strip()
            if gold_answer == "There is no answer":
                gold_answer = "NONE OF ABOVE"



            
            # parse gold entity
            if "all_true" in log_path:
                gold_entity = "NONE"
            else:
                gold_entity = match[5].replace('<Gold Entity>', '').replace("\n", " ").replace("=", '').strip()
                
              #  if 'airport' in log_path and gold_answer == '4':
                 #   gold_entity = gold_entity.split(',')[0].strip()


            # add to dataframe
            res.append(int(entity_id))
            res.append(int(question_id))
            res.append(question)
            res.append(options_joined)
            res.append(gold_answer)
            res.append(gold_entity)
            res.append(model_answer)
            res.append(response_1)
            res_df.loc[len(res_df.index)] = res 

        # save dataframe
        res_df.to_csv(csv_path)

    return res_df


def qa_split(log_path, model='', num_choices = 4, demo = False):
    if 'gpt' in model or model == 'gemini_v':
        return qa_split_gpt_gemini_v(log_path)
    elif model == 'gemini':
        return qa_split_gemini(log_path)
    elif model == 'mistral':
        if demo ==False:
            if num_choices == 2:
                return qa_split_mistral_2(log_path)
            else:
                return qa_split_mistral_34(log_path)
        else:
            return qa_split_mistral_demo(log_path)
    elif model == 'llama':
        if demo == False:
            return qa_split_llama(log_path)
        else:
            return qa_split_llama_demo(log_path)
    elif model == 'claude':
        return qa_split_claude(log_path)
    else:
        raise ValueError("MODEL NOT YET DEFINED!!")

def check_reasoning(task, mixed_flag, row):
    '''
    Check model reasoning with heuristics for given task.
    '''
    gold_answer = row['gold_answer']
    gold_entity = row['gold_entity']
    model_response = row['model_response_all']

    if mixed_flag:
        if row['gold_entity'] == "NONE":
            return None

   
    # task-specific heuristics
    gold_entity_list = []

    if task == 'soccer':
        gold_entity_list.append(unidecode(gold_entity))
        gold_entity_list.append(gold_entity.replace(' CF', ''))
        gold_entity_list.append(gold_entity.replace(' FK', ''))
        gold_entity_list.append(gold_entity.replace('FC ', ''))
        gold_entity_list.append(gold_entity.replace('AC ', ''))
        gold_entity_list.append(gold_entity.replace('RC ', ''))
        gold_entity_list.append(gold_entity.replace('U.C. ', 'U.C.'))
        gold_entity_list.append(gold_entity.replace(' de ', ' '))
        gold_entity_list.append(gold_entity.replace('Al ', ''))
        gold_entity_list.append(gold_entity.replace(' SK', ''))
        gold_entity_list.append(gold_entity.replace('SL ', ''))
        gold_entity_list.append(gold_entity.replace('SV ', ''))
        gold_entity_list.append(gold_entity.replace(' City', ''))
        gold_entity_list.append(gold_entity.replace('1. ', ''))
       

        if "Bayern München" in gold_entity:
            gold_entity_list.append("Bayern Munich")

        if 'Leverkusen' in gold_entity:
            gold_entity_list.append('Bayer Leverkusen')

        if 'Real Betis' in gold_entity:
            gold_entity_list.append('Real Betis')

        if "Hoffenheim" in gold_entity:
            gold_entity_list.append("Hoffenheim")

        if "Bilbao" in gold_entity:
            gold_entity_list.append("Bilbao")

        if "Madrid" in gold_entity:
            gold_entity_list.append("Madrid")

        if "Tigres" in gold_entity:
            gold_entity_list.append("Tigres")   

        if "Montpellier" in gold_entity:
            gold_entity_list.append("Montpellier")

        if "Lille" in gold_entity:
            gold_entity_list.append("Lille")  

        if "Spain" in gold_entity:
            gold_entity_list.append("Spanish")

        if "Liga MX" in gold_entity:
            gold_entity_list.append("Liga MX")



    elif task == 'airport':
        if gold_answer in ['2', '3']: # longitudes and latitudes
            # round ver
            gold_entity_list.append("{:.2f}".format(float(gold_entity)))
            # no round ver
            splitted = gold_entity.split('.')
            splitted[-1] = splitted[-1][:2]
            gold_entity_list.append('.'.join(splitted))

        if gold_answer == '4': # country code (e.g. CL, Chile)
            gold_entity_list.append(gold_entity.split(',')[0].strip())
            gold_entity_list.append(gold_entity.split(',')[1].strip())
            if 'United States' in gold_entity.split(',')[1]:
                gold_entity_list.append('United States')



    elif task == 'movie':
        if gold_answer == '1': # movie director name
            if ' ' in gold_entity:
                gold_entity_list.append(gold_entity.split(' ')[0].strip())
                gold_entity_list.append(gold_entity.split(' ')[-1].strip())

        if gold_entity == 'USA':
            gold_entity_list.append('United States')
            gold_entity_list.append('American')
            gold_entity_list.append('English')

        if gold_entity == 'UK':
            gold_entity_list.append('United Kingdom')
            gold_entity_list.append('British')

        if gold_entity == 'China':
            gold_entity_list.append('Chinese')

        if gold_entity == 'France':
            gold_entity_list.append('French')

        if gold_entity == 'non-animation':
            gold_entity_list.append('animation')
            gold_entity_list.append('animated')
            gold_entity_list.append('action')
            gold_entity_list.append('drama')
            gold_entity_list.append('comedy')
            gold_entity_list.append('crime')
            gold_entity_list.append('thriller')
            gold_entity_list.append('advanture')
            gold_entity_list.append('fantasy')
            gold_entity_list.append('horror')
            gold_entity_list.append('romance')
            gold_entity_list.append('sci-fi')
            gold_entity_list.append('science')
            gold_entity_list.append('documentary')
            gold_entity_list.append('western')
            gold_entity_list.append('disaster')
            gold_entity_list.append('superhero')
            

        if gold_entity == 'animation':
            gold_entity_list.append('animated')




    elif task == 'music':
        if gold_answer == '2': # genre
            gold_entity_list.append(gold_entity.split('/')[0].strip())
            gold_entity_list.append(gold_entity.split('/')[1].strip())

    
    
    elif task == 'book':
        if gold_answer == '3': # publisher
            gold_entity_list.append(gold_entity.split(' ')[0].strip())
            

    
    # check if gold entity is in model response with heuristics
    gold_entity_list.append(gold_entity)
    model_response = model_response.lower()
    for entity in gold_entity_list:
        if entity.lower() in model_response:
            return True
    
    return False



def analyze_result(task, df, save_path, invalid_ids = None, common = False):
    '''
    Save QA results as txt file and csv file.
    '''
    output = {}


    # df config
    output['num_total_entity'] = str(df['entity_idx'].nunique())

    # exclude invalid ids
    if invalid_ids is not None:
        # output['num_invalid_entity'] = str(len(invalid_ids))
        output['num_valid_entity'] = str(df['entity_idx'].nunique() - len(list(filter(lambda x: x < df['entity_idx'].nunique(),invalid_ids))))
        df.drop(df[df['entity_idx'].isin(invalid_ids)].index, inplace=True)
        if len(df) == 0 or df['entity_idx'].nunique() == 0:
            print("No valid data")
            return
    # num of questions
    output['num_questions'] = str(df.shape[0])
    binding_acc = (df['model_answer'] == 'BINDING ERROR').mean() 
    output['binding_error_rate'] = round(binding_acc, 2)


    # total acc
    qa_acc = (df['model_answer'] == df['gold_answer']).mean() 
    output['qa_acc'] = round(qa_acc, 2)

    # prompt ver. acc
    prompt_list = df['question_idx'].unique()
    for prompt in prompt_list:
        prompt_df = df.loc[df['question_idx'] == prompt]
        prompt_qa_acc = (prompt_df['model_answer'] == prompt_df['gold_answer']).mean() 
        output[f"prompt_{prompt}_qa_acc"] = round(prompt_qa_acc, 2)

    mixed_flag = ("mixed" in save_path)

    # rationale acc
    df.loc[: ,'model_reasoning'] = df.apply(lambda row: 1 if check_reasoning(task, mixed_flag, row) else 0, axis=1)
    rationale_acc = df['model_reasoning'].mean() 
    output['rationale_acc'] = round(rationale_acc, 2)

    # prompt ver. rationale acc
    for prompt in prompt_list:
        prompt_df = df.loc[df['question_idx'] == prompt]
        prompt_rationale_acc = prompt_df['model_reasoning'].mean() 
        output[f"prompt_{prompt}_rationale_acc"] = round(prompt_rationale_acc, 2)


    # Accuracy for both correct answer and correct rationale
    output['answer and rationale accuracy'] = round(df.loc[df['model_answer'] == df['gold_answer']]['model_reasoning'].mean() *  (df['model_answer'] == df['gold_answer']).mean(), 2)

    # Compute hallucination rate
    h_words = ['additional', 'impossible', 'further', 'enough', 'specific', 'actual', 'additional', 'cannot', 'sorry']
    df1 = df
    df = df1.copy(deep= True)
    df = df[df['model_answer']== 'NONE OF ABOVE']
    if len(df) ==0:
        h = 1 - (float(qa_acc))
    else:
        r_wrong = df[df['model_response_all'].apply(lambda x: any([y in x.lower() for y in h_words]))]
        h = 1 - (float(qa_acc)) - (len(r_wrong)/int(output['num_questions']))

    output['hallucination rate'] = round(h,2)

    # save df and txt file
    if invalid_ids is None: # without invalid ids
        df1.to_csv(save_path.replace('.txt', '.csv'))
        with open(save_path, 'w') as f:
            json.dump(output, indent=4, fp = f)
    else:
        if common == True: # for union of invalid ids from other models
            df1.to_csv(save_path.replace('.txt', '_common.csv'))
            with open(save_path.replace('.txt', '_common.txt'), 'w') as f:
                json.dump(output, indent=4, fp = f)            
        else: # with invalid ids
            df1.to_csv(save_path.replace('.txt', '_validated.csv'))
            with open(save_path.replace('.txt', '_validated.txt'), 'w') as f:
                json.dump(output, indent=4, fp = f)
    return               




def proceed_validation(val_path):
    '''
    Analyze validation log file and save it as csv file.
    We consider QA pairs as invalid (i.e., model does not have internal knowledge of given entity) 
    if the model response includes "no" in the answer.
    '''

    # check if validation log exists
    log_path = val_path.replace('.csv', '.log')
    if not os.path.exists(log_path):
        raise ValueError(f"Log file does not exist: {log_path}")
    
    # process log file as csv
    with open(log_path, 'r') as f:
        corpus = f.read()
        pattern = r'(\d+)-(\d+[a-z]{2}\squestion).*?(<Question>.*?)(<Answer>.*?)(?=\n<|\n\d|$)'
        
        matches = re.findall(pattern, corpus, re.DOTALL)

        # create dataframe to process as csv
        res_df = pd.DataFrame(columns=['entity_idx', 'question_idx', 'question', 'model_response'])     
        
        # matched patterns in whole log file
        for match in matches:
            res = []

            # parse entity id, question id, question, answer
            entity_id = match[0]
            question_id = match[1].replace('th question', '')
            if 'rag' in log_path:
                question = re.match(r'<Question>\s+(.*?)\nA:\n\n', match[2], re.DOTALL).group(1).strip()
            else:
                question = re.match(r'<Question>\s+Q:(.*?)\nA:\n\n', match[2], re.DOTALL).group(1).strip()
            answer = match[3].replace('<Answer>', '').replace("\n\n", "\n").replace("\n", " ").replace('=', '').strip()

            # heuristics: 
            # - if "no" is included in answer, it is invalid
            pattern_no = r'\bno\b'

            if re.search(pattern_no, answer, re.IGNORECASE):
                res.append(int(entity_id))
                res.append(int(question_id))
                res.append(question)
                res.append(answer)
                res_df.loc[len(res_df)] = res

        
        res_df.to_csv(val_path)

   

def get_invalid_ids(num_choices, valid_path):
    '''
    From validation file, get invalid ids (i.e. idx of unknown entities of given model).
    '''
    valid_df = pd.read_csv(valid_path)

    # check validity w.r.t. option number
    valid_df = valid_df.loc[valid_df['question_idx'] <= num_choices]

    # get invalid ids
    invalid_ids = valid_df['entity_idx'].unique()

    return invalid_ids




def parse_args():
    '''
    Config
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices = ['movie', 'soccer', 'airport', 'book', 'music'], required = True)
    parser.add_argument("--mixed", type=float, default=0.0, help='if set to [0, 1), mix all_true option for given probability')
    parser.add_argument("--model", type=str, choices = ['gpt35', 'gpt4', 'mistral', 'llama', 'gemini', 'gemini_v', 'claude', 'gpt_v'], required=True)
    parser.add_argument("--rag", action='store_true', default=False)
    parser.add_argument("--demo", action='store_true', default=False )
    parser.add_argument("--only_val", "-v", action='store_true', default=False)
    parser.add_argument("--only_common", "-c", action='store_true', default=False)

    args = parser.parse_args()

    return args



def main():
    # config
    args = parse_args()
    task = args.task.strip()
    model = args.model
    demo = args.demo
    valid_dir = '../dataset/validated'
    log_dir = '../results'

    # set maximum number of choices
    MAX_CHOICES = {'movie': 3, 'soccer': 4, 'airport': 4, 'music': 2, 'book': 3}
    num_choices = MAX_CHOICES[task]
    print(f"Running to analyze {task} analysis with {num_choices} options, mixed = {args.mixed}, rag = {args.rag}")


    # get validation file path
    val_csv_path = get_savename(valid_dir, task, demo=args.demo, rag=args.rag, model = model, endswith='.csv')
    if args.rag:
        val_csv_path = val_csv_path.replace('_rag.csv', '.csv')
    elif args.demo:
        val_csv_path = val_csv_path.replace('_demo.csv', '.csv')

    # If validation file does not exist, proceed validation and save csv file
    if not os.path.exists(val_csv_path):
        print(f"Proceeding validation... ")
        proceed_validation(val_csv_path)



    # get indices to exlcude from validation file
    print(f"Reading validation file...")
    invalid_ids = get_invalid_ids(num_choices, val_csv_path)
    print(f"Number of invalid ids in {model}: {len(invalid_ids)}")

 

    # process log file to save as csv
    # If already processed, read csv file
    print("Processing QA log file...")
    model_log_path = get_savename(log_dir, task, num_choices = num_choices, mixed = args.mixed, model = model, rag = args.rag, demo = demo)
    
    if not os.path.exists(model_log_path):
        raise ValueError(f"Log path does not exist: {model_log_path}")
    else:
        print(f"Reading log file... {model_log_path}")
    
    
    # process log file to save as csv
    #csv_path = model_log_path.replace('.log', '.csv')
    
    print("Processing QA log file...")
    df = qa_split(model_log_path, model, num_choices, demo=demo)

    # analyzed result and save it
    print("Analyzing result...")
    result_file_path = model_log_path.replace('.log', '.txt')

    if args.only_val: # save only validation type (2)
        print("Processing with own validation...")
        analyze_result(task, df, result_file_path, invalid_ids) # with invalid ids

    elif args.rag:
        print("Processing with RAG...")

        # get common valid ids from non-rag and rag
        total_num = 1485 if task == 'movie' else 1500
        orig_valid = set(range(0, total_num)) - set(invalid_ids)
        rag_valid = set(df['entity_idx'].unique())
        common_valid = orig_valid.intersection(rag_valid)
        invalid_ids = set(range(0, total_num)) - set(common_valid)
        print(f"Changed number of invalid ids in {args.model}: {len(invalid_ids)}")

        # analyze for rag
        analyze_result(task, df, result_file_path.replace('.txt', '_validated.txt'), invalid_ids)


    elif args.only_common: # save only validation type (3)
        print("Processing with common validation...")
        invalid_ids = pd.read_csv(f"../dataset/validated/union_total/{task}.csv")["invalid_ids"].to_list()
        print(f"Changed number of invalid ids in {args.model}: {len(invalid_ids)}")
        analyze_result(task, df, result_file_path, invalid_ids, common = True) # with invalid ids
    
    else: # save validation type (1), (2), (3)
        df1 = df.copy(deep = True)
        df2 = df.copy(deep = True)

        common_invalid_ids = pd.read_csv(f"../dataset/validated/union_total/{task}.csv")["invalid_ids"].to_list()
        
        analyze_result(task, df, result_file_path) # without invalid ids
        analyze_result(task, df1, result_file_path, invalid_ids) # with invalid ids
        analyze_result(task, df2, result_file_path, common_invalid_ids, common = True)

    print("Done saving results.")



if __name__  =="__main__":
    main()