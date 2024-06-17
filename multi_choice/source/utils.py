'''
Utils for Multiple-choice QA.
'''

import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from google.ai import generativelanguage as glm
import PIL.Image
import urllib.request
from openai import AzureOpenAI
import os
import json
import anthropic
import base64

# Log file management ==========================================
def get_savename(save_dir, task, num_choices=None, model='', mixed = 0, demo = False, rag = False, endswith = '.log'):
    '''
    Given save_dir and task, return savename of gpt-3.5 and gpt-4 log.
    num_choices, all_true, hard is optional.

    Returns:
    gpt35_log: save_dir/gpt35/task.log
    gpt4_log: save_dir/gpt4/task.log
    '''
    if model not in {'gpt35', 'gpt4', 'llama', 'mistral', 'gemini', 'gemini_v', 'claude', 'gpt_v'}:
        raise ValueError("MODEL NOT YET DEFINED!!")
    model_log = f'{save_dir}/{model}/{task}'

    if num_choices is not None:
        model_log += f'/choice_{num_choices}'

    if demo:
        model_log += f'_demo'

    if rag:
        model_log += f'_rag'

    if mixed > 0:
        gpt35_log += f'_mixed_{mixed}'

    model_log += endswith

    return model_log


def is_number(value):
    flag = True
    try:
        n = float(value)
        flag = (n == n)
    except:
        flag = False
    
    return flag


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def write_to_file(savename, index, prompt, response, answer=None, entity=None):
    with open(savename,'a') as f:
        if index.split('-')[1] =='0':
            f.write("========================================\n")
        f.write(f"{index}th question\n")
        f.write("<Question> \n")
        f.write(prompt)
        f.write("\n\n")
        f.write("<Answer> \n")
        try:
            f.write(response)
        except:
            f.write("")
        f.write("\n\n")

        if answer:
            f.write('<Gold Answer> \n')
            f.write(str(answer))
            f.write("\n\n")

        if entity:
            f.write('<Gold Entity> \n')
            f.write(str(entity))
            f.write('\n\n')

    return


# =============================================================




# Running Gemini-Vision ========================================
def download_image(url, filename):
    '''
    Download image for given url, and save it as filename.
    '''
    response = requests.get(url)

    if response.status_code == 200:
 
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"Image downloaded as {filename}")
    else:
        print("Failed to download image")

def fetch_google_images(queries, folder_name='...', num_images=5):
    '''
    Given queries, fetch images from google and save them in folder_name.
    '''
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

def run_gemini_vision(task, prompt, img_keyword, tasktype='multiqa'):
    '''
    Run Gemini-Vision with image fetched with given keyword, and return response.
    '''
    
    # in case we fetched invalid image
    if img_keyword == 'Undefined Club': 
        return 'No. (Invalid Image)'
    
    # save folder for fetched images
    folder_name = f'../dataset/multimodal/{task}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # fetch image
    if not os.path.exists(f'{folder_name}/{img_keyword}'):
        print('Fetching image...')
        fetch_google_images([img_keyword], folder_name)
    else:
        print('Image already exists')
    img = PIL.Image.open(f'{folder_name}/{img_keyword}/image.jpg')

    # prepare prompt
    if tasktype=='multiqa':
        system_msg = "Answer the following multiple choice question, and then explain why."
    elif tasktype =='validate':
        system_msg = "Answer the following question in yes or no. Be concise."
    msg = [system_msg, prompt, img]


    # run gemini_v
    genai.configure(api_key='...')
    model = genai.GenerativeModel('gemini-pro-vision')
    config = {"temperature" : 0}
    safety_settings = {
            glm.HarmCategory.HARM_CATEGORY_HARASSMENT: glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            glm.HarmCategory.HARM_CATEGORY_HATE_SPEECH: glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            glm.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            glm.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        }
    response = model.generate_content(msg, generation_config = config, safety_settings = safety_settings)


    return response.text
# =============================================================





def run_claude(prompt, tasktype='multiqa'):

    if tasktype == 'validate':
        system_msg = "Answer the following question in yes or no. Be concise"
    elif tasktype == 'multiqa':
        system_msg = "Answer the following multiple choice question, and then explain why."
    elif tasktype == 'multiqa_hint':
        system_msg = "Answer the following multiple choice question, and then explain why. Say 'None of above' if all options are correct."
    elif tasktype == 'multiqa_rag':
        system_msg = "The first passage is the hint, which may not contain all information. In this case, answer the following multiple choice questions which choose the false option with your own knowledge, and then explain why."


    end_point = "https://api.anthropic.com/v1/messages"

    api_key = "..."
    
    payload = {
        "model": "claude-3-sonnet-20240229",
        "temperature": 0.0,
        "max_tokens": 1024,
        "system": system_msg,
        "messages": [
           {"role": "user", "content":prompt}
        ]
    }
    
    headers = {
        "x-api-key": api_key,
        "anthropic-version" : "...",
        "content-type": "application/json",      
    }
    
    response = requests.post(end_point, headers=headers, json=payload)
    return response.json()




# Running GPTs =================================================
def run_gpt35(prompt, tasktype = 'multiqa'):
    '''
    Run GPT-3.5 and return response.
    System message is predefined for validate, multiqa, multiqa_hint (used for all true questions), and rag.
    Temperature is set to 0 to get deterministic response.
    '''

    client = AzureOpenAI( 
        api_version = "...",
        azure_endpoint= "...",
        api_key="..."
    )

    if tasktype == 'validate':
        system_msg = "Answer the following question in yes or no. Be concise"
    elif tasktype == 'multiqa':
        system_msg = "Answer the following multiple choice question, and then explain why."
    elif tasktype == 'multiqa_hint':
        system_msg = "Answer the following multiple choice question, and then explain why. Say 'None of above' if all options are correct."
    elif tasktype == 'multiqa_rag':
        system_msg = "The first passage is the hint, which may not contain all information. In this case, answer the following multiple choice questions which choose the false option with your own knowledge, and then explain why."
   
    msg = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content":prompt},
        ]
    
    completion = client.chat.completions.create(
        model="gpt3_5_2",
        messages = msg,
        temperature = 0)
    
    return completion.choices[0].message.content

def run_gpt4(prompt, tasktype='multiqa'):
    '''
    Run GPT-4 and return response.
    System message is predefined for validate, multiqa, multiqa_hint (used for all true questions), and rag.
    Temperature is set to 0 to get deterministic response.
    '''

    client = AzureOpenAI( 
        api_version = "...",
        azure_endpoint= "...",
        api_key= "..."
    )

    if tasktype == 'validate':
        system_msg = "Answer the following question in yes or no. Be concise"
    elif tasktype == 'multiqa':
        system_msg = "Answer the following multiple choice question, and then explain why."
    elif tasktype == 'multiqa_hint':
        system_msg = "Answer the following multiple choice question, and then explain why. Say 'None of above' if all options are correct."
    elif tasktype == 'multiqa_rag':
        system_msg = "The first passage is the hint, which may not contain all information. In this case, answer the following multiple choice questions which choose the false option with your own knowledge, and then explain why."
    
    msg = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content":prompt},
        ]

    completion = client.chat.completions.create(
        model="...",
        messages = msg, 
        temperature = 0
    )

    return completion.choices[0].message.content
# =============================================================



# Running Mistral/Llama/Gemini =================================================
def run_mistral(prompt, tasktype='multiqa',cot_q=[], cot_a=[], cot=False):
    data = {
        "input_data": {
            "input_string": [],
            "parameters": {
                "max_new_tokens": 300,
                #"do_sample": True,
                "temperature": 0,
                "return_full_text": True
            }
        }
    }
    if tasktype=='normal':
        system_msg = "Answer the following question in yes or no, and then explain why. Say 'unsure' if you don't know and then explain why."
    elif tasktype =='validate':
        system_msg = "Answer the following question in yes or no. Be concise."
    elif tasktype == 'multiqa':
        system_msg = "Answer the following multiple choice question, and then explain why."
    elif tasktype == 'multiqa_hint':
        system_msg = "Answer the following multiple choice question, and then explain why. Say 'None of above' if all options are correct."

    if cot: 
        assert(len(cot_q)==len(cot_a))
        #msg = [
        #    {"role": "system", "content": system_msg}
        #]
        msg = []
        for q,a in zip(cot_q, cot_a):
            msg.append({'role': 'user', 'content': q})
            msg.append({'role': 'assistant', 'content': a})
        msg.append({"role": "user", "content":system_msg + " " + prompt})
    else:
        msg = [
                #{"role": "system", "content": system_msg},
                #{"role": "user", "content":system_msg + ' ' + prompt},
                system_msg + ' ' + prompt
            ]
    data['input_data']["input_string"] = msg
    body = str.encode(json.dumps(data))

    url = '...'
    # Replace this with the primary/secondary key or AMLToken for the endpoint
    api_key = '...'
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    # The azureml-model-deployment header will force the request to go to a specific deployment.
    # Remove this header to have the request observe the endpoint traffic rules
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        print(result)

        result_dict = eval(result.decode('utf8'))[0]
        last_answer = result_dict[str(len(result_dict)-1)]
        return last_answer.split("A:")[-1]
    
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))

def run_llama(prompt, tasktype='multiqa',cot_q=[], cot_a=[], cot=False):
    data = {
        "input_data": {
            "input_string": [],
            "parameters": {
                "temperature": 0,
                #"top_p": 0.9,
                #"do_sample": True,
                "max_new_tokens": 200
            }
        }
    }
    if tasktype=='normal':
        system_msg = "Answer the following question in yes or no, and then explain why. Say 'unsure' if you don't know and then explain why."
    elif tasktype =='validate':
        system_msg = "Answer the following question in yes or no. Be concise."
    elif tasktype == 'multiqa':
        system_msg = "Answer the following multiple choice question, and then explain why."
    elif tasktype == 'multiqa_hint':
        system_msg = "Answer the following multiple choice question, and then explain why. Say 'None of above' if all options are correct."

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

    url = '...'
    # Replace this with the primary/secondary key or AMLToken for the endpoint
    api_key = '...'
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    # The azureml-model-deployment header will force the request to go to a specific deployment.
    # Remove this header to have the request observe the endpoint traffic rules
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key) }

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        #print(result)
        #print(result.decode('ascii'))
        #print(eval(result.decode('ascii'))["output"])
        return eval(result.decode('utf8'))["output"]
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))

def run_gemini(prompt, tasktype='multiqa',cot_q=[], cot_a=[], cot=False):

    if tasktype=='normal':
        system_msg = "Answer the following question in yes or no, and then explain why. Say 'unsure' if you don't know and then explain why."
    elif tasktype =='validate':
        system_msg = "Answer the following question in yes or no. Be concise."
    elif tasktype == 'multiqa':
        system_msg = "Answer the following multiple choice question, and then explain why."
    elif tasktype == 'multiqa_hint':
        system_msg = "Answer the following multiple choice question, and then explain why. Say 'None of above' if all options are correct."
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

    genai.configure(api_key='...')

    model = genai.GenerativeModel('gemini-pro')

    config = {"temperature" : 0, "max_output_tokens" : 200}
    safety_settings = {
        glm.HarmCategory.HARM_CATEGORY_HARASSMENT: glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        glm.HarmCategory.HARM_CATEGORY_HATE_SPEECH: glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        glm.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        glm.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    }
    response = model.generate_content(msg, generation_config = config, safety_settings = safety_settings)

    #print(response)
    #print(msg)
    #print(response.prompt_feedback)

    return response.text

def run_llm(prompt, tasktype='normal',cot_q=[], cot_a=[], cot=False, model='', task ='', img_keyword = ''):
    if model == 'gpt35':
        return run_gpt35(prompt, tasktype)
    elif model == 'gpt4':
        return run_gpt4(prompt, tasktype)
    elif model == 'mistral':
        return run_mistral(prompt, tasktype,cot_q, cot_a, cot)
    elif model == 'llama':
        return run_llama(prompt, tasktype,cot_q, cot_a, cot)
    elif model == 'gemini':
        return run_gemini(prompt, tasktype,cot_q, cot_a, cot)
    elif model == 'gemini_v':
        return run_gemini_vision(task, prompt, img_keyword)
    elif model == 'claude':
        return run_claude(prompt, tasktype)
    elif model == 'gpt_v':
        return run_gpt_vision(task, prompt, img_keyword, tasktype)
    else:
        raise ValueError("MODEL NOT YET DEFINED!!")
    



def gpt4V_encode(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def run_gpt_vision(task, prompt, img_keyword, tasktype='multiqa'):
    '''
    Run GPT-Vision with image fetched with given keyword, and return response.
    '''

    # in case we fetched invalid image
    if img_keyword == 'Undefined Club': 
        return 'No. (Invalid Image)'
    
    # save folder for fetched images
    folder_name = f'../dataset/multimodal/{task}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # fetch image
    # fetch image
    if not os.path.exists(f'{folder_name}/{img_keyword}'):
        print('Fetching image...')
        fetch_google_images([img_keyword], folder_name)
    else:
        print('Image already exists')
    data_url =gpt4V_encode(f'{folder_name}/{img_keyword}/image.jpg')

    # prepare prompt
    if tasktype=='multiqa':
        system_msg = "Answer the following multiple choice question, and then explain why."
    elif tasktype =='validate':
        system_msg = "Answer the following question in yes or no. Be concise."

    # run gpt-v
    end_point = "..."
    api_key = '...'

    headers = {
    "Content-Type": "application/json",
    "api-key": api_key,
    }   
      
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
    # print(response.json())
    return response.json()
    # return response.json()['choices'][0]['message']['content']