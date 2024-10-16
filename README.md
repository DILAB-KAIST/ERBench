# ERBench
## Binary Tasks
### How to Run
#### binary/run_qa.py
- Description
    
    This python file triggers the data preprocessing steps using the data in the in the {task}_data/crafted folders and runs the models (GPT, Gemini, Llama, Mistral, Claude). The output of the program will be saved under the results folder.

- Arguments
    - task
        
        the dataset that one ought to test
    - tasktype

        validation step or the main test step
    - index
    
        LLM APIs tend to abort their program due to various issues such as timeout errors or sensitive content errors. Then the user should add the corresponding index to the code to skip the index. After that give the corresponding index as the parameter such that the model can continue its testing from the index
    - demo
        
        Used for few-shot (parameter value: demo) or Chain-of-Thought Prompting (parameter value: cot)
    - rag
    
        Used for Retrieval Augmented Generation (Wikipedia), if wanting to enable RAG, give True as the parameter value
    - model
    
        the model that one ought to test 

#### binary/error_analysis.py

This python file returns the numerical analyses results based on the output log files outputted by run_qa.py.  Four lines of output will be shown in the terminal, each corresponding to A, R, AR, H. The first numeric result for each row is the value for the basic prompt and the second result is the value for the negated prompt.

- Arguments
    - task
        
        the dataset that one ought to test
    - demo
    
        Used for few-shot (parameter value: demo) or Chain-of-Thought Prompting (parameter value: cot)
    - model
    
        the model that one ought to test 
    - rag
    
        Used for Retrieval Augmented Generation (Wikipedia), if wanting to enable RAG, give True as the parameter value

#### binary/finetune_dataset.py
This python file creates the dataset needed for finetuning for GPT models.
- Arguments
    - n
    
        the number of data points that one ought to use
    
    If the user wants to modify the datasets that will be used for finetuning, change line 313 datasets parameter.

#### finetune.ipynb
This python file executes finetuning based on the dataset created by finetune_dataset.py.

#### correctness_ver.py
This python file verifies the correctness of ERBench compared to human analyses and GPT-Judge. Note that some values are omiotted in the paper due to small sample size (less than 4) and entity resolution problems were dealt manually.

### Experiment Procedure

#### General Tasks
```
python run_qa.py --model [MODEL] --task [TASK] --tasktype validate
python run_qa.py --model [MODEL] --task [TASK]
python error_analysis.py --model [MODEL] --task [TASK]
python correctness_ver.py --model [MODEL] --task [TASK]
```

#### Finetuning
```
python finetune_dataset.py --n [N]
run finetune.ipynb via ipynb kernel
python run_qa.py --model [FINETUNED_MODEL] --task [TASK] --tasktype validate
python run_qa.py --model [FINETUNED_MODEL] --task [TASK]
python error_analysis.py --model [FINETUNED_MODEL] --task [TASK]
```
finetune_dataset.py -> finetune.ipynb -> run_qa.py -> error_analysis.py

### Images for Multimodal Models (Gemini Vision Pro)

https://drive.google.com/drive/folders/1WXCGCG4ZPzkV1qUjR2Z0IegzkPgSSKwl?usp=drive_link 


---


## Multi-choice (MC) Tasks

> ### How to generate MC question templates with your DB

(1) Prepare your FD based on your DB schema
    
movie title, released year $\rightarrow$ director, length, movie publisher

(2) Generate your question and corresponding options

- Include left-hand side attribute values in the question

    ```What is the false option about the movie {movie title} released in {year}? Provide an explanation.```

- Convert each right-hand side attribute value to each option

    ```Option A: It was directed by {director}.```

    ```Option B: The runtime of the movie is {length} minutes.```

    ```Option C: The production company of this movie is {movie publisher}.```

- Note that one of your options will be using a **FALSE** attribute value (e.g., different director name), which makes the option as the answer of this MC question. Please refer to *run_qa* function in run_qa.py. 

> ### How to run
#### (1) multi_choice/source/run_qa.py
- Description

    This python file (1) preprocess dataset and (2) run QA/validation task.
    
    (1) preprocessing 

     Use dataset/crafted to reproduce the results in the paper. You should define your preprocessing function if you want to use your own database.

    (2) running tasks
        
    - QA

        Run main QA tasks with LLMs. The output of the program (log file) will be saved under the results folder.

        For example,

        ```python run_qa.py --task movie --model gpt35 --tasktype multiqa```

    - Validation
    
        Run validation tasks with LLMs. The output of the program (log file) will be saved under dataset/validated folder. 

        For example,

        ```python run_qa.py –task movie –model gpt35 –tasktype validate```
- Argument
    -	task
        
        choose dataset (movie/soccer/airport/music/book)
    -	tasktype
        
        choose QA or validation task (multiqa/validate)
    -	index
    
        entity id to resume code. This is useful when API for LLM fail during the code. When specified, skip the entities before the given index.
    -	mixed
    
        [0, 1) proportion to mix questions of None-of-above type and normal type. For example, 0.2 means 20% question is None-of-above type.
    -	random_seed
    
        random seed to reproduce results.
    -	model
    
        choose model (gpt35/gpt4/mistral/llama/gemini/claude/gemini_v/gpt_v)
    -	rag
    
        run QA with knowledge augmentation (RAG). Before running code with RAG mode, you should run your normal type QA first. 
    -	demo
    
        run QA with few-shot demonstrations. Before running code with demo mode, make sure you have demonstrations in dataset/demo.
	
#### (2) multi_choice/source/error_analysis.py
-	Description

    This python file must run after run_qa.py. This code (1) process validation log file to dataframe (.csv), (2) process QA log file to dataframe (.csv) and (2) analyze performance metrics w.r.t. these processed dataframes.

    (1) Processing validation log file

    Make sure you have validation log file from run_qa.py. The output of the program (csv) will be saved under the dataset/validated folder.

    (2)	Processing QA log file

    Make sure you have QA log file from run_qa.py. The output of the program (csv) will be saved under the results folder.

    (3)	Analyzing performances

    We provide 3 modes w.r.t. validation output when analyzing QA output: (a) consider all QA pairs (no validation) (b) consider QA pairs w.r.t. given LLM’s own valid entities (c) consider QA pairs w.r.t. all LLMs’ valid entities. We refer “valid entities” as “entities that LLM already knows”. Please refer to our paper for more details.

    ```
    python error_anlaysis.py --task movie --model gpt35 # get type (a), (b), (c) at once 
    python error_anlaysis.py --task movie --model gpt35 --only_val # get type (b) only
    python error_anlaysis.py --task movie --model gpt35 --only_common # get type (c) only
    ```

-	Arguments
    -   task
    
        choose dataset (movie/soccer/airport/music/book)
    -	mixed
    
        [0, 1) proportion to mix questions of None-of-above type and normal type. For example, 0.2 means 20% question is None-of-above type.
    -	model
    
        choose model (gpt35/gpt4/mistral/llama/gemini/gemini_v)
    -	rag
    
        run QA with knowledge augmentation (RAG). Before running code with RAG mode, you should run your normal type QA first. 
    -	demo
        
        run QA with few-shot demonstrations. Before running code with demo mode, make sure you have demonstrations in dataset/demo.
    -	only_val
    
        save only type (2) analysis results.
    -	only_common
    
        save only type (3) analysis results.

> ### Experiment Procedure
```
python run_qa.py --model [MODEL] --task [TASK] --tasktype validate
python run_qa.py --model [MODEL] --task [TASK]
python error_analysis.py --model [MODEL] --task [TASK]
```



