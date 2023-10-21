# UMWP

![License](https://img.shields.io/badge/License-Apache%20License%202.0-green)![Data License](https://img.shields.io/badge/Data%20License-CC--BY--SA--4.0-blue)



## Introduction

![](/assets/intro.png)



## Dataset

The *UMWP* dataset is located in the [StandardDataset.json](data/StandardDataset.json) file. 

When using it, please adhere to the CC-BY-SA-4.0 license. 

Below are two examples of the data:

**Answerable Question:**

```json
{
    "id": 1,
    "question": "Bryan took a look at his books and magazines. If he has 9 books and 46 magazines in each of his 10 bookshelves.How many magazines does he have in total?",
    "answer": [
        460.0
    ],
    "answerable": true,
    "category": null,
    "relevant_ids": null,
    "source": "SVAMP"
}
```



**Unanswerable Question:**

```json
{
    "id": 3226,
    "question": "At the arcade Dave won more than 11 tickets. If he spent 5 tickets on a beanie and later won 10 more tickets, how many would he have? ",
    "answer": null,
    "answerable": false,
    "category": 2,
    "relevant_ids": [
        726
    ],
    "source": "MultiArith"
}
```



| **Attribute** | **Type** | **Description**                                              |
| ------------- | -------- | ------------------------------------------------------------ |
| question_id   | Integer  | Question ID                                                  |
| question      | String   | Description                                                  |
| answer        | List     | Answer                                                       |
| answerable    | Bool     | Answerable or Unanswerable                                   |
| relevant_ids  | List     | Relevant Question ID                                         |
| category      | Integer  | If it's an Answerable Question, then the category is set to 0. <br />If it's an Unanswerable Question, the category takes values from 1 to 5. |
| source        | String   | Data Source                                                  |



## Installation

Python 3.9

```bash
conda create -n UMWP python=3.9
conda activate UMWP
pip install -r requirements.txt
```



## Run

Here is an example of generating the output of the `gpt-3.5-turbo-0613` model under the ICL input form with `Temperature=0.7`. `sk-xxx` is your openAI API-KEY.

```bash
python run.py --input-form ICL --model-name gpt-3.5-turbo-0613 --temperature 0.7 --API-Key sk-xxx 
```

There are three input forms: Direct, Instruction, and ICL.

Available models are listed in the `run.py`. You are free to add your own model.



## Evaluation

Here is an example of evaluating the output of the `text-davinci-003` under the Direct input form with `Temperature=0.7`.

```bash
python eval.py --filename text-davinci-003_Direct_text-davinci-003_T_0.7.jsonl
```

Another Example:

```bash
python eval.py --filename llama-v2-13b-chat_ICL_T_0.7.jsonl
```