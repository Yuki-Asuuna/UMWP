import os
import openai
import argparse
import backoff
import torch
from tqdm import tqdm
import jsonlines
from tenacity import retry, stop_after_attempt, wait_random_exponential
from StandardDataset import StandardDataset, StandardDatasetExample
from transformers import AutoTokenizer, AutoModelForCausalLM
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

GPT_list = ["ada", "babbage", "babbage-002", "curie", "davinci", "davinci-002", "text-ada-001", "text-babbage-001",
            "text-curie-001",
            "text-davinci-001", "text-davinci-002",
            "text-davinci-003"]
ChatGPT_list = ["gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613", "gpt-4-0314", "gpt-4-0613"]
# LLaMAv2
# llama_list = ["llama-7b", "llama-13b", "llama-30b", "llama-65b", "alpaca-7b", "alpaca-13b", "vicuna-7b", "vicuna-13b"]
model_dict = {"llama-7b": "decapoda-research/llama-7b-hf", "llama-13b": "decapoda-research/llama-13b-hf",
              "llama-30b": "decapoda-research/llama-30b-hf", "llama-65b": "decapoda-research/llama-65b-hf",
              "alpaca-7b": "chavinlo/alpaca-native", "alpaca-13b": "chavinlo/alpaca-13b",
              "vicuna-7b": "eachadea/vicuna-7b-1.1", "vicuna-13b": "eachadea/vicuna-13b-1.1"}
llama_list = ["llama-7b", "llama-13b", "llama-30b", "llama-65b", "llama-v2-7b", "alpaca-13b", "vicuna-7b", "vicuna-13b"]
claude_list = ["claude-1", "claude-2"]
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()

parser.add_argument("--API-Key", type=str, help="OpenAI API Key")
parser.add_argument("--input-form", type=str, default="Direct", choices=["Direct", "Instruction", "ICL"],
                    help="Input Form")
parser.add_argument(
    "--model-name",
    type=str,
    default="ada",
    choices=ChatGPT_list + GPT_list + llama_list + claude_list,
    help="Model for testing",
)
parser.add_argument("--temperature", default=0.7, type=float, help="Temperature when generating")
args = parser.parse_args()

Direct_Prompt = "Q:{}\nA:"
Instruction_Prompt = """You are a skilled mathematician. Please analyze the following math problem step by step. If you find any missing or ambiguous information, impractical conditions, or incomplete statements that prevent a solution, conclude with "Thus, this problem cannot be answered."
Q: {}
A:"""
ICL_Prompt = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.
Q: Shawn has five toys. For Christmas, he got several toys each from his mom and dad. How many toys does he have now?
A: Since we don't have any information about the number of toys given by his mom or dad, we can't determine a specific numerical value for the total number of toys. Thus, this problem cannot be answered.
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did Jason have at the end of wednesday?
A: Since no information has been provided about Jason in the problem statement, we cannot answer the question regarding Jason. Thus, this problem cannot be answered.
Q: Olivia has $23. She bought five bagels. How much money does she have left?
A: Without knowing the cost of a single bagel, we cannot determine how much money Olivia spent on the 5 bagels, and therefore cannot determine how much money she has left. Thus, this problem cannot be answered.
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.
Q: There were -9 computers in the server room. 6 more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: Having -9 computers is not practically possible in real life. The negative value can't represent a physical quantity of objects. Thus, this problem cannot be answered.
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.
Q: {}
A:"""


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def sendChatGPTRequest(input_context, model_name="gpt-3.5-turbo", temperature=0.7):
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are an excellent question responder.",
            },
            {
                "role": "user",
                "content": input_context,
            },
        ],
        temperature=temperature,
    )
    return response["choices"][0]["message"]["content"]


# GPT-3
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def sendGPTRequest(input_context, model_engine, temperature=0.7):
    prompt = input_context

    # GPT3
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=256,
        n=1,
        stop=None,
        temperature=temperature,
        presence_penalty=0.5,
    )

    return response.choices[0].text


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(500))
def sendClaudeRequest(input_context: str, api_key: str, model_engine: str = "claude-2"):
    anthropic = Anthropic(api_key=api_key)
    completion = anthropic.completions.create(
        model=model_name,
        max_tokens_to_sample=300,
        prompt=f"{HUMAN_PROMPT} {input_context} {AI_PROMPT}",
    )
    return completion.completion


def generate_input_context(question, input_form):
    if input_form == "Direct":
        input_context = Direct_Prompt.format(question)
    elif input_form == "Instruction":
        input_context = Instruction_Prompt.format(question)
    elif input_form == "ICL":
        input_context = ICL_Prompt.format(question)
    return input_context


if __name__ == '__main__':
    s = StandardDataset()
    s = s.FromJSONLFile("StandardDataset.jsonl")
    openai.api_key = args.API_Key
    input_form = args.input_form
    model_name = args.model_name
    temperature = args.temperature

    if model_name in llama_list:
        model = AutoModelForCausalLM.from_pretrained(model_dict[model_name]).half().cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_dict[model_name])

    file_path = "{}_{}_{}_T_{}.jsonl".format(model_name, input_form, model_name, temperature)
    if not os.path.exists(file_path):
        with open(file_path, mode="w") as f:
            pass  # 创建空文件

    for item in tqdm(s.FetchAll()):
        # print("QID:{} Q:{} A:{}".format(item.id, item.question, sendChatGPTRequest(item.question)))
        # print("QID:{} Q:{} A:{}".format(item.id, item.question, item.answer))
        input_context = generate_input_context(item.question, input_form)
        generated_text = ""
        if model_name in ChatGPT_list:
            generated_text = sendChatGPTRequest(input_context, model_name=model_name, temperature=temperature)
        elif model_name in GPT_list:
            generated_text = sendGPTRequest(input_context, model_engine=model_name, temperature=temperature)
        elif model_name in claude_list:
            generated_text = sendClaudeRequest(input_context, api_key="", model_engine=model_name)
        elif model_name in llama_list:
            input_ids = tokenizer.encode(input_context, return_tensors="pt").to(device)
            output = model.generate(input_ids, temperature=temperature, num_return_sequences=1, max_length=1024)
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)[len(input_context):]

        with jsonlines.open(file_path, mode="a") as writer:
            writer.write(
                {"id": item.id, "question": item.question, "answer": item.answer, "answerable": item.answerable,
                 "category": item.category, "relevant_ids": item.relevant_ids, "source": item.source,
                 "generated_text": generated_text})
