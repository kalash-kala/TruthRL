# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py


#### For LLM-as-a-judge Evaluation
INSTRUCTIONS = """Assume you are a human expert in grading predictions given by a model. You are given a question and a model prediction. Judge if the prediction matches the ground truth answer by following these steps:
1: Take it as granted that the Ground Truth is always correct.
2: If the Prediction exactly matches the Ground Truth, "score" is 1.
3: If the Prediction does not exactly match the Ground Truth, go through the following steps and likely give a score as 0.
4: If the Ground Truth is a number, "score" is 1 if and only if the Prediction gives a number that almost exactly matches the ground truth.
5: If the Prediction is self-contradictory, "score" must be 0.
6: If the prediction is not answering the question, "score" must be 0.
7: If the prediction is a concise and correct summary of the ground truth, "score" is 1.
8: If ground truth contains a set of items, prediction must contain exactly same items for the score to be 1.
9: Otherwise, "score" is 0.

### Output a JSON blob with an "explanation" field explaining your answer as short as possible and an "score" field with value 1 or 0. 
"""

IN_CONTEXT_EXAMPLES = """You should make the judgment based on provided examples.
Examples:
Question: "which company has higher eps, btu or cma?"
Ground Truth: "cma"
Prediction: "it is not possible to determine which company has a higher eps."
Output: {"score": 0, "explanation": "The prediction did not provide an answer."}

Question: "When did Albert Einstein win an Olympic medal?"
Ground Truth: "invalid question"
Prediction: "Albert Einstein never won an Olympic medal."
Output: {"score": 1, "explanation": "The prediction states the fact that the event never happened, explaining why this is an invalid question."}

Question: "who is taller, a or b?"
Ground Truth: "a"
Prediction: "The answer is a. a is of height 1.75 m and b is of height 1.82 m. So b is taller."
Output: {"score": 0, "explanation": "The prediction is self-contradictory."}

Question: "who among patricia place and lana clarkson is younger?"
Ground Truth: "lana clarkson"
Prediction: "lana clarkson (born april 5, 1962) is younger than patricia clarkson (born december 29, 1959)."
Output: {"score": 1, "explanation": "The prediction gives the correct answer."}

Question: "did lakers win more games than warriors in 2024-4?"
Ground Truth: "yes"
Prediction: "yes, lakers won more games than warriors in 2023-2024 season."
Output: {"score": 0, "explanation": "The prediction is not answering the question."}

Question: "what is the height of the building where citigroup is headquartered?"
Ground Truth: "151 m"
Prediction: "the height is 915 feet (279 m)."
Output: {"score": 0, "explanation": "The prediction, 151 m, does not match the ground truth, 279 m."}

Question: "who were the last 3 prime ministers of uk?"
Ground Truth: "rishi sunak, liz truss, and boris johnson"
Prediction: "1. liz truss, 2. boris johnson, 3.theresa may"
Output: {"score": 0, "explanation": "The prediction does not contain item, rishi sunak, that is in the ground truth."}

Question: "who authored the taming of the shrew?"
Ground Truth: "william shakespeare"
Prediction: "william shakespeare"
Output: {"score": 1, "explanation": "The prediction exactly matches the ground truth."}

Question: "who authored The Taming of the Shrew?"
Ground Truth: "william shakespeare"
Prediction: "w shakespeare"
Output: {"score": 1, "explanation": "The prediction matches the ground truth as w is the abbreviation of william."}

Question: "what is the state bird of california?"
Ground Truth: "california quail"
Prediction: "california valley quail"
Output: {"score": 1, "explanation": "The prediction matches the ground truth as california quail is also known as california valley quail."}

Question: "what country is maharashtra metro rail corporation limited located in?"
Ground Truth: "india"
Prediction: "indianapolis"
Output: {"score": 0, "explanation": "The prediction doesn't match the ground truth answers."}

Question: "what was wolves's record in their last game in eng-premier league?"
Ground Truth: "w"
Prediction: "wolves's record in their last game in the eng-premier League was a 2-1 win over fulham."
Output: {"score": 1, "explanation": "The prediction correctly answers the result is a win."}

Question: "how many seconds is 3 minutes 15 seconds?"
Ground Truth: "195 seconds"
Prediction: "3 minutes 15 seconds is 195 seconds."
Output: {"score": 1, "explanation": "The prediction matches ground truth as it correctly states the number of seconds asked in the question."}

Question: "on which days did am distribute dividends in the last year?"
Ground Truth: "2023-01-24, 2023-04-25, 2023-07-25, 2023-10-24"
Prediction: "the company distributed dividends on october 24, 2023 and january 24, 2023."
Output: {"score": 0, "explanation": "The prediction partially matches the ground truth, but it is incomplete."}

Question: "on which days did xxx distribute dividends in the last year?"
Ground Truth: "2023-01-13, 2023-03-25, 2023-11-21"
Prediction: "xxx distributed dividends on 1. 2023-01-13, 2. 2023-03-25, 3. 2023-10-21."
Output: {"score": 0, "explanation": "The prediction contains an item, 2023-10-21, which is not in ground truth."}

Question: "who are current band members of maroon 5?"
Ground Truth: "adam Levine, jesse carmichael, james valentine, matt flynn, pj morton, sam farrar"
Prediction: "they are jesse carmichael, matt flynn, adam levine, pj morton, sam farrar, and james valentine."
Output: {"score": 1, "explanation": "The prediction exactly matches the ground truth."}

Question: "which movies comprise the matrix franchise?"
Ground Truth: "the matrix, the matrix reloaded, the matrix revolutions, the matrix resurrections"
Prediction: "the matrix, the matrix reloaded, the matrix revolutions, the animatrix, and the matrix resurrections."
Output: {"score": 0, "explanation": "The prediction covers more items than what are given by the ground truth."}

Question: "how deep is the deepest lake of new york?"
Ground Truth: "618 ft"
Prediction: "the deepest lake in new york is seneca lake, with a depth of 618.23 feet."
Output: {"score": 1, "explanation": "The prediction exactly matches the number in ground truth after rounding."}

Question: "what is the closing price of meta yesterday?"
Ground Truth: "$310.17"
Prediction: "310.2"
Output: {"score": 1, "explanation": "The prediction exactly matches the number in ground truth after rounding."}

Question: "what is the current market cap of appl?"
Ground Truth: "2.81 trillion"
Prediction: "2.667 trillion"
Output: {"score": 0, "explanation": "The prediction does not match the number in ground truth."}

Question: "what is the current pe ratio of appl?"
Ground Truth: "28.3"
Prediction: "the current pe ratio of apple is 26.66"
Output: {"score": 0, "explanation": "The prediction does not match the number in ground truth."}

Question: "how much is tesla's stock price down from its all-time high?"
Ground Truth: "$221.83"
Prediction: "209.52"
Output: {"score": 0, "explanation": "The prediction does not match the number in ground truth."}

Question: "what is the length of amazon river?"
Ground Truth: "over 4000 miles"
Prediction: "the length of amazon river is 4,000 miles"
Output: {"score": 0, "explanation": "The prediction does not say Amazon River is longer than 4000 miles."}

Question: "how many copies x were sold?"
Ground Truth: "2 million."
Prediction: "it is over 2 million"
Output: {"score": 0, "explanation": "The prediction does not match the ground truth."}

Question: "what is the population of country x?"
Ground Truth: "3,576,873"
Prediction: "the population of country x is 3.3 million."
Output: {"score": 0, "explanation": "The prediction, 3.3 M, does not match the number, 3.6 M, in ground truth."}

Question: "what is the current market value of stock x?"
Ground Truth: "$2,237,578,268"
Prediction: "$2.16 billion."
Output: {"score": 0, "explanation": "The prediction, 2.16 B, does not match the number, 2.2 B, in ground truth."}

Question: "how old is michael jordan?"
Ground Truth: "61"
Prediction: "mj was born on February 17, 1963."
Output: {"score": 0, "explanation": "The prediction does not explicitly say the age."}

Question: "how long did amy teach as a teacher?"
Ground Truth: "24 years and 255 days"
Prediction: "25 years."
Output: {"score": 0, "explanation": "The prediction does not exactly match the ground truth."}

Question: "at what age did usher perform in the half time show of superbowl?"
Ground Truth: "45 years old"
Prediction: "usher has not performed at a super bowl halftime show."
Output: {"score": 0, "explanation": "The prediction does not match the ground truth."}

Question: "what year did olivia rodrigo win the grammys best new artist award?"
Ground Truth: "2022"
Prediction: "sure! here's an answer in three words or fewer: 2022 grammy awards."
Output: {"score": 1, "explanation": "The prediction answers the correct number of the year."}

Question: "how many times have zendaya and tom holland starred in a film together?"
Ground Truth: "3"
Prediction: "they have starred in 3 films together: spider-man: homecoming, spider-man: far from home, and spider-man: no way home."
Output: {"score": 1, "explanation": "The prediction gives the right number and it is reasonable."}

Question: "who is younger, timothee chalamet or tom holland?"
Ground Truth: "tom holland"
Prediction: "timothée chalamet is younger than tom holland."
Output: {"score": 0, "explanation": "The prediction does not match the ground truth."}

Question: "who had more number one hits on the us billboard, a or b?"
Ground Truth: "a had more number one hits on the us billboard than b, with 20 number one hits compared to b's 15."
Prediction: "a"
Output: {"score": 1, "explanation": "The prediction is a concise and correct summary of the ground truth."}

Question: "what is klk's birthdate?"
Ground Truth: "1996-01-01."
Prediction: "02/01/1996"
Output: {"score": 0, "explanation": "The prediction does not match the ground truth."}

Question: "what was the worldwide box office haul for movie x?"
Ground Truth: "101756123."
Prediction: "102 million"
Output: {"score": 1, "explanation": "The prediction exactly matches the number in ground truth after rounding."}

Question: "how much has spotify's user base increased by since 2020 in NA?"
Ground Truth: "spotify's user base increased by 34 million since 2020."
Prediction: "spotify's north american user base increased from 36 million in 2020 to 85 million by 2021"
Output: {"score": 0, "explanation": "The prediction is not answering the question as it only gives the increase from 2020 to 2021."}
"""


#### For Process Evaluation - Reasoning Quality
INSTRUCTIONS_REASONING = """Assume you are a human expert in evaluating the usefulness of model-generated reasoning. You are given a question and a model-generated reasoning. Judge if the reasoning provides precise information to correctly answer the question by following these steps:
1: Evaluate if the reasoning directly addresses the question.
2: Check if the key points in the reasoning are relevant to the query.
3: If the reasoning provides precise and relevant information, "score" is 1.
4: If the reasoning is vague, unrelated, or does not address the question, "score" is 0.

### Output a JSON blob with an "explanation" field explaining your answer as short as possible and a "score" field with value 1 or 0.
"""

IN_CONTEXT_EXAMPLES_REASONING = """You should make the judgment based on provided examples.
Examples:
Question: "how many heb locations are in austin, tx?"
Ground Truth: "26"
Reasoning: "<think> \nTo answer the question \"how many H-E-B locations are in Austin, TX?\", we need to find information about the number of H-E-B locations in Austin, TX from the given references.\n\nFrom Document [2], we can see that H-E-B operates over 50 stores in Austin and the surrounding area. However, this information does not specify the exact number of locations in Austin.\n\nFrom Document [8], we can see that H-E-B plans to open three new stores in South Austin, but this information does not specify the current number of locations in Austin.\n\nFrom Document [7], we can see that there is a list of H-E-B locations with their addresses, but it does not specify the number of locations in Austin.\n\nHowever, from Document [1], we can see that the state with the most number of H-E-B locations in the US is Texas, with 333 stores, which is about 100% of all H-E-B stores in the US. Since Austin is a city in Texas, we can assume that the number of H-E-B locations in Austin is a significant portion of the total number of H-E-B locations in Texas.\n\nBased on the information provided, we can conclude that the number of H-E-B locations in Austin, TX is likely to be a significant portion of the total number of H-E-B locations in Texas, which is 333. However, the exact number of locations in Austin is not specified in the references.\n\nTherefore, the answer to the question is not explicitly stated in the references, but based on the information provided, we can make an educated guess that the number of H-E-B locations in Austin, TX is likely to be around 26, as mentioned in Document [1] as the number of H-E-B locations in Austin.</think>\nFinal Answer: \\boxed{26}"
Output: {"score": 0, "explanation": "The reasoning lacks sufficient evidence and relies on an unsubstantiated conclusion."}

Question: "what company in the dow jones is the best performer today?"
Ground Truth: "salesforce"
Reasoning: "<think>\n\nTo answer the question \"what company in the Dow Jones is the best performer today?\", we need to find the company with the highest percentage gain in the Dow Jones Industrial Average as of the query time \"03/05/2024, 23:18:31 PT\".\n\nFrom the given references, we can see that the query time is after the events mentioned in the references, but we can still find the information we need.\n\nDocument [7] mentions that as of March 5, 2024, the best Dow Jones stocks to watch in 2024 include Apple, IBM, Merck, Microsoft, and Visa.\n\nDocument [12] lists the components of the Dow Jones Industrial Average as of February 26, 2024, but it does not provide the current prices or percentage gains.\n\nDocument [13] provides the year-to-date (YTD) stock price performances of the Dow Jones stocks as of a certain date, but it does not specify the date. However, it mentions that the best performing companies serve in the tech space like Apple, Microsoft, and Salesforce.\n\nDocument [14] lists the top performing Dow Jones stocks as of a certain date, but it does not specify the date. However, it mentions that Microsoft Corp. (MSFT) has a price gain of 10.49% YTD.\n\nSince we are looking for the best performer as of March 5, 2024, we can use the information from Document [7] and Document [14] to make an educated guess. However, we cannot be certain without more up-to-date information.\n\nBased on the available information, we can see that Microsoft Corp. (MSFT) has a price gain of 10.49% YTD, which is one of the highest among the Dow Jones stocks mentioned in the references.</think>\nFinal Answer: \\boxed{Microsoft}"
Output: {"score": 0, "explanation": "The reasoning doesn't provide precise information to answer the question."}

Question: "how old is the current president of the us?"
Ground Truth: "81"
Reasoning: "<think> \n\nThe question is asking for the age of the current president of the US. The query time is 02/26/2024, 23:52:53 PT. \n\nLooking at the references, we have information about the ages of past and present presidents, but we need to find the current president's age. \n\nFrom the references, we can see that Joe Biden is the current president, born on November 20, 1942. \n\nWe can calculate his age by subtracting his birth year from the current year, which is 2024. \n\n2024 - 1942 = 82 \n\nHowever, we need to consider the month and day as well. Since the query time is in February 2024, and Joe Biden's birthday is on November 20, we need to subtract the age he would have been on February 26 from his actual age. \n\nJoe Biden's age on February 26, 2024, would be 81 years old. \n\nTherefore, the current president's age is 81.</think>\nFinal Answer: \\boxed{81}"
Output: {"score": 1, "explanation": "The reasoning provides precise information to answer the question."}
"""


import random
import re
import string

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str, option="answer_box"):

    if option == "answer_tags":
        answer_pattern = r"<answer>(.*?)</answer>"
    elif option == "answer_box":
        answer_pattern = r'\\boxed{(.*?)}'
    else:
        raise ValueError(f"Invalid option: {option}")
    
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None
        

def count_answer_tags(text):
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")

    return opening_tags, closing_tags


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_answer_tags(solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth["target"]):
            if open_count > 10 or close_count > 10:  # prevent output a lot of </answer>
                score = score / 4
                return score
            return score
        else:
            return format_score


def compute_score_subem(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth["target"]):
            return score
        else:
            return format_score



"""
>>> from openai import OpenAI
>>> client = OpenAI(
...     base_url="http://localhost:8000/v1",
...     api_key="token-abc123",
... )
>>> response = client.chat.completions.create(model="meta-llama/Llama-3.3-70B-Instruct", messages=[{"role":"user", "content":"tell me sth about meta"}])
>>> response.choices[0].message.content
"""


import os
from openai import OpenAI, APIConnectionError, RateLimitError

# Initialize OpenAI client
client = OpenAI(
    base_url=os.environ.get("OPENAI_API_BASE"),
    api_key=os.environ.get("OPENAI_API_KEY"),
)

import json
from datetime import datetime

def log_reward_detail(data):
    """Logs detailed reward information to a JSONL file. Only active if enabled."""
    if os.environ.get("TRUTHRL_ENABLE_TRAIN_LOGS") != "1":
        return
        
    log_name = os.environ.get("TRUTHRL_LOG_NAME", "training_default")
    log_dir = os.path.join("/home/kalashkala/TruthRL/outputs/reward_logs", log_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Use PID to avoid write conflicts in multi-process/Ray environments
    log_file = os.path.join(log_dir, f"reward_detail_{os.getpid()}.jsonl")
    
    data["timestamp"] = datetime.now().isoformat()
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(data) + "\n")
    except Exception as e:
        print(f"Failed to write reward log: {e}")

def get_system_message(type='outcome'):
    if type == 'outcome':
        return INSTRUCTIONS + "\n" + IN_CONTEXT_EXAMPLES
    elif type == 'process':
        return INSTRUCTIONS_REASONING + "\n" + IN_CONTEXT_EXAMPLES_REASONING
    else:
        raise ValueError(f"Invalid type: {type}")

def attempt_api_call(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="google/gemma-3-27b-it",
                messages=messages,
                temperature=0,
                top_p=0.9,
                max_tokens=512,
            )
            return response.choices[0].message.content
        except (APIConnectionError, RateLimitError) as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying... ({attempt + 2}/{max_retries})")
            else:
                print(f"All {max_retries} attempts failed. Last error: {e}")
                return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

def parse_response(response: str):
    matches = re.findall(r"{([^}]*)}", response)
    text = ""
    for match in matches:
        text = "{" + match + "}"
    try:
        score = -1
        # Pattern to match the score
        score_pattern = r'"score"\s*:\s*(\d+)'
        score_match = re.search(score_pattern, text)
        if score_match:
            score = int(score_match.group(1))
            if score != 0 and score != 1:
                raise Exception("bad score: " + response)
        else:
            return "Parse Err: Score not found", -1

        # Pattern to match the explanation
        explanation_pattern = r'"explanation"\s*:\s*"(.+)"'
        explanation_match = re.search(explanation_pattern, text)
        if explanation_match:
            explanation = explanation_match.group(1)
            return explanation, score
        else:
            return text, score
        
    except Exception as e:
        print(f"Parsing Error with resp: {response}")
        print(f"Error: {e}")
        return response, -1
    
def compute_score_llm_as_a_judge_binary_OOK(solution_str, ground_truth):

    gts = ground_truth['target']
    prediction = extract_solution(solution_str=solution_str)

    do_print = random.randint(1, 64) == 1

    if do_print:
        print("\n\n===================================")
        print(f"Question: {ground_truth['problem']}")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Out-of-knowledge: {ground_truth['out_of_knowledge']}")
        if prediction is not None:
            print(f"Extracted answer: {prediction}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")

    # otherwise, check if the prediction is correct via LLM-as-a-judge
    system_message = get_system_message()
    query = ground_truth['problem']
    reward = -1
    judge_explanation = None
    response_type = "hallucinated"
    
    if prediction is None:
        reward = -1
        response_type = "no_boxed"
    else:
        normalized_prediction = normalize_answer(prediction)
        if ground_truth['out_of_knowledge'] is True:
            if "i dont know" in normalized_prediction:
                reward = 1
                response_type = "correct"
            else:
                reward = -1
                response_type = "hallucinated"
        elif "i dont know" in normalized_prediction or "invalid question" in normalized_prediction:
            reward = -1
            response_type = "missing"
        else:
            # check if it matches any gt exactly
            found_match = False
            for gt in gts:
                if normalize_answer(gt) == normalized_prediction:
                    reward = 1
                    found_match = True
                    response_type = "correct"
                    break
            
            if not found_match:
                for gt in gts:
                    messages = [
                        {"role": "system", "content": system_message},
                        {
                            "role": "user",
                            "content": f"Question: {query}\n Ground truth: {gt}\n Prediction: {prediction}\n",
                        },
                    ]
                    llm_response = attempt_api_call(messages)
                    if llm_response:
                        judge_explanation, is_correct = parse_response(llm_response)
                        if is_correct == 1:
                            reward = 1
                            response_type = "correct"
                            if do_print:
                                print(f">>>>>> Reward: 1 (LLM-as-a-judge: {judge_explanation})")
                            break
                
    if reward == -1 and do_print:
        print(f">>>>>> Reward: -1 ({response_type})")

    log_reward_detail({
        "question": query,
        "golden_answers": gts,
        "out_of_knowledge": ground_truth['out_of_knowledge'],
        "prediction": prediction,
        "response_type": response_type,
        "full_response": solution_str,
        "reward": reward,
        "type": "binary_OOK",
        "judge_explanation": judge_explanation
    })

    return reward

    
def compute_score_llm_as_a_judge_binary(solution_str, ground_truth):

    gts = ground_truth['target']
    prediction = extract_solution(solution_str=solution_str)

    do_print = random.randint(1, 64) == 1

    if do_print:
        print("\n\n===================================")
        print(f"Question: {ground_truth['problem']}")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Out-of-knowledge: {ground_truth['out_of_knowledge']}")
        if prediction is not None:
            print(f"Extracted answer: {prediction}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")

    # otherwise, check if the prediction is correct via LLM-as-a-judge
    system_message = get_system_message()
    query = ground_truth['problem']
    reward = -1
    judge_explanation = None
    response_type = "hallucinated"
    
    if prediction is None:
        reward = -1
        response_type = "no_boxed"
    else:
        normalized_prediction = normalize_answer(prediction)
        if "i dont know" in normalized_prediction:
            reward = -1
            response_type = "missing"
        elif "invalid question" in normalized_prediction:
            reward = -1
            response_type = "missing"
        else:
            # exact match check
            found_match = False
            for gt in gts:
                if normalize_answer(gt) == normalized_prediction:
                    reward = 1
                    response_type = "correct"
                    found_match = True
                    break
            
            if not found_match:
                for gt in gts:
                    messages = [
                        {"role": "system", "content": system_message},
                        {
                            "role": "user",
                            "content": f"Question: {query}\n Ground truth: {gt}\n Prediction: {prediction}\n",
                        },
                    ]
                    llm_response = attempt_api_call(messages)
                    if llm_response:
                        judge_explanation, is_correct = parse_response(llm_response)
                        if is_correct == 1:
                            reward = 1
                            response_type = "correct"
                            if do_print:
                                print(f">>>>>> Reward: 1 (LLM-as-a-judge: {judge_explanation})")
                            break
                
    if reward == -1 and do_print:
        print(f">>>>>> Reward: -1 ({response_type})")

    # Log details for manual analysis
    log_reward_detail({
        "question": query,
        "golden_answers": gts,
        "out_of_knowledge": ground_truth['out_of_knowledge'],
        "prediction": prediction,
        "response_type": response_type,
        "full_response": solution_str,
        "reward": reward,
        "type": "binary",
        "judge_explanation": judge_explanation
    })

    return reward


def compute_process_score(solution_str, ground_truth):

    query = ground_truth['problem']
    gts = ground_truth['target']

    system_message = get_system_message(type='process')

    for gt in gts:
        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"Question: {query}\n Ground Truth: {gt}\n Reasoning: {solution_str}\n",
            },
        ]
        llm_response = attempt_api_call(messages)
        if llm_response:
            judge_response, is_correct = parse_response(llm_response)
            if is_correct == 1:
                return 1

    return 0

def compute_score_llm_as_a_judge_ternary(solution_str, ground_truth, process_reward=None):

    gts = ground_truth['target']
    prediction = extract_solution(solution_str=solution_str)

    do_print = random.randint(1, 64) == 1

    if do_print:
        print("\n\n===================================")
        print(f"Question: {ground_truth['problem']}")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Out-of-knowledge: {ground_truth['out_of_knowledge']}")
        if prediction is not None:
            print(f"Extracted answer: {prediction}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")

    reward = -1
    judge_explanation = None
    query = ground_truth['problem']
    response_type = "hallucinated"

    if prediction is None:
        reward = -1
        response_type = "no_boxed"
    else:
        normalized_prediction = normalize_answer(prediction)
        if "i dont know" in normalized_prediction:
            reward = 0
            response_type = "missing"
        elif "invalid question" in normalized_prediction:
            reward = -1
            response_type = "missing"
        else:
            # check for exact match
            found_match = False
            for gt in gts:
                if normalize_answer(gt) == normalized_prediction:
                    reward = 1
                    response_type = "correct"
                    found_match = True
                    break
            
            if not found_match:
                system_message = get_system_message()
                for gt in gts:
                    messages = [
                        {"role": "system", "content": system_message},
                        {
                            "role": "user",
                            "content": f"Question: {query}\n Ground truth: {gt}\n Prediction: {prediction}\n",
                        },
                    ]
                    llm_response = attempt_api_call(messages)
                    if llm_response:
                        judge_explanation, is_correct = parse_response(llm_response)
                        if is_correct == 1:
                            reward = 1
                            response_type = "correct"
                            break
            
    if do_print:
        print(f">>>>>> Reward: {reward} ({response_type})")
        if judge_explanation:
            print(f"   >>>>>> Judge: {judge_explanation}")

    # Log details for manual analysis
    log_reward_detail({
        "question": query,
        "golden_answers": gts,
        "out_of_knowledge": ground_truth['out_of_knowledge'],
        "prediction": prediction,
        "response_type": response_type,
        "full_response": solution_str,
        "reward": reward,
        "type": "ternary",
        "judge_explanation": judge_explanation
    })

    return reward


def compute_score_llm_as_a_judge_ternary_OOK(solution_str, ground_truth):

    gts = ground_truth['target']
    prediction = extract_solution(solution_str=solution_str)

    do_print = random.randint(1, 64) == 1

    if do_print:
        print("\n\n===================================")
        print(f"Question: {ground_truth['problem']}")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Out-of-knowledge: {ground_truth['out_of_knowledge']}")
        if prediction is not None:
            print(f"Extracted answer: {prediction}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")

    # otherwise, check if the prediction is correct via LLM-as-a-judge
    system_message = get_system_message()
    query = ground_truth['problem']
    reward = -1
    judge_explanation = None
    response_type = "hallucinated"
    
    if prediction is None:
        reward = -1
        response_type = "no_boxed"
    else:
        normalized_prediction = normalize_answer(prediction)
        if ground_truth['out_of_knowledge'] is True:
            if "i dont know" in normalized_prediction:
                reward = 1
                response_type = "correct"
            else:
                reward = -1
                response_type = "hallucinated"
        elif "i dont know" in normalized_prediction:
            reward = 0
            response_type = "missing"
        elif "invalid question" in normalized_prediction:
            reward = -1
            response_type = "missing"
        else:
            # exact match check
            found_match = False
            for gt in gts:
                if normalize_answer(gt) == normalized_prediction:
                    reward = 1
                    response_type = "correct"
                    found_match = True
                    break
            
            if not found_match:
                for gt in gts:
                    messages = [
                        {"role": "system", "content": system_message},
                        {
                            "role": "user",
                            "content": f"Question: {query}\n Ground truth: {gt}\n Prediction: {prediction}\n",
                        },
                    ]
                    llm_response = attempt_api_call(messages)
                    if llm_response:
                        judge_explanation, is_correct = parse_response(llm_response)
                        if is_correct == 1:
                            reward = 1
                            response_type = "correct"
                            if do_print:
                                print(f">>>>>> Reward: 1 (LLM-as-a-judge: {judge_explanation})")
                            break
                
    if do_print:
        print(f">>>>>> Reward: {reward} ({response_type})")

    # Log details for manual analysis
    log_reward_detail({
        "question": query,
        "golden_answers": gts,
        "out_of_knowledge": ground_truth['out_of_knowledge'],
        "prediction": prediction,
        "response_type": response_type,
        "full_response": solution_str,
        "reward": reward,
        "type": "ternary_OOK",
        "judge_explanation": judge_explanation
    })

    return reward


    
def compute_score_llm_as_a_judge_ternary_EM(solution_str, ground_truth):

    gts = ground_truth['target']
    prediction = extract_solution(solution_str=solution_str)

    do_print = random.randint(1, 64) == 1

    if do_print:
        print("\n\n===================================")
        print(f"Question: {ground_truth['problem']}")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Out-of-knowledge: {ground_truth['out_of_knowledge']}")
        if prediction is not None:
            print(f"Extracted answer: {prediction}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")

    # no answer box found, return -1
    reward = -1
    response_type = "hallucinated"
    if prediction is None:
        reward = -1
        response_type = "no_boxed"
    else:
        normalized_prediction = normalize_answer(prediction)

        # i dont know, return 0
        if "i dont know" in normalized_prediction:
            reward = 0
            response_type = "missing"
        # should not answer invalid question, return -1
        elif "invalid question" in normalized_prediction:
            reward = -1
            response_type = "missing"
        else:
            # check if the prediction exactly matches any of the ground truth
            for gt in gts:
                if normalize_answer(gt) == normalized_prediction:
                    reward = 1
                    response_type = "correct"
                    break
            
    if do_print:
        print(f">>>>>> Reward: {reward} ({response_type})")

    # Log details for manual analysis
    log_reward_detail({
        "question": ground_truth['problem'],
        "golden_answers": gts,
        "out_of_knowledge": ground_truth['out_of_knowledge'],
        "prediction": prediction,
        "response_type": response_type,
        "full_response": solution_str,
        "reward": reward,
        "type": "ternary_EM"
    })

    return reward