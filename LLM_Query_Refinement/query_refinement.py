import torch
import re
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import textwrap
import pandas as pd
from datetime import datetime
now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

model_name = "dnotitia/Smoothie-Qwen3-14B"

# GPU 메모리 분산 설정
max_memory = {0: "11GiB", 1: "11GiB"}

# Tokenizer 로드
# Load LLM model
model_name = "dnotitia/Smoothie-Qwen3-14B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
    #cache_dir="./model_cache"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

questions = pd.read_csv('/volume/yeeun_jeon/prompt/dataset/dataset_3labels.csv')

# 평가 리스트
ls = [0] * len(questions)

# 평가 결과 저장
answers = pd.DataFrame(columns=['text', 'label', 'ambiguity','example1','example2','time'])

for i in range(len(questions)):
    q = questions.iloc[i]
    query = q.iloc[0]
    start = time.time()

    # LLM을 사용한 애매함 판단 및 구체적 질문 생성
    system_prompt = f"""
    당신은 사용자 질문이 애매한지 판단하는 전문가입니다. 가능한 애매한 질문으로 분류하는 것을 최소화하세요.

    ### 가이드라인
    <판단 기준>
    다음 중 하나에 해당되면 애매한 질문으로 분류합니다.  
    - (유형 1) 업무 방법을 질문할 수도 있고, 업무 담당자를 찾을 수도 있어 사용자의 의도가 불명확한 질문
    - (유형 2) '이것', '그게 뭐야?' 등과 같이 문장 내에서 지시어가 지칭하는 것이 무엇인지 모르는 경우
    </판단 기준>

    <애매한 질문 예시>
    - "휴가 신청 어디서 해" → (유형 1) 휴가 신청 위치에 대한 질문인지, 휴가 신청 담당자를 찾는 질문인지 불명확함.
    - "저거 해결 좀 해줘" → (유형 2) '저거'가 지시하는 것이 무엇인지 모름.
    </애매한 질문 예시>

    ### 출력
    1. 사용자 질문이 애매한 경우
    - "A"에 'X'를 작성합니다. 
    - **사용자가 보험 회사 직원일 때, 재작성된 사용자 질문 2개를 만들어주세요.** 
    - B 안에 리스트 형식으로 작성합니다.
    - 재작성된 질문은 애매하지 않고 간결해야 합니다.
    - (유형 1)의 경우, "재작성된 질문1"은 '담당자'라는 단어를 포함하여 사용자 질문을 재작성해주세요. "재작성된 질문2"는 '방법'이라는 단어를 포함하여 사용자 질문을 재작성해주세요. 
    - (유형 2) 사용자 질문에 '이것', '그' 등의 지시어가 있는 경우, 지시어를 '노트북 구매', '보험료 계산'과 같은 구체적인 용어로 지시어를 대체하여 질문을 재작성합니다.
    2. 사용자 질문이 애매하지 않은 경우
    - 사용자 질문이 담당자, 관리자, 책임자, 팀, 부서, 업무를 아는 사람 등 문의처를 찾는 질문이면 "A"에 'P'를 작성합니다. 
    - 절차, 규정, 기준, 방법, 지원사항 등 문의처를 찾는 질문이 아니라면 "A"에 'M' 을 작성합니다. 
    - B에는 빈 리스트를 작성합니다. 

    모든 응답은 반드시 한국어로만 작성되어야 합니다. 
    생각 과정을 보여주지 마세요. 아래 형식만 반환하세요.:
    {{"A": "<'X' or 'P' or 'M'>", "B": "<["재작성된 질문1", "재작성된 질문2"]>"}}
    """
    user_prompt = f"""
    <사용자 질문>
    {query}
    </사용자 질문>"""

    messages = [
        {"role": "system", "content": textwrap.dedent(system_prompt)},
        {"role": "user", "content": textwrap.dedent(user_prompt)}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        chat_template_kwargs={"enable_thinking": False}
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=100,
        temperature=0.1,
        do_sample=False
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    llm_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(type(llm_response))
    print(llm_response)
    try:
        response = json.loads(re.search(r"(\{.*?})", llm_response, re.DOTALL).group(0))
        if response["A"]=='X' and len(response["B"])!=2:
            response={"A": "X", "B": [query+' 담당자', query+' 방법']}
    except:
        response={"A": "X", "B": [query+' 담당자', query+' 방법']}
    response_a=response["A"]
    questions_b=response["B"]
    end = time.time()

    print(f"번호: {i}, Query: {q.iloc[0]}, Label: {q.iloc[1]}, Ambiguity : {response_a}, Questions : {questions_b}, time:{end-start}")

    if response_a =="X" and  q.iloc[1]== '관련없음':
        ls[i] = 1
    elif response_a == 'M' and q.iloc[1]== '지원업무':
        ls[i] = 1
    elif response_a == 'P' and q.iloc[1]== '업무분장':
        ls[i] = 1

    if response_a =="X":
        answers.loc[len(answers)] = [query, q.iloc[1], response["A"], response["B"][0], response["B"][1], end-start]
    else: 
        answers.loc[len(answers)] = [query, q.iloc[1], response["A"], response["A"], response["A"], end-start]

print("Accuracy :", round(sum(ls)/len(ls), 3))

filename = f"test_{now_str}.csv"
answers.to_csv(filename)
