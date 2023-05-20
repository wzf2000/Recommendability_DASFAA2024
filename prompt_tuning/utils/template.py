from openprompt.prompts import ManualTemplate
from transformers import PreTrainedTokenizer
from loguru import logger

template_texts = {
    'zh': [
        '假设你是一个智能的对话推荐系统，你可以智能地在对话过程中判定是否需要给出推荐还是继续常规对话，以下是你和一个用户的对话历史：{"placeholder":"text_a"} 你会选择？选项：0. 不进行推荐；1. 进行推荐。回答：{"mask"}',
        '''
        任务描述：
        推荐系统在推荐前应当在推荐前判断是否是一个合适的推荐时机，本任务要求模型根据用户与系统的对话历史判断是否是一个合适的推荐时机。
        输入：
        ```
        {"placeholder":"text_a"}
        ```
        选项：
        0. 不是一个合适的推荐时机；1. 是一个合适的推荐时机。
        回答：
        {"mask"}
        ''',
        '假设你是一个智能的电商客服，你可以智能地在与顾客对话的过程中判断顾客的需求，并在需要推荐的是否给出推荐，以下是你和一个顾客的对话历史：{"placeholder":"text_a"} 你会选择？选项：0. 不进行推荐；1. 进行推荐。回答：{"mask"}',
        '''
        任务描述：
        电商客服在与顾客对话过程中，应具有判断顾客需求的能力，本任务要求模型根据顾客与客服的对话历史判断是否是一个合适的推荐时机。
        输入：
        ```
        {"placeholder":"text_a"}
        ```
        选项：
        0. 不是一个合适的推荐时机；1. 是一个合适的推荐时机。
        回答：
        {"mask"}
        ''',
    ],
    'en': [
        'Assuming that you are an intelligent dialogue recommendation system. You can intelligently determine whether to give a recommendation or continue a regular dialogue during the dialogue. The following is the dialogue history between you and a user: {"placeholder":"text_a"} You will choose? Options: 0: no recommendation; 1: recommendation. Answer: {"mask"}',
        '''
        Task description:
        The recommender system should determine whether it is an appropriate recommendation time before the recommendation. This task requires the model to determine whether it is an appropriate recommendation time based on the dialogue history between the user and the system.
        Input:
        ```
        {"placeholder":"text_a"}
        ```
        Options:
        0: It is not an appropriate recommendation time; 1: It is an appropriate recommendation time.
        Answer:
        {"mask"}
        ''',
        'Assuming that you are an intelligent e-commerce customer service, you can intelligently determine the needs of customers in the process of communicating with customers, and give recommendations when needed. The following is the dialogue history between you and a customer: {"placeholder":"text_a"} You will choose? Options: 0: no recommendation; 1: recommendation. Answer: {"mask"}',
        '''
        Task description:
        E-commerce customer service should have the ability to judge customer needs during the process of communicating with customers. This task requires the model to determine whether it is an appropriate recommendation time based on the dialogue history between the customer and the customer service.
        Input:
        ```
        {"placeholder":"text_a"}
        ```
        Options:
        0: It is not an appropriate recommendation time; 1: It is an appropriate recommendation time.
        Answer:
        {"mask"}
        ''',
    ]
}

assert len(template_texts['zh']) == len(template_texts['en'])

template_num = len(template_texts['zh'])

def get_template(tokenizer: PreTrainedTokenizer, language: str, template_id: int, model_name: str) -> ManualTemplate:
    template_text = template_texts[language][template_id]
    if language == 'en' and model_name == 't5':
        # T5 has a special token for 0
        # replace the '0' in the template with 'zero' and '1' with 'one'
        template_text = template_text.replace('0', 'zero')
        template_text = template_text.replace('1', 'one')
    logger.info(f'[Using Template {template_id}: {template_text}]')
    return ManualTemplate(
        text=template_text,
        tokenizer=tokenizer
    )