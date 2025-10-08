import openai
from rich import print as rprint
import time
from typing import Union
from .utils import convert_messages_to_prompt, retry_with_exponential_backoff

# Refer to https://platform.openai.com/docs/models/overview
TOKEN_LIMIT_TABLE = {
    "text-davinci-003": 4080,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
    # Add lenient defaults for newer model names
    "gpt-4.1-nano": 4096,
    "gpt-4o-mini": 16384,
}


class Module(object):
    """
    This module is responsible for communicating with GPTs.
    """
    def __init__(self, 
                 role_messages, 
                 model="gpt-3.5-turbo-0301",
                 retrival_method="recent_k",
                 K=3):
        '''
        args:  
        use_similarity: 
        dia_num: the num of dia use need retrival from dialog history
        '''

        self.model = model
        self.retrival_method = retrival_method
        self.K = K

        self.chat_model = True if "gpt" in self.model else False
        self.instruction_head_list = role_messages
        self.dialog_history_list = []
        self.current_user_message = None
        self.cache_list = None

    def add_msgs_to_instruction_head(self, messages: Union[list, dict]):
        if isinstance(messages, list):
            self.instruction_head_list += messages
        elif isinstance(messages, dict):
            self.instruction_head_list += [messages]

    def add_msg_to_dialog_history(self, message: dict):
        self.dialog_history_list.append(message)
    
    def get_cache(self)->list:
        if self.retrival_method == "recent_k":
            if self.K > 0:
                return self.dialog_history_list[-self.K:]
            else: 
                return []
        else:
            return None 
           
    @property
    def query_messages(self)->list:
        return self.instruction_head_list + self.cache_list + [self.current_user_message]
    
    @retry_with_exponential_backoff
    def query(self, key, stop=None, temperature=0.0, debug_mode = 'Y', trace = True):
        """Unified query supporting legacy ChatCompletion and new Responses API."""
        # Prefer new SDK client if available
        client = None
        try:
            from openai import OpenAI  # new SDK
            client = OpenAI(api_key=key)
        except Exception:
            pass

        # Build prompt/messages
        rec = self.K  
        if trace == True: 
            self.K = 0 
        self.cache_list = self.get_cache()
        messages = self.query_messages
        if trace == False: 
            messages[len(messages) - 1]['content'] += " Based on the failure explanation and scene description, analyze and plan again." 
        self.K = rec 

        get_response = False
        retry_count = 0
        response_text = None
        while not get_response:  
            if retry_count > 3:
                rprint("[red][ERROR][/red]: Query GPT failed for over 3 times!")
                return {}
            try:  
                if self.model in ['text-davinci-003']:
                    # Legacy completions (rarely used here)
                    prompt = convert_messages_to_prompt(messages) 
                    if client is not None and hasattr(client, 'completions'):
                        resp = client.completions.create(model=self.model, prompt=prompt, temperature=temperature, max_tokens=256, stop=stop)
                        response_text = resp.choices[0].text
                    else:
                        openai.api_key = key
                        resp = openai.Completion.create(model=self.model, prompt=prompt, temperature=temperature, max_tokens=256, stop=stop)
                        response_text = resp["choices"][0]["text"]
                else:
                    # Chat-style models: try new Responses API first for 4.1/4o families
                    if client is not None and ("4.1" in self.model or "4o" in self.model):
                        # Convert chat messages to a single prompt text for responses
                        prompt = convert_messages_to_prompt(messages)
                        resp = client.responses.create(
                            model=self.model,
                            input=prompt,
                            temperature=temperature,
                        )
                        # Extract text
                        if resp.output and len(resp.output) > 0 and hasattr(resp.output[0], 'content') and len(resp.output[0].content) > 0:
                            response_text = resp.output[0].content[0].text
                        else:
                            # Fallback generic
                            response_text = getattr(resp, 'output_text', None) or str(resp)
                    else:
                        # Legacy ChatCompletion path
                        if client is not None and hasattr(client, 'chat'):
                            resp = client.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                temperature=temperature,
                                max_tokens=256,
                                stop=stop,
                            )
                            response_text = resp.choices[0].message.content
                        else:
                            openai.api_key = key
                            resp = openai.ChatCompletion.create(
                                model=self.model,
                                messages=messages,
                                temperature=temperature,
                                max_tokens=256,
                                stop=stop,
                            )
                            response_text = resp["choices"][0]["message"]["content"]

                time.sleep(0.1)
                get_response = True

            except Exception as e:
                retry_count += 1
                rprint("[red][OPENAI ERROR][/red]:", e)
                time.sleep(2)
        return response_text

    def restrict_dialogue(self):
        """
        The limit on token length for gpt-3.5-turbo-0301 is 4096.
        If token length exceeds the limit, we will remove the oldest messages.
        """
        limit = TOKEN_LIMIT_TABLE.get(self.model, 4096)
        print(f'Current token: {self.prompt_token_length}')
        while self.prompt_token_length >= limit:
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            print(f'Update token: {self.prompt_token_length}')
        
    def reset(self):
        self.dialog_history_list = []

