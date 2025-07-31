import torch
import torch.nn.functional as F
from AHT_human_intervention.language.language_conditioned_policy import HuggingFaceLangConditionedPolicy, get_hf_tokenizer, MAX_LEN, build_env_prompt
from AHT_human_intervention.intervention_LLM_module import process_command
import json

class SharedLangAgent:
    def __init__(self, mdp, agent_idx=0, model=None, model_path=None, device="cpu"): 
        self.device = torch.device(device)
        self.mdp = mdp
        self.agent_idx = agent_idx
        self.current_command = ""
        self.heuristic = None
        self.tokenizer = get_hf_tokenizer()
        # Model setup
        if model is not None:
            self.model = model
        else:
            # Infer state_dim and num_actions from MDP
            example_state = torch.FloatTensor(
                self.mdp.lossless_state_encoding(self.mdp.get_standard_start_state())[0].flatten()
            )
            state_dim = example_state.numel()
            num_actions = len(self.mdp.action_idx_to_name)
            self.model = HuggingFaceLangConditionedPolicy(
                state_dim=state_dim,
                num_actions=num_actions,
                text_dim=768,
                hidden_dim=256,
                freeze_bert=True
            )
            if model_path is not None:
                # NOTE: model_path should be a state_dict for the policy head only, not the full transformer
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.policy_head.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()

    def set_command(self, command):
        self.current_command = command

    def act(self, state):
        cmd = self.current_command.strip().lower()
        # LLM-based heuristic switching
        llm_result = process_command(cmd)
        new_heuristic = llm_result.get("ego_agent_new_heuristic", None)
        if new_heuristic is not None:
            if new_heuristic != self.heuristic:
                print(f"[LLM] Changing heuristic to: {new_heuristic}")
                self.heuristic = new_heuristic
        if self.heuristic:
            print(f"[Agent] Current heuristic: {self.heuristic}")
        # Tokenize prompt for HuggingFace model
        prompt = build_env_prompt(state) + "\nHuman: " + self.current_command
        tokens = self.tokenizer(prompt, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)
        arr = self.mdp.lossless_state_encoding(state)[self.agent_idx].flatten()
        s = torch.FloatTensor(arr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(s, input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)
            act_idx = int(torch.argmax(probs, dim=-1)[0])
        return act_idx 