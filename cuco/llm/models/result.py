from typing import Dict, List, Optional


class QueryResult:
    def __init__(
        self,
        content: str,
        msg: str,
        system_msg: str,
        new_msg_history: List[Dict],
        model_name: str,
        kwargs: Dict,
        input_tokens: int,
        output_tokens: int,
        cost: float = 0.0,
        input_cost: float = 0.0,
        output_cost: float = 0.0,
        thought: str = "",
        model_posteriors: Optional[Dict[str, float]] = None,
    ):
        self.content = content
        self.msg = msg
        self.system_msg = system_msg
        self.new_msg_history = new_msg_history
        self.model_name = model_name
        self.kwargs = kwargs
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost = cost
        self.input_cost = input_cost
        self.output_cost = output_cost
        self.thought = thought
        self.model_posteriors = model_posteriors or {}

    def to_dict(self):
        return {
            "content": self.content,
            "msg": self.msg,
            "system_msg": self.system_msg,
            "new_msg_history": self.new_msg_history,
            "model_name": self.model_name,
            "kwargs": self.kwargs,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "thought": self.thought,
            "model_posteriors": self.model_posteriors,
        }
