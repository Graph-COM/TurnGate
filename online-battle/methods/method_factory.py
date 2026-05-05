from typing import Dict, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from methods.abstract_method import AbstractJailbreakMethod
from methods.proposed.cka_agent import CKAAgentMethod
from methods.proposed.maj_agent import MultiAgentJailbreakMethod


def create_method(method_name: str, config=None, model=None):
    method_name = method_name.lower()
    if method_name == "cka-agent":
        return CKAAgentMethod(name=method_name, config=config, model=model)
    elif method_name == "multi-agent-jailbreak":
        return MultiAgentJailbreakMethod(name=method_name, config=config, model=model)
    raise ValueError(
        f"Unsupported method: {method_name}. "
        f"Supported methods: {supported_methods()}"
    )


def supported_methods():
    return ["cka-agent", "multi-agent-jailbreak"]
