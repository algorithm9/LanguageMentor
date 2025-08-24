# src/agents/scenario_agent.py

import random

from langchain_core.messages import AIMessage, HumanMessage  # 导入消息类

from .session_history import get_session_history  # 导入会话历史相关方法
from .agent_base import AgentBase
from utils.logger import LOG


class ScenarioAgent(AgentBase):
    """
    场景代理类，负责处理特定场景下的对话。
    """
    def __init__(self, scenario_name, session_id=None):
        prompt_file = f"prompts/{scenario_name}_prompt.txt"
        intro_file = f"content/intro/{scenario_name}.json"
        super().__init__(
            name=scenario_name,
            prompt_file=prompt_file,
            intro_file=intro_file,
            session_id=session_id
        )

    def _convert_history_to_gradio_format(self, history):
        """
        将 LangChain 的 history 对象转换为 Gradio Chatbot 兼容的格式。
        """
        gradio_history = []
        for msg in history.messages:
            if isinstance(msg, HumanMessage):
                gradio_history.append({'role': 'user', 'content': msg.content})
            elif isinstance(msg, AIMessage):
                gradio_history.append({'role': 'assistant', 'content': msg.content})
            # SystemMessage 通常不在聊天界面上显示，所以我们在这里忽略它
        return gradio_history

    def start_new_session(self, session_id=None):
        """
        开始一个新的场景会话，并返回格式化后的完整聊天记录以更新UI。

        参数:
            session_id (str, optional): 会话的唯一标识符

        返回:
            list[dict]: Gradio Chatbot 格式的完整聊天记录
        """
        if session_id is None:
            session_id = self.session_id

        history = get_session_history(session_id)

        # 如果是新会话，添加初始消息
        if not history.messages:
            initial_ai_message = random.choice(self.intro_messages)
            history.add_message(AIMessage(content=initial_ai_message))

        LOG.debug(f"[history][{session_id}]:{history.messages}")

        # 无论如何，都返回格式化后的完整历史记录
        return self._convert_history_to_gradio_format(history)
