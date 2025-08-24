import json
import os
from abc import ABC, abstractmethod

from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

from .session_history import get_session_history
from utils.logger import LOG
from utils.config_loader import get_active_model_config

class AgentBase(ABC):
    """
    Abstract base class providing common functionality for agents.
    """
    def __init__(self, name, prompt_file, intro_file=None, session_id=None):
        self.name = name
        self.prompt_file = prompt_file
        self.intro_file = intro_file
        self.session_id = session_id if session_id else self.name
        self.prompt = self.load_prompt()
        self.intro_messages = self.load_intro() if self.intro_file else []
        self.model_config = get_active_model_config()
        self.create_chatbot()

    def load_prompt(self):
        """Loads the system prompt from a file."""
        try:
            with open(self.prompt_file, "r", encoding="utf-8") as file:
                return file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_file}!")

    def load_intro(self):
        """Loads introductory messages from a JSON file."""
        try:
            with open(self.intro_file, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Intro file not found: {self.intro_file}!")
        except json.JSONDecodeError:
            raise ValueError(f"Intro file {self.intro_file} contains invalid JSON!")

    def _initialize_model(self):
        """Initializes the language model based on the configuration."""
        provider = self.model_config.get("provider")
        model_name = self.model_config.get("model_name")
        temperature = self.model_config.get("temperature", 0.8)
        max_tokens = self.model_config.get("max_tokens", 8192)

        LOG.info(f"Initializing model with provider: {provider}")

        if provider == "ollama":
            # NOTE: For newer versions, ChatOllama is in langchain_community
            return ChatOllama(
                model=model_name,
                temperature=temperature,
            )
        elif provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable not set for OpenAI provider")
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif provider == "deepseek":
            if not os.getenv("DEEPSEEK_API_KEY"):
                raise ValueError("DEEPSEEK_API_KEY environment variable not set for DeepSeek provider")
            return ChatDeepSeek(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            raise ValueError(f"Unsupported model provider: {provider}")

    def create_chatbot(self):
        """Initializes the chatbot with a system prompt and message history."""
        system_prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

        llm = self._initialize_model()
        self.chatbot = system_prompt | llm
        self.chatbot_with_history = RunnableWithMessageHistory(self.chatbot, get_session_history)

    def chat_with_history(self, user_input, session_id=None):
        """
        Processes user input and generates a response with chat history.
        """
        if session_id is None:
            session_id = self.session_id

        response = self.chatbot_with_history.invoke(
            [HumanMessage(content=user_input)],
            {"configurable": {"session_id": session_id}},
        )

        LOG.debug(f"[ChatBot][{self.name}] {response.content}")
        return response.content