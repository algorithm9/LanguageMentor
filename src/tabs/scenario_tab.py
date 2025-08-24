# tabs/scenario_tab.py

import gradio as gr
from agents.scenario_agent import ScenarioAgent
from utils.logger import LOG

# 初始化场景代理
agents = {
    "job_interview": ScenarioAgent("job_interview"),
    "hotel_checkin": ScenarioAgent("hotel_checkin"),
    "salary_negotiation": ScenarioAgent("salary_negotiation"),
    "renting_apartment": ScenarioAgent("renting_apartment"),
    "requesting_leave": ScenarioAgent("requesting_leave"),
}

def get_page_desc(scenario):
    try:
        with open(f"content/page/{scenario}.md", "r", encoding="utf-8") as file:
            scenario_intro = file.read().strip()
        return scenario_intro
    except FileNotFoundError:
        LOG.error(f"场景介绍文件 content/page/{scenario}.md 未找到！")
        return "场景介绍文件未找到。"

# 获取场景介绍并启动新会话的函数
def start_new_scenario_chatbot(scenario):
    if not scenario:
        return gr.update(value=None, interactive=False), gr.update(value=None)
    initial_ai_message = agents[scenario].start_new_session()  # 启动新会话并获取初始AI消息

    return gr.Chatbot(
        value=[(None, initial_ai_message)],  # 设置聊天机器人的初始消息
        height=600,  # 聊天窗口高度
        type="messages"
    )

# 场景代理处理函数，根据选择的场景调用相应的代理
def handle_scenario(user_input, chat_history, scenario):
    bot_message = agents[scenario].chat_with_history(user_input)  # 获取场景代理的回复
    LOG.info(f"[ChatBot]: {bot_message}")  # 记录场景代理的回复
    return bot_message  # 返回场景代理的回复

def create_scenario_tab():
    with gr.Tab("场景"):  # 场景标签
        gr.Markdown("## 选择一个场景完成目标和挑战")  # 场景选择说明

        # 创建单选框组件
        scenario_radio = gr.Radio(
            choices=[
                ("求职面试", "job_interview"),
                ("酒店入住", "hotel_checkin"),
                ("薪酬谈判", "salary_negotiation"),
                ("租房", "renting_apartment"),
                ("单位请假", "requesting_leave")
            ],
            label="场景"  # 单选框标签
        )

        scenario_intro = gr.Markdown()  # 场景介绍文本组件
        scenario_chatbot = gr.Chatbot(
            placeholder="<strong>你的英语私教 DjangoPeng</strong><br><br>选择场景后开始对话吧！",  # 聊天机器人的占位符
            height=600,  # 聊天窗口高度
            type="messages"
        )

        # 更新场景介绍并在场景变化时启动新会话
        scenario_radio.change(
            fn=lambda s: (get_page_desc(s), start_new_scenario_chatbot(s)),  # 更新场景介绍和聊天机器人
            inputs=scenario_radio,  # 输入为选择的场景
            outputs=[scenario_intro, scenario_chatbot],  # 输出为场景介绍和聊天机器人组件
        )

        # 场景聊天界面
        gr.ChatInterface(
            fn=handle_scenario,  # 处理场景聊天的函数
            chatbot=scenario_chatbot,  # 聊天机器人组件
            additional_inputs=scenario_radio,  # 额外输入为场景选择
            submit_btn="发送",  # 发送按钮文本
        )