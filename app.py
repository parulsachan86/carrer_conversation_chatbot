import os
import json
import requests
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from pydantic import BaseModel
from typing import List

# --- INITIALIZATION ---
load_dotenv(override=True)

class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str

# --- CONFIGURATION & CONSTANTS ---
MODELNAME = "gemini-3.1-flash-lite-preview"
PUSHOVER_URL = "https://api.pushover.net/1/messages.json"
RESUME_PATH = "me/Parul_Sachan_Resume.pdf"
SUMMARY_PATH = "me/summary.txt"
NAME = "Parul"

client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# --- CORE CHATBOT CLASS ---
class CareerChatbot:
    def __init__(self):
        self.resume = self._load_resume()
        self.summary = self._load_summary()
        self.system_prompt = self._build_main_prompt()
        self.evaluator_system_prompt = self._build_evaluator_prompt()
        self.tools = self._get_tool_schemas()

    # --- DATA LOADING ---
    def _load_resume(self):
        reader = PdfReader(RESUME_PATH)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content: text += content
        return text

    def _load_summary(self):
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            return f.read()

    # --- PROMPT FACTORIES ---
    def _build_main_prompt(self):
        prompt = (
            f"You are acting as {NAME}. You are answering questions on {NAME}'s website, "
            f"particularly questions related to {NAME}'s career, background, skills and experience. "
            f"Your responsibility is to represent {NAME} for interactions on the website as faithfully as possible. "
            f"You are given a summary of {NAME}'s background and Resume which you can use to answer questions. "
            "Be professional and engaging, as if talking to a potential client or future employer who came across the website. "
            "If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. "
            "If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "
        )
        prompt += f"\n\n## Summary:\n{self.summary}\n\n## Resume:\n{self.resume}\n\n"
        prompt += f"With this context, please chat with the user, always staying in character as {NAME}."
        return prompt

    def _build_evaluator_prompt(self):
        prompt = (
            "You are an evaluator that decides whether a response to a question is acceptable. "
            "You are provided with a conversation between a User and an Agent. Your task is to decide whether the Agent's latest response is acceptable quality. "
            f"The Agent is playing the role of {NAME} and is representing {NAME} on their website. "
            "The Agent has been instructed to be professional and engaging, as if talking to a potential client or future employer who came across the website. "
            f"The Agent has been provided with context on {NAME} in the form of their summary and Resume details. Here's the information:"
        )
        prompt += f"\n\n## Summary:\n{self.summary}\n\n## Resume:\n{self.resume}\n\n"
        prompt += "With this context, please evaluate the latest response, replying with whether the response is acceptable and your feedback."
        return prompt

    # --- TOOLS & SCHEMAS ---
    def _get_tool_schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "record_user_details",
                    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "email": {"type": "string", "description": "The email address of this user"},
                            "name": {"type": "string", "description": "The user's name, if they provided it"},
                            "notes": {"type": "string", "description": "Any additional information worth recording"},
                        },
                        "required": ["email"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "record_unknown_question",
                    "description": "Always use this tool to record any question that couldn't be answered",
                    "parameters": {
                        "type": "object",
                        "properties": {"question": {"type": "string", "description": "The question that couldn't be answered"}},
                        "required": ["question"],
                    },
                },
            },
        ]

    # --- ACTIONS ---
    def push(self, message):
        print(f"Push: {message}")
        user, token = os.getenv("PUSHOVER_USER"), os.getenv("PUSHOVER_TOKEN")
        if user and token:
            payload = {"user": user, "token": token, "message": message}
            requests.post(PUSHOVER_URL, data=payload)

    def record_user_details(self, email, name="Name not provided", notes="not provided"):
        self.push(f"Recording interest from {name} with email {email} and notes {notes}")
        return {"recorded": "ok"}

    def record_unknown_question(self, question):
        self.push(f"Recording {question} asked that I couldn't answer")
        return {"recorded": "ok"}

    # --- LOGIC HANDLERS ---
    def handle_tool_calls(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            if tool_name == "record_user_details":
                result = self.record_user_details(**arguments)
            elif tool_name == "record_unknown_question":
                result = self.record_unknown_question(**arguments)
            
            results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
        return results

    def evaluate(self, reply, message, history) -> Evaluation:
        user_prompt = (
            f"Here's the conversation: \n\n{history}\n\n"
            f"Latest User message: \n\n{message}\n\n"
            f"Latest Agent response: \n\n{reply}\n\n"
            "Please evaluate the response, replying with whether it is acceptable and your feedback."
        )
        messages = [{"role": "system", "content": self.evaluator_system_prompt},
                    {"role": "user", "content": user_prompt}]
        
        response = client.beta.chat.completions.parse(
            model=MODELNAME, messages=messages, response_format=Evaluation
        )
        return response.choices[0].message.parsed

    def rerun(self, reply, message, history, feedback):
        updated_prompt = self.system_prompt + f"\n\n## Previous answer rejected\n## Your attempted answer:\n{reply}\n\n## Reason for rejection:\n{feedback}\n\n"
        messages = [{"role": "system", "content": updated_prompt}] + history + [{"role": "user", "content": message}]
        response = client.chat.completions.create(model=MODELNAME, messages=messages)
        return response.choices[0].message.content

    def chat_logic(self, message, history):
        # messages = [{"role": "system", "content": self.system_prompt}] + history + [{"role": "user", "content": message}]
        
        # while True:
        #     response = client.chat.completions.create(model=MODELNAME, messages=messages, tools=self.tools)
        #     choice = response.choices[0]
            
        #     if choice.finish_reason == "tool_calls":
        #         msg = choice.message
        #         messages.append(msg)
        #         messages.extend(self.handle_tool_calls(msg.tool_calls))
        #         # --- THE FIX ---
        #         # We must 'continue' the loop so the LLM can see the tool results 
        #         # and generate a human-readable text reply to return to the UI.
        #         continue
        #     else:
        #         reply = choice.message.content
        #         # To enable the evaluation retry logic, uncomment the following:
        #         # eval_res = self.evaluate(reply, message, history)
        #         # if not eval_res.is_acceptable:
        #         #     return self.rerun(reply, message, history, eval_res.feedback)
        #         return reply

        messages = [{"role": "system", "content": self.system_prompt}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            # response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
            response = client.chat.completions.create(model=MODELNAME, messages=messages, tools=self.tools)
            print("Messages:", messages)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_calls(tool_calls)
                messages.append(message)
                messages.extend(results)
                continue
            else:
                done = True
        return response.choices[0].message.content

# --- MAIN ---
def main():
    bot = CareerChatbot()
    port = int(os.environ.get("PORT", 10000))
    gr.ChatInterface(bot.chat_logic).launch(
        server_name="0.0.0.0", 
        server_port=port
    )

if __name__ == "__main__":
    main()