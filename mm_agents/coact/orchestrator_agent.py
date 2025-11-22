import base64
import json
import os
import traceback
from google import genai
from typing import Any, Callable, Literal, Optional, Union
from desktop_env.desktop_env import DesktopEnv
from desktop_env.providers import ProviderConfig

from .autogen.llm_config import LLMConfig
from .autogen.agentchat.conversable_agent import ConversableAgent
from .autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent

from .cua_agent import run_openai_cua, run_claude_cua, run_claude_cua_bedrock, run_uitars_cua, run_opencua_cua
from .coding_agent import TerminalProxyAgent, CODER_SYSTEM_MESSAGE, CONVERSATION_REVIEW_PROMPT


class OrchestratorAgent(MultimodalConversableAgent):
    CALL_GUI_AGENT_TOOL = {
        "type": "function",
        "function": {
            "name": "call_gui_operator",
            "description": """- Can interact with the OS by GUI operations, including clicking, scrolling, typing, and using hotkeys.
- After the task completed, a summarizer will summarize the task completion process.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Describe the target followed by a detailed, step-by-step task instructions.",
                    },
                },
            },
        },
    }

    CALL_CODING_AGENT_TOOL = {
        "type": "function",
        "function": {
            "name": "call_programmer",
            "description": """- Can run Python or Bash code to interact with the system.
- Needs a target description with detailed task instructions.
- Can use any Python package you specify.
- After modifying a file, ALWAYS verify every change by yourself.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Describe the target followed by a detailed, step-by-step task instructions."
                    },
                },
            },
        },
    }

    CALL_API_SUMMARY_AGENT_TOOL = {
        "type": "function",
        "function": {
            "name": "call_api_summary_agent",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "[REQUIRED] A url of the API response."},
                },
            },
        },
    }

    def __init__(
        self,
        name: str,
        mode: str = "coact",
        system_message: Optional[str] = None,
        llm_config: Optional[Union[LLMConfig, dict[str, Any], Literal[False]]] = None,
        is_termination_msg: Optional[Callable[[dict[str, Any]], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        code_execution_config: Optional[Union[dict[str, Any], Literal[False]]] = False,
        description: Optional[str] = "",
        genai_client: Optional[genai.Client] = None,
        mimic_human_return: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            name,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            description=description,
            mimic_human_return=mimic_human_return,
            **kwargs,
        )

        if system_message is not None:
            self.update_system_message(system_message)

        if mode in ["hybrid", "coact_opensource_sft"]:
            self.update_tool_signature(self.CALL_CODING_AGENT_TOOL, is_remove=False)
            self.update_tool_signature(self.CALL_GUI_AGENT_TOOL, is_remove=False)
        elif mode == "coact_cua_only":
            self.update_tool_signature(self.CALL_GUI_AGENT_TOOL, is_remove=False)
        elif mode == "coact_coding_only":
            self.update_tool_signature(self.CALL_CODING_AGENT_TOOL, is_remove=False)



class OrchestratorUserProxyAgent(MultimodalConversableAgent):
    """(In preview) A proxy agent for the captain agent, that can execute code and provide feedback to the other agents."""

    DEFAULT_AUTO_REPLY = "Please continue the task. Note that the user's task is: {user_instruction}. If everything is done, please reply me only with 'TERMINATE'. If the task is impossible to solve, please reply me only with 'INFEASIBLE'."

    def __init__(
        self,
        name: str,
        is_termination_msg: Optional[Callable[[dict[str, Any]], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        code_execution_config: Optional[Union[dict[str, Any], Literal[False]]] = {},
        default_auto_reply: Optional[Union[str, dict[str, Any]]] = DEFAULT_AUTO_REPLY,
        llm_config: Optional[Union[LLMConfig, dict[str, Any], Literal[False]]] = False,
        system_message: Optional[Union[str, list]] = "",
        description: Optional[str] = None,

        # GUI Agent config
        provider_name: str = "docker",
        path_to_vm: str = None,
        observation_type: str = "screenshot",
        screen_width: int = 1920,
        screen_height: int = 1080,
        sleep_after_execution: float = 1.0,
        truncate_history_inputs: int = 51,
        cua_max_steps: int = 50,
        coding_max_steps: int = 30,
        history_save_dir: str = "",
        coding_model: str = "o4-mini",
        summarizer_model: str = "o3",
        llm_config_path: str = "",
        region: str = "",
        client_password: str = "",
        user_instruction: str = "",
        cua_client_config: dict = {},
        cua_model: str = "computer-use-preview",
        video_reflection: bool = False,
        genai_client: Optional[genai.Client] = None,
        mimic_human_return: bool = True,
        remote_ip_port: str = None,
    ):
        description = description if description is not None else ""
        super().__init__(
            name=name,
            system_message=system_message,
            mimic_human_return=mimic_human_return,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply.format(user_instruction=user_instruction),
            description=description,
        )
        self.register_function(
            function_map={
                "call_gui_operator": lambda **args: self._call_gui_operator(**args, screen_width=screen_width, screen_height=screen_height),
                "call_programmer": lambda **args: self._call_programmer(**args),
            }
        )
        self._code_execution_config = code_execution_config
        self.cua_config = {
            "max_steps": cua_max_steps,
            "sleep_after_execution": sleep_after_execution,
            "truncate_history_inputs": truncate_history_inputs,
        }
        self.cua_client_config = cua_client_config
        self.region = region
        self.client_password = client_password
        self.task_start_time = 0.0
        self.use_remote_env = True if remote_ip_port is not None else False
        if self.use_remote_env:
            provider_config = ProviderConfig(
                host=remote_ip_port.split(":")[0],
                port=int(remote_ip_port.split(":")[1]),
            )

            self.env = DesktopEnv(
                provider_name="docker_remote_fc_v1",
                provider_config=provider_config,
                action_space="pyautogui",
                os_type="Ubuntu",
                region=region,
                snapshot_name="init_state",
                screen_size=(screen_width, screen_height),
                headless=True,
                enable_proxy=True,
                require_a11y_tree=observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"],
            )
        else:
            self.env = DesktopEnv(
                path_to_vm=path_to_vm,
                action_space="pyautogui",
                provider_name=provider_name,
                os_type="Ubuntu",
                region=region,
                snapshot_name="init_state",
                screen_size=(screen_width, screen_height),
                headless=True,
                enable_proxy=True,
                require_a11y_tree=observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"],
            )

        self.history_save_dir = history_save_dir
        self.cua_call_count = 0
        self.coding_call_count = 0
        self.coding_max_steps = coding_max_steps
        self.coding_model_config = LLMConfig.from_json(path=llm_config_path).where(model=coding_model)
        self.summarizer_model_config = LLMConfig.from_json(path=llm_config_path).where(model=summarizer_model)
        self.cua_model = cua_model

        if video_reflection:
            self.genai_client = genai_client
            if self.genai_client is None:
                self.genai_client = genai.Client(
                    vertexai=True, project='salesforce-research-internal', location='us-west1'
                )
        else:
            self.genai_client = None

    def set_task_start_time(self, task_start_time: float):
        self.task_start_time = task_start_time

    def reset(self, task_config: dict[str, Any], sleep_time: int = 20):
        if self.use_remote_env:
            obs = self.env.reset_docker_remote_fc_v1(task_config=task_config, sleep_time=sleep_time)
        else:
            obs = self.env.reset(task_config=task_config, sleep_time=sleep_time)
        print(f"VM started on localhost:{self.env.vnc_port}", flush=True)
        print(f"Screen size: {self.env.controller.get_vm_screen_size()}", flush=True)
        return obs

    def _call_gui_operator(self, task: str, screen_width: int = 1920, screen_height: int = 1080) -> str:
        """Run a GUI agent to solve the task."""
        cua_path = os.path.join(self.history_save_dir, f'cua_output_{self.cua_call_count}')
        screen_size = self.env.controller.get_vm_screen_size()
        width = screen_size["width"]
        height = screen_size["height"]
        if not os.path.exists(cua_path):
            os.makedirs(cua_path)
        with open(os.path.join(cua_path, "subtask.txt"), "w") as f:
            f.write(task)
        
        cua_function = None
        if self.cua_model == "computer-use-preview":
            cua_function = run_openai_cua
        elif 'claude' in self.cua_model and 'anthropic' not in self.cua_model:
            cua_function = run_claude_cua
        elif 'anthropic' in self.cua_model:
            cua_function = run_claude_cua_bedrock
        elif 'UI-TARS-1.5' in self.cua_model:
            cua_function = run_uitars_cua
        elif 'OpenCUA' in self.cua_model:
            cua_function = run_opencua_cua

        try:
            history_inputs, result, cost = cua_function(self.env,
                                                        task,
                                                        save_path=cua_path,
                                                        screen_width=width,
                                                        screen_height=height,
                                                        client_password=self.client_password,
                                                        cua_client_config=self.cua_client_config,
                                                        cua_model=self.cua_model,
                                                        **self.cua_config)
            screenshot = self.env.controller.get_screenshot()
            with open(os.path.join(cua_path, "history_inputs.json"), "w") as f:
                json.dump(history_inputs, f)
            with open(os.path.join(cua_path, "result.txt"), "w") as f:
                f.write(result)
            with open(os.path.join(cua_path, "cost.txt"), "w") as f:
                f.write(str(cost))

            self.cua_call_count += 1

        except Exception:
            return f"# Call GUI operator error: {traceback.format_exc()}"

        if "TERMINATE" in result:
            result = result.replace("TERMINATE", "").strip()
        else:
            result = f"I've reach the max steps and have to stop. Please check the screenshot and see what to do next."
        return f"# Response from the GUI operator: \n{result}\n<img data:image/png;base64,{base64.b64encode(screenshot).decode('utf-8')}>"
    
    def _call_programmer(self, task: str) -> str:
        """Run a coding agent to solve the task."""
        default_auto_reply = "I'm a code interpreter and I can only execute your code or end the conversation. Did you check your result carefully and make sure the things out of the user's instruction are not changed? If you think the task completed, please reply me only with 'TERMINATE'."
        try:
            screenshot = self.env.controller.get_screenshot()
            coding_agent = MultimodalConversableAgent(
                name="coding_agent",
                llm_config=self.coding_model_config,
                system_message=CODER_SYSTEM_MESSAGE.format(CLIENT_PASSWORD=self.client_password),
            )
            code_interpreter = TerminalProxyAgent(
                name="code_interpreter",
                human_input_mode="NEVER",
                code_execution_config={
                    "use_docker": False,
                    "timeout": 300,
                    "last_n_messages": 1,
                },
                max_consecutive_auto_reply = None,
                default_auto_reply = default_auto_reply,
                description = None,
                is_termination_msg=lambda x: x.get("content", "") and x.get("content", "")[0]["text"].lower() == "terminate",
                env=self.env,
            )
            coding_agent.update_system_message(CODER_SYSTEM_MESSAGE.format(CLIENT_PASSWORD=self.client_password))

            code_interpreter.initiate_chat(
                recipient=coding_agent,
                message=f"# Task\n{task}\n\n<img data:image/png;base64,{base64.b64encode(screenshot).decode('utf-8')}>",
                max_turns=self.coding_max_steps,
            )
        
            chat_history = []
            key = list(code_interpreter.chat_messages.keys())[0]
            chat_messages = code_interpreter.chat_messages[key]
            for item in chat_messages:
                for content in item['content']:
                    if content['type'] == 'image_url':
                        content['image_url']['url'] = '<image>'
                chat_history.append(item)
            
            if not os.path.exists(os.path.join(self.history_save_dir, f'coding_output_{self.coding_call_count}')):
                os.makedirs(os.path.join(self.history_save_dir, f'coding_output_{self.coding_call_count}'))
                
            with open(os.path.join(self.history_save_dir, f'coding_output_{self.coding_call_count}', "chat_history.json"), "w") as f:
                json.dump(chat_history, f)
            with open(os.path.join(self.history_save_dir, f'coding_output_{self.coding_call_count}', f'coding_agent_system_prompt.txt'), "w") as f:
                f.write(CODER_SYSTEM_MESSAGE)
            with open(os.path.join(self.history_save_dir, f'coding_output_{self.coding_call_count}', f'subtask.txt'), "w") as f:
                f.write(task)
            self.coding_call_count += 1

            # Review the group chat history
            summarizer = ConversableAgent(
                name="summarizer",
                llm_config=self.summarizer_model_config,
                system_message="",
            )
            summarized_history = summarizer.generate_oai_reply(
                messages=[
                    {
                        "role": "user",
                        "content": CONVERSATION_REVIEW_PROMPT.format(task=task, chat_history=chat_history),
                    }
                ]
            )[1]
        except Exception:
            return f"# Call programmer error: {traceback.format_exc()}"

        return f"# Response from the programmer: {summarized_history}"
