import base64
import json
import logging
import os
import time
from typing import Any, Dict, List, Tuple

import openai
from desktop_env.desktop_env import DesktopEnv
from openai import OpenAI  # pip install --upgrade openai>=1.30
from ..utils import env_step

logger = logging.getLogger("desktopenv")

GPT4O_INPUT_PRICE_PER_1M_TOKENS = 3.00
GPT4O_OUTPUT_PRICE_PER_1M_TOKENS = 12.00

PROMPT_TEMPLATE = """# Your role
You can control the computer by clicking, scrolling, dragging, and typing. 

# Credentials
The user's password is **"{CLIENT_PASSWORD}"**. Use it when a system password prompt appears.

# Rules
- Keep apps open at the end of the task.
- If the OS doesn't respond as expected, perform a brief, deterministic retry (e.g., refocus and re-click).
- Do not close the window, minimize the window unless told to do so.
- Do not ask the user for help.

# Response protocol
When you think the task is completed, reply **exactly**:
`TERMINATE: <What you did>`

When you notice the OS does not respond as the user expected, do not wait and reply **exactly**:
`UNEXPECTED: <Reason>`
""".strip()
DEFAULT_REPLY = "Please continue the user task. If you think the task is completed, reply with the following format: `TERMINATE: <1. detailed description of what you see currently. As detailed as possible. 2. What you did to complete the task or why this task cannot be completed.>`"


def _to_input_items(output_items: list) -> list:
    """
    Convert `response.output` into the JSON-serialisable items we're allowed
    to resend in the next request.  We drop anything the CUA schema doesn't
    recognise (e.g. `status`, `id`, …) and cap history length.
    """
    cleaned: List[Dict[str, Any]] = []

    for item in output_items:
        raw: Dict[str, Any] = item if isinstance(item, dict) else item.model_dump()

        # ---- strip noisy / disallowed keys ---------------------------------
        raw.pop("status", None)
        cleaned.append(raw)

    return cleaned  # keep just the most recent 50 items


def call_openai_cua(client: OpenAI,
                    history_inputs: list,
                    screen_width: int = 1920,
                    screen_height: int = 1080,
                    environment: str = "linux",
                    model: str = "computer-use-preview") -> Tuple[Any, float]:
    retry = 0
    response = None
    while retry < 3:
        try:
            response = client.responses.create(
                model=model,
                tools=[{
                    "type": "computer_use_preview",
                    "display_width": screen_width,
                    "display_height": screen_height,
                    "environment": environment,
                }],
                input=history_inputs,
                reasoning={
                    "summary": "concise"
                },
                truncation="auto",
            )
            break
        except openai.BadRequestError as e:
            retry += 1
            logger.error(f"Error in response.create: {e}")
            time.sleep(0.5)
        except openai.InternalServerError as e:
            retry += 1
            logger.error(f"Error in response.create: {e}")
            time.sleep(0.5)
    if retry == 3:
        raise Exception("Failed to call OpenAI.")

    cost = 0.0
    if response and hasattr(response, "usage") and response.usage:
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        input_cost = (input_tokens / 1_000_000) * GPT4O_INPUT_PRICE_PER_1M_TOKENS
        output_cost = (output_tokens / 1_000_000) * GPT4O_OUTPUT_PRICE_PER_1M_TOKENS
        cost = input_cost + output_cost

    return response, cost


def run_openai_cua(
    env: DesktopEnv,
    instruction: str,
    max_steps: int,
    save_path: str = './',
    screen_width: int = 1920,
    screen_height: int = 1080,
    sleep_after_execution: float = 0.3,
    truncate_history_inputs: int = 100,
    client_password: str = "",
    cua_client_config: dict = {},
    cua_model: str = "computer-use-preview",
) -> Tuple[str, float]:
    client = OpenAI(
        **cua_client_config
    )

    # 0 / reset & first screenshot
    logger.info(f"Instruction: {instruction}")
    obs = env.controller.get_screenshot()
    screenshot_b64 = base64.b64encode(obs).decode("utf-8")
    with open(os.path.join(save_path, "initial_screenshot.png"), "wb") as f:
        f.write(obs)
    history_inputs = [{
        "role": "system",
        "content": [{
            "type": "input_text",
            "text": PROMPT_TEMPLATE.format(CLIENT_PASSWORD=client_password)
        }]
    }, {
        "role": "user",
        "content": [{
            "type": "input_text",
            "text": instruction
        }, {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{screenshot_b64}"
        }]
    }]

    response, cost = call_openai_cua(client, history_inputs, screen_width, screen_height, model=cua_model)
    total_cost = cost
    logger.info(f"Cost: ${cost:.6f} | Total Cost: ${total_cost:.6f}")
    step_no = 0
    
    reasoning_list = []
    reasoning = ""

    # 1 / iterative dialogue
    while step_no < max_steps:
        step_no += 1
        history_inputs += _to_input_items(response.output)

        # --- robustly pull out computer_call(s) ------------------------------
        calls: List[Dict[str, Any]] = []
        # completed = False
        breakflag = False
        for i, o in enumerate(response.output):
            typ = o["type"] if isinstance(o, dict) else getattr(o, "type", None)
            if not isinstance(typ, str):
                typ = str(typ).split(".")[-1]
            if typ == "computer_call":
                calls.append(o if isinstance(o, dict) else o.model_dump())
            elif typ == "reasoning" and len(o.summary) > 0:
                reasoning = o.summary[0].text
                logger.info(f"[Reasoning]: {reasoning}")
            elif typ == 'message':
                if 'TERMINATE' in o.content[0].text:
                    termination_message = o.content[0].text
                    reasoning = f"{termination_message}\nPlease check the screenshot carefully and see if it fulfills your requirements."
                    breakflag = True
                    break
                if 'UNEXPECTED' in o.content[0].text:
                    reasoning = f"Unexpected system response: {o.content[0].text}. Please check the screenshot and your previous plan."
                    breakflag = True
                    break
                try:
                    json.loads(o.content[0].text)
                    history_inputs.pop(len(history_inputs) - len(response.output) + i)
                    step_no -= 1
                except Exception as e:
                    logger.info(f"[Message]: {o.content[0].text}")
                    if '?' in o.content[0].text:
                        history_inputs += [{
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": DEFAULT_REPLY},
                            ],
                        }]
                    elif "{" in o.content[0].text and "}" in o.content[0].text:
                        history_inputs.pop(len(history_inputs) - len(response.output) + i)
                        step_no -= 1
                    else:
                        logger.info(f"[Message]: {o.content[0].text}")
                        history_inputs.pop(len(history_inputs) - len(response.output) + i)
                        reasoning = o.content[0].text
                        reasoning_list.append(reasoning)
                        step_no -= 1

        if breakflag:
            break

        for action_call in calls:
            # --- execute in VM ---------------------------------------------------
            obs = env_step(env, action_call["action"], "gui", sleep_after_execution=sleep_after_execution)

            # --- send screenshot back -------------------------------------------
            while True:
                try:
                    screenshot_b64 = base64.b64encode(obs["screenshot"]).decode("utf-8")
                    break
                except Exception as e:
                    logger.error(f"Error in base64 encoding screenshot: {e}")
                    obs = {
                        "screenshot": env.controller.get_screenshot()
                    }
                    time.sleep(sleep_after_execution)
            with open(os.path.join(save_path, f"step_{step_no}.png"), "wb") as f:
                f.write(obs["screenshot"])
            history_inputs += [{
                "type": "computer_call_output",
                "call_id": action_call["call_id"],
                "output": {
                    "type": "computer_screenshot",
                    "image_url": f"data:image/png;base64,{screenshot_b64}",
                },
            }]
            if "pending_safety_checks" in action_call and len(action_call.get("pending_safety_checks", [])) > 0:
                history_inputs[-1]['acknowledged_safety_checks'] = [
                    {
                        "id": psc["id"],
                        "code": psc["code"],
                        "message": "Please acknowledge this warning if you'd like to proceed."
                    }
                    for psc in action_call.get("pending_safety_checks", [])
                ]
        
        response, cost = call_openai_cua(client, history_inputs, screen_width, screen_height, model=cua_model)
        total_cost += cost
        logger.info(f"Cost: ${cost:.6f} | Total Cost: ${total_cost:.6f}")
    
    logger.info(f"Total cost for the task: ${total_cost:.4f}")
    history_inputs[1]['content'][1]['image_url'] = "<image>"
    for item in history_inputs:
        if item.get('type', None) == 'computer_call_output':
            item['output']['image_url'] = "<image>"
    return history_inputs, reasoning, total_cost

