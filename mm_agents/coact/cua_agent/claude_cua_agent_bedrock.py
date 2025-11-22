import logging
import os
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import boto3
from tenacity import retry, stop_after_attempt, wait_exponential
from desktop_env.desktop_env import DesktopEnv
from PIL import Image

_resampling = getattr(Image, "Resampling", None)
_LANCZOS = getattr(_resampling, "LANCZOS", 1)

logger = logging.getLogger("desktopenv")

GPT4O_INPUT_PRICE_PER_1M_TOKENS = 3.00
GPT4O_OUTPUT_PRICE_PER_1M_TOKENS = 12.00

CLAUDE_COMPUTER_DISPLAY_WIDTH = 1280
CLAUDE_COMPUTER_DISPLAY_HEIGHT = 720

CLIENT_POOL = [
    'us-west-1',
    'us-west-2',
    'us-east-1',
    'us-east-2'
]

PROMPT_TEMPLATE = """# Task
{instruction}
""".strip()

SYSTEM_PROMPT = """You are controlling a computer with a display resolution of {display_width}x{display_height} pixels. 
When performing actions like clicking, use coordinates within this range.

# Instructions
- Sudo password is "{client_password}".
- Keep the windows/applications opened at the end of the task.
- Do not use shortcut to reload the application except for the browser, just close and reopen.
- If "The document has been changed by others" pops out, you should click "cancel" and reopen the file.
- If you have completed the user task, reply with the information you want the user to know along with 'TERMINATE'.
- If there is any issue, reply with the issue along with 'TERMINATE'.
""".strip()

AUTO_REPLY = "Please continue the user task. If you have completed the user task, reply with the information you want the user to know along with 'TERMINATE'. If there is any issue, reply with the issue along with 'TERMINATE'."


def _prepare_image_block(image_bytes: bytes, target_size: Optional[Tuple[int, int]]) -> Optional[Dict[str, Any]]:
    resized_data = image_bytes
    if target_size:
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                if img.size != target_size:
                    img = img.resize(target_size, _LANCZOS)
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                resized_data = buffer.getvalue()
        except Exception as e:
            # Fall back to the original payload if resizing fails
            logger.error(f"Error resizing image: {e}")

    return {
        "image": {
            "format": "png",
            "source": {
                "bytes": resized_data,
            }
        }
    }


def _convert_tool_input(
    tool_input: Dict[str, Any],
    coord_scale: Tuple[float, float],
) -> dict:
    action_name = tool_input.get("action", "wait")
    converted = {}
    scale_x, scale_y = coord_scale

    def _extract_coord(value: Any) -> Optional[Tuple[int, int]]:
        """Anthropic's computer-use tool returns either [x, y] or {'x': ..., 'y': ...}."""
        if value is None:
            return None

        x_val: Optional[Any] = None
        y_val: Optional[Any] = None
        if isinstance(value, (list, tuple)) and len(value) == 2:
            x_val, y_val = value
        elif isinstance(value, dict):
            x_val = value.get("x")
            y_val = value.get("y")

        if x_val is None or y_val is None:
            return None

        try:
            return float(x_val), float(y_val)
        except (TypeError, ValueError):
            return None

    def _scale_coord_pair(coord: Optional[Tuple[float, float]]) -> Optional[Tuple[int, int]]:
        if coord is None:
            return None
        scaled_x = int(round(coord[0] * scale_x))
        scaled_y = int(round(coord[1] * scale_y))
        return scaled_x, scaled_y

    if action_name == "mouse_move":
        coord = _scale_coord_pair(_extract_coord(tool_input.get("coordinate")))
        if coord:
            converted.update({"type": "move", "x": coord[0], "y": coord[1]})
        else:
            converted["type"] = "wait"
    
    elif action_name == "left_click_drag":
        end_coord = _scale_coord_pair(_extract_coord(tool_input.get("coordinate")))
        start_coord = _scale_coord_pair(_extract_coord(tool_input.get("start_coordinate"))) or (None, None)
        path: List[Dict[str, Optional[int]]] = []
        if start_coord:
            path.append({"x": start_coord[0], "y": start_coord[1]})
        if end_coord:
            path.append({"x": end_coord[0], "y": end_coord[1]})
        converted.update({"type": "drag", "path": path})
    
    elif action_name in {"left_click", "right_click", "middle_click", "double_click", "triple_click"}:
        coord = _scale_coord_pair(_extract_coord(tool_input.get("coordinate")))
        button_map = {
            "left_click": "left",
            "right_click": "right",
            "middle_click": "middle",
        }
        action_type = "double_click" if action_name == "double_click" else "click"
        converted.update({
            "type": action_type,
            "x": coord[0] if coord else None,
            "y": coord[1] if coord else None,
        })
        if action_type == "click":
            converted["button"] = button_map.get(action_name, "left")
    
    elif action_name == "key":
        text = tool_input.get("text", "")
        keys = [k.strip().lower() for k in text.split("+") if k.strip()]
        converted.update({"type": "keypress", "keys": keys})
    
    elif action_name == "type":
        converted.update({"type": "type", "text": tool_input.get("text", "")})
    
    elif action_name == "scroll":
        amount = int(tool_input.get("scroll_amount", 0) or 0)
        direction = tool_input.get("scroll_direction", "down")
        coord = _scale_coord_pair(_extract_coord(tool_input.get("coordinate")))
        scroll_amount = -amount if direction == "down" or direction == "left" else amount
        if "left" or "right" in direction:
            converted.update({
                "type": "scroll",
                "scroll_x": scroll_amount*1.5,
                "x": coord[0] if coord else None,
                "y": coord[1] if coord else None,
            })
        else:
            converted.update({
                "type": "scroll",
                "scroll_y": scroll_amount*1.5,
                "x": coord[0] if coord else None,
                "y": coord[1] if coord else None,
            })
    
    elif action_name == "wait":
        converted["type"] = "wait"
    
    else:
        converted["type"] = "wait"

    return converted


def _cua_to_pyautogui(action) -> str:
    """Convert an Action (dict **or** Pydantic model) into a pyautogui call."""
    
    def fld(key: str, default: Any = None) -> Any:
        return action.get(key, default) if isinstance(action, dict) else getattr(action, key, default)

    act_type = fld("type")
    if not isinstance(act_type, str):
        act_type = str(act_type).split(".")[-1]
    act_type = act_type.lower()

    if act_type in ["click", "double_click"]:
        button = fld('button', 'left')
        if button == 1 or button == 'left':
            button = 'left'
        elif button == 2 or button == 'middle':
            button = 'middle'
        elif button == 3 or button == 'right':
            button = 'right'

        if act_type == "click":
            x_val, y_val = fld('x'), fld('y')
            if x_val is None or y_val is None:
                return f"pyautogui.click(button='{button}')"
            return f"pyautogui.click({x_val}, {y_val}, button='{button}')"
        if act_type == "double_click":
            x_val, y_val = fld('x'), fld('y')
            if x_val is None or y_val is None:
                return f"pyautogui.doubleClick(button='{button}')"
            return f"pyautogui.doubleClick({x_val}, {y_val}, button='{button}')"
    
    if act_type == "scroll":
        cmd = ""
        scroll_y = fld('scroll_y', 0)
        scroll_x = fld('scroll_x', 0)
        if scroll_y != 0:
            x_val = fld('x')
            y_val = fld('y')
            if x_val is None or y_val is None:
                cmd += f"pyautogui.scroll({scroll_y});"
            else:
                cmd += f"pyautogui.scroll({scroll_y}, x={x_val}, y={y_val});"
        if scroll_x != 0:
            x_val = fld('x')
            y_val = fld('y')
            if x_val is None or y_val is None:
                cmd += f"pyautogui.hscroll({scroll_x});"
            else:
                cmd += f"pyautogui.hscroll({scroll_x}, x={x_val}, y={y_val});"
        return cmd
    
    if act_type == "drag":
        path = fld('path', [{"x": 0, "y": 0}, {"x": 0, "y": 0}])
        if not isinstance(path, list):
            path = [{"x": fld('x', 0), "y": fld('y', 0)}]
        if len(path) == 1:
            path.append(path[0])
        start_point = path[0]
        end_point = path[1]
        start_x = start_point.get('x')
        start_y = start_point.get('y')
        end_x = end_point.get('x')
        end_y = end_point.get('y')
        move_cmd = "" if start_x is None or start_y is None else f"pyautogui.moveTo({start_x}, {start_y}, _pause=False); "
        drag_cmd = f"pyautogui.dragTo({end_x}, {end_y}, duration=0.5, button='left')"
        cmd = move_cmd + drag_cmd
        return cmd

    if act_type == 'move':
        x_val, y_val = fld('x'), fld('y')
        if x_val is None or y_val is None:
            return "WAIT"
        return f"pyautogui.moveTo({x_val}, {y_val})"

    if act_type == "keypress":
        keys = fld("keys", []) or [fld("key")]
        if len(keys) == 1:
            return f"pyautogui.press('{keys[0].lower()}')"
        else:
            return "pyautogui.hotkey('{}')".format("', '".join(keys)).lower()
        
    if act_type == "type":
        text = str(fld("text", ""))
        return "pyautogui.typewrite({:})".format(repr(text))
    
    if act_type == "wait":
        return "WAIT"
    
    return "WAIT"  # fallback


@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=10, max=60), reraise=True)
def call_claude_cua(
    client: boto3.client,
    history_inputs: List[Any],
    model: str = "claude-sonnet-4",
    max_output_tokens: int = None,
    **kwargs: Any,
) -> Tuple[Any, float]:
    if not history_inputs:
        raise ValueError("Claude computer-use call requires at least one message.")

    tool_type = "computer_20250124"
    betas = ["computer-use-2025-01-24"]
    tools = [{
        "name": "computer",
        "type": tool_type,
        "display_width_px": CLAUDE_COMPUTER_DISPLAY_WIDTH,
        "display_height_px": CLAUDE_COMPUTER_DISPLAY_HEIGHT,
        "display_number": 1,
    }]

    if max_output_tokens is None:
        max_output_tokens = 4096
    response = None
    try:
        response = client.converse(
            modelId=model,
            messages=history_inputs,
            additionalModelRequestFields={
                "tools": tools,
                "anthropic_beta": betas
            },
            system=[{
                "text": SYSTEM_PROMPT.format(
                    display_width=CLAUDE_COMPUTER_DISPLAY_WIDTH, 
                    display_height=CLAUDE_COMPUTER_DISPLAY_HEIGHT,
                    client_password=kwargs['client_password'],
                )
            }],
            toolConfig={
                'tools': [
                    {
                        'toolSpec': {
                            'name': 'none',
                            'inputSchema': {
                                'json': {
                                    'type': 'object'
                                }
                            }
                        }
                    }
                ]
            }
        )
    except Exception as exc:
        logger.error("Error in Claude computer-use call: %s", exc)

    if response is None:
        return "", 0.0

    return response['output']['message'], 0.0


def run_claude_cua_bedrock(
    env: DesktopEnv,
    instruction: str,
    max_steps: int,
    save_path: str = './',
    screen_width: int = 1920,
    screen_height: int = 1080,
    sleep_after_execution: float = 1,
    client_password: str = "",
    cua_model: str = "claude-sonnet-4",
    **kwargs: Any,
) -> Tuple[str, float]:

    target_image_size = (CLAUDE_COMPUTER_DISPLAY_WIDTH, CLAUDE_COMPUTER_DISPLAY_HEIGHT)
    scale_x = screen_width / CLAUDE_COMPUTER_DISPLAY_WIDTH if CLAUDE_COMPUTER_DISPLAY_WIDTH else 1.0
    scale_y = screen_height / CLAUDE_COMPUTER_DISPLAY_HEIGHT if CLAUDE_COMPUTER_DISPLAY_HEIGHT else 1.0
    coord_scale = (scale_x, scale_y)

    # 0 / reset & first screenshot
    logger.info(f"Instruction: {instruction}")
    obs = env.controller.get_screenshot()
    # Turn the screenshot into bytes
    with open(os.path.join(save_path, "initial_screenshot.png"), "wb") as f:
        f.write(obs)
    history_inputs = [{
        "role": "user",
        "content": [
            {"text": PROMPT_TEMPLATE.format(instruction=instruction, CLIENT_PASSWORD=client_password)},
            _prepare_image_block(obs, target_image_size),
        ],
    }]

    environment = "windows" if getattr(env, "os_type", "").lower().startswith("win") else "linux"

    # api_key = _get_claude_api_key()
    # client = Anthropic(api_key=api_key, max_retries=5, **cua_client_config)
    client = boto3.client(
                service_name="bedrock-runtime",
                region_name="us-east-1"
            )

    response, cost = call_claude_cua(
        client,
        history_inputs,
        environment=environment,
        model=cua_model,
        client_password=client_password,
    )
    total_cost = cost
    logger.info(f"Cost: ${cost:.6f} | Total Cost: ${total_cost:.6f}")
    step_no = 0
    
    reasoning_list = []
    reasoning = ""

    # 1 / iterative dialogue
    while step_no < max_steps:
        step_no += 1
        history_inputs.append({
            "role": "assistant",
            "content": response['content']
        })

        # --- robustly pull out computer_call(s)
        calls: List[Dict[str, Any]] = []
        # completed = False
        breakflag = False
        for item in response['content']:
            if "toolUse" in item.keys():
                calls.append({
                    "id": item['toolUse']['toolUseId'],
                    "action": item['toolUse']['input'],
                })
            
            elif 'text' in item.keys():
                if 'TERMINATE' in item['text']:
                    reasoning_list.append(f"Final output: {item['text']}")
                    reasoning = "My thinking process\n" + "\n- ".join(reasoning_list) + '\nPlease check the screenshot and see if it fulfills your requirements.'
                    breakflag = True
                    break
                else:
                    reasoning_list.append(item['text'])

        if breakflag:
            break

        if not calls:
            history_inputs.append({
                "role": "user",
                "content": [{'text': AUTO_REPLY}]
            })
            response, cost = call_claude_cua(
                client,
                history_inputs,
                environment=environment,
                model=cua_model,
                client_password=client_password,
            )
            continue
        
        for action_call in calls:
            py_cmd = _cua_to_pyautogui(_convert_tool_input(action_call["action"], coord_scale))

            # execute in VM
            obs, *_ = env.step(py_cmd, sleep_after_execution)

            # send screenshot back
            with open(os.path.join(save_path, f"step_{step_no}.png"), "wb") as f:
                f.write(obs["screenshot"])

            history_inputs.append({
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": action_call["id"],
                            "content": [
                                {"text": "Successfully executed the action."},
                                _prepare_image_block(obs['screenshot'], target_image_size)
                            ]
                        }
                    }
                ]
            })
        
        response, cost = call_claude_cua(
            client,
            history_inputs,
            environment=environment,
            model=cua_model,
            client_password=client_password,
        )
        total_cost += cost
        logger.info(f"Cost: ${cost:.6f} | Total Cost: ${total_cost:.6f}")
    
    logger.info(f"Total cost for the task: ${total_cost:.4f}")
    history_inputs[0]['content'][1]['image_url'] = "<image>"
    for item in history_inputs:
        if item['role'] == "user":
            if len(item['content']) > 1:
                item['content'][1] = "<image>"
            else:
                item['content'] = "<image>"
    return history_inputs, reasoning, total_cost