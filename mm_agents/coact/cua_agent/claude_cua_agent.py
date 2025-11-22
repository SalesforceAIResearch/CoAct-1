import base64
import io
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from anthropic import Anthropic, APIError, APIResponseValidationError, APIStatusError
from tenacity import retry, stop_after_attempt, wait_exponential
from desktop_env.desktop_env import DesktopEnv
from PIL import Image

logger = logging.getLogger("desktopenv")

GPT4O_INPUT_PRICE_PER_1M_TOKENS = 3.00
GPT4O_OUTPUT_PRICE_PER_1M_TOKENS = 12.00

CLAUDE_COMPUTER_DISPLAY_WIDTH = 1280
CLAUDE_COMPUTER_DISPLAY_HEIGHT = 720

PROMPT_TEMPLATE = """# Task
{instruction}

# Hints
- Sudo password is "{CLIENT_PASSWORD}".
- Keep the windows/applications opened at the end of the task.
- Do not use shortcut to reload the application except for the browser, just close and reopen.
- If "The document has been changed by others" pops out, you should click "cancel" and reopen the file.
- If you have completed the user task, reply with the information you want the user to know along with 'TERMINATE'.
- If you don't know how to continue the task, reply your concern or question along with 'IDK'.
""".strip()
DEFAULT_REPLY = "Please continue the user task. If you have completed the user task, reply with the information you want the user to know along with 'TERMINATE'."


DEFAULT_CLAUDE_MODEL = "claude-sonnet-4"
CLAUDE_MODEL_ALIASES: Dict[str, str] = {
    "claude": DEFAULT_CLAUDE_MODEL,
    "claude-sonnet": DEFAULT_CLAUDE_MODEL,
    "claude-sonnet-4": DEFAULT_CLAUDE_MODEL,
}


class AttrDict(dict):
    """Dictionary subclass with attribute-style access used to mimic OpenAI SDK objects."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def model_dump(self) -> Dict[str, Any]:
        return _deep_convert(self)


def _deep_convert(value: Any) -> Any:
    if isinstance(value, AttrDict):
        return {k: _deep_convert(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_convert(v) for v in value]
    return value


def _parse_data_url(data_url: str) -> Tuple[str, Optional[str]]:
    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        return "image/png", None
    header, _, data = data_url.partition(",")
    media_type = header.split(";")[0].split(":", 1)[-1] or "image/png"
    return media_type, data or None


def _prepare_image_block(data_url: str, target_size: Optional[Tuple[int, int]]) -> Optional[Dict[str, Any]]:
    media_type, data = _parse_data_url(data_url)
    if not data:
        return None

    resized_data = data
    resized_media_type = media_type or "image/png"

    if target_size:
        try:
            raw = base64.b64decode(data)
            with Image.open(io.BytesIO(raw)) as img:
                if img.size != target_size:
                    # Use Image.Resampling.LANCZOS for newer Pillow versions, fallback to Image.LANCZOS if needed
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                resized_data = base64.b64encode(buffer.getvalue()).decode("ascii")
                resized_media_type = "image/png"
        except Exception:
            # Fall back to the original payload if resizing fails
            resized_data = data

    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": resized_media_type,
            "data": resized_data,
        },
    }


def _normalise_claude_model(model: Optional[str]) -> str:
    if not model:
        return DEFAULT_CLAUDE_MODEL
    model_lower = model.lower()
    if model_lower in CLAUDE_MODEL_ALIASES:
        return CLAUDE_MODEL_ALIASES[model_lower]
    return model


def _get_claude_api_key() -> str:
    for key_name in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "SK_ANTHROPIC"):
        api_key = os.getenv(key_name)
        if api_key:
            return api_key
    return os.getenv("CLAUDE_CUA_API_KEY", "")


def _convert_history_to_anthropic(
    history_inputs: List[Any],
    image_target_size: Optional[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    pending_thinking: Optional[str] = None

    for item in history_inputs:
        if not isinstance(item, dict):
            continue

        if item.get("role"):
            content_blocks = []
            for block in item.get("content", []):
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "input_text":
                    content_blocks.append({"type": "text", "text": block.get("text", "")})
                elif block_type == "input_image":
                    image_block = _prepare_image_block(block.get("image_url", ""), image_target_size)
                    if image_block:
                        content_blocks.append(image_block)
            if content_blocks:
                messages.append({"role": item.get("role", "user"), "content": content_blocks})
            continue

        item_type = item.get("type")
        if item_type == "message":
            role = item.get("role", "assistant")
            content_blocks = []
            # if role == "assistant" and pending_thinking:
            #     content_blocks.append({"type": "thinking", "thinking": pending_thinking, "signature": ""})
            #     pending_thinking = None
            for block in item.get("content", []):
                if isinstance(block, dict):
                    text_value = block.get("text") or block.get("content")
                    if text_value:
                        content_blocks.append({"type": "text", "text": text_value})
            if content_blocks:
                messages.append({"role": role, "content": content_blocks})
        elif item_type == "computer_call":
            claude_input = item.get("claude_input")
            if claude_input is None:
                claude_input = _convert_openai_action_to_claude(item.get("action", {}))
            content_blocks = []
            # if pending_thinking:
            #     content_blocks.append({"type": "thinking", "thinking": pending_thinking, "signature": ""})
            #     pending_thinking = None
            content_blocks.append({
                "type": "tool_use",
                "id": item.get("call_id"),
                "name": item.get("name", "computer"),
                "input": claude_input,
            })
            messages.append({
                "role": "assistant",
                "content": content_blocks,
            })
        elif item_type == "computer_call_output":
            output = item.get("output", {})
            result_content = []
            image_block = _prepare_image_block(output.get("image_url", ""), image_target_size)
            if image_block:
                result_content.append(image_block)
            acknowledged = item.get("acknowledged_safety_checks")
            if acknowledged:
                result_content.append({
                    "type": "text",
                    "text": "Acknowledged safety checks.",
                })
            if not result_content:
                result_content.append({"type": "text", "text": "No screenshot captured."})
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": item.get("call_id"),
                    "content": result_content,
                }],
            })
        # elif item_type == "reasoning":
            # summary_blocks = item.get("summary", [])
            # text_parts = []
            # for block in summary_blocks:
            #     if isinstance(block, dict) and block.get("text"):
            #         text_parts.append(block["text"])
            # if text_parts:
            #     pending_thinking = "\n".join(text_parts)

    return messages


def _convert_openai_action_to_claude(action: Any) -> Dict[str, Any]:
    if not isinstance(action, dict):
        return {"action": "wait"}
    act_type = action.get("type")
    if act_type == "move":
        return {"action": "mouse_move", "coordinate": [action.get("x", 0), action.get("y", 0)]}
    if act_type == "drag":
        path = action.get("path", [])
        start = path[0] if path else {"x": action.get("x", 0), "y": action.get("y", 0)}
        end = path[-1] if path else start
        return {
            "action": "left_click_drag",
            "start_coordinate": [start.get("x", 0), start.get("y", 0)],
            "coordinate": [end.get("x", 0), end.get("y", 0)],
        }
    if act_type == "click":
        button_map = {"left": "left_click", "right": "right_click", "middle": "middle_click"}
        return {
            "action": button_map.get(action.get("button", "left"), "left_click"),
            "coordinate": [action.get("x", 0), action.get("y", 0)],
        }
    if act_type == "double_click":
        return {
            "action": "double_click",
            "coordinate": [action.get("x", 0), action.get("y", 0)],
        }
    if act_type == "keypress":
        keys = action.get("keys") or []
        return {"action": "key", "text": "+".join(keys)}
    if act_type == "type":
        return {"action": "type", "text": action.get("text", "")}
    if act_type == "scroll":
        amount = action.get("scroll_y", 0)
        direction = "down" if amount >= 0 else "up"
        return {
            "action": "scroll",
            "scroll_direction": direction,
            "scroll_amount": abs(amount),
            "coordinate": [action.get("x", 0), action.get("y", 0)],
        }
    if act_type == "wait":
        return {"action": "wait", "duration": action.get("time", 0.5)}
    return {"action": "wait"}


def _convert_tool_input(
    tool_input: Dict[str, Any],
    last_cursor: Optional[Tuple[int, int]],
    coord_scale: Tuple[float, float],
) -> Tuple[AttrDict, Optional[Tuple[int, int]]]:
    action_name = tool_input.get("action", "wait")
    converted = AttrDict()
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
            last_cursor = coord
        else:
            converted["type"] = "wait"
    elif action_name == "left_click_drag":
        end_coord = _scale_coord_pair(_extract_coord(tool_input.get("coordinate")))
        start_coord = _scale_coord_pair(_extract_coord(tool_input.get("start_coordinate"))) or last_cursor
        path: List[Dict[str, Optional[int]]] = []
        if start_coord:
            path.append({"x": start_coord[0], "y": start_coord[1]})
        if end_coord:
            path.append({"x": end_coord[0], "y": end_coord[1]})
            last_cursor = end_coord
        converted.update({"type": "drag", "path": path})
    elif action_name in {"left_click", "right_click", "middle_click", "double_click", "triple_click"}:
        coord = _scale_coord_pair(_extract_coord(tool_input.get("coordinate"))) or last_cursor
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
        last_cursor = coord or last_cursor
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
        scroll_y = amount if direction == "down" else -amount
        converted.update({
            "type": "scroll",
            "scroll_y": scroll_y,
            "x": coord[0] if coord else None,
            "y": coord[1] if coord else None,
        })
    elif action_name == "wait":
        converted["type"] = "wait"
    else:
        converted["type"] = "wait"

    return converted, last_cursor


def _convert_claude_response(response: Any, coord_scale: Tuple[float, float]) -> Tuple[List[AttrDict], AttrDict]:
    output: List[AttrDict] = []
    last_cursor: Optional[Tuple[int, int]] = None

    for block in getattr(response, "content", []) or []:
        block_type = getattr(block, "type", None)

        if block_type == "thinking":
            thinking_text = getattr(block, "thinking", None) or getattr(block, "text", "")
            if thinking_text:
                reasoning_item = AttrDict(
                    type="reasoning",
                    reasoning=[AttrDict(type="output_text", text=thinking_text)],
                    summary=[AttrDict(type="output_text", text=thinking_text)],
                )
                output.append(reasoning_item)
        elif block_type == "text":
            text_value = getattr(block, "text", "")
            message_item = AttrDict(
                type="message",
                role="assistant",
                content=[AttrDict(type="output_text", text=text_value)],
            )
            output.append(message_item)
        elif block_type == "tool_use":
            tool_dict = block.model_dump() if hasattr(block, "model_dump") else dict(block)
            tool_input = tool_dict.get("input", {}) if isinstance(tool_dict, dict) else {}
            converted_action, last_cursor = _convert_tool_input(tool_input, last_cursor, coord_scale)
            call_item = AttrDict(
                type="computer_call",
                call_id=tool_dict.get("id"),
                name=tool_dict.get("name", "computer"),
                action=converted_action,
                claude_input=tool_input,
            )
            output.append(call_item)

    raw_usage = getattr(response, "usage", {}) or {}
    if hasattr(raw_usage, "model_dump"):
        raw_usage = raw_usage.model_dump()
    usage = AttrDict(
        input_tokens=raw_usage.get("input_tokens", 0),
        output_tokens=raw_usage.get("output_tokens", 0),
    )

    return output, usage


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
        if scroll_y != 0:
            x_val = fld('x')
            y_val = fld('y')
            if x_val is None or y_val is None:
                cmd += f"pyautogui.scroll({-scroll_y / 100});"
            else:
                cmd += f"pyautogui.scroll({-scroll_y / 100}, x={x_val}, y={y_val});"
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
        target_x = end_x if end_x is not None else (start_x if start_x is not None else 0)
        target_y = end_y if end_y is not None else (start_y if start_y is not None else 0)
        drag_cmd = f"pyautogui.dragTo({target_x}, {target_y}, duration=0.5, button='left')"
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


class ClaudeResponse:
    def __init__(self, output: List[AttrDict], usage: AttrDict):
        self.output = output
        self.usage = usage


@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=10, max=60), reraise=True)
def call_claude_cua(
    history_inputs: List[Any],
    screen_width: int = 1920,
    screen_height: int = 1080,
    environment: str = "linux",
    model: Optional[str] = None,
    max_output_tokens: int = None,
) -> Tuple[Any, float]:
    claude_model = _normalise_claude_model(model)
    print("Using Claude model:", claude_model)
    target_image_size = (CLAUDE_COMPUTER_DISPLAY_WIDTH, CLAUDE_COMPUTER_DISPLAY_HEIGHT)
    scale_x = screen_width / CLAUDE_COMPUTER_DISPLAY_WIDTH if CLAUDE_COMPUTER_DISPLAY_WIDTH else 1.0
    scale_y = screen_height / CLAUDE_COMPUTER_DISPLAY_HEIGHT if CLAUDE_COMPUTER_DISPLAY_HEIGHT else 1.0
    coord_scale = (scale_x, scale_y)
    messages = _convert_history_to_anthropic(history_inputs, target_image_size)
    if not messages:
        raise ValueError("Claude computer-use call requires at least one message.")


    api_key = _get_claude_api_key()
    client = Anthropic(api_key=api_key, max_retries=0)

    tool_type = "computer_20250124"
    betas = ["computer-use-2025-01-24"]
    tools = [{
        "name": "computer",
        "type": tool_type,
        "display_width_px": CLAUDE_COMPUTER_DISPLAY_WIDTH,
        "display_height_px": CLAUDE_COMPUTER_DISPLAY_HEIGHT,
        "display_number": 1,
    }]
    # thinking_budget = 1024
    # if max_output_tokens is None or max_output_tokens <= thinking_budget:
    #     max_output_tokens = thinking_budget + 1
    # extra_body = {"thinking": {"type": "enabled", "budget_tokens": thinking_budget}}

    if max_output_tokens is None:
        max_output_tokens = 4096
    response = None
    try:
        response = client.beta.messages.create(
            model=claude_model,
            messages=messages,
            tools=tools,
            betas=betas,
            max_tokens=max_output_tokens,
            # extra_body=extra_body,
            system=(
                f"You are controlling a computer with a display resolution of "
                f"{CLAUDE_COMPUTER_DISPLAY_WIDTH}x{CLAUDE_COMPUTER_DISPLAY_HEIGHT} pixels. "
                "When performing actions like clicking, use coordinates within this range."
            )
        )
    except (APIError, APIStatusError, APIResponseValidationError) as exc:
        logger.error("Error in Claude computer-use call: %s", exc)

    if response is None:
        raise RuntimeError("Failed to call Claude computer-use API after retries.")

    output, usage = _convert_claude_response(response, coord_scale)
    return ClaudeResponse(output, usage), 0.0


def run_claude_cua(
    env: DesktopEnv,
    instruction: str,
    max_steps: int,
    save_path: str = './',
    screen_width: int = 1920,
    screen_height: int = 1080,
    sleep_after_execution: float = 0.3,
    client_password: str = "",
    cua_model: str = "claude-sonnet-4",
    **kwargs: Any,
) -> Tuple[str, float]:
    # 0 / reset & first screenshot
    logger.info(f"Instruction: {instruction}")
    obs = env.controller.get_screenshot()
    screenshot_b64 = base64.b64encode(obs).decode("utf-8")
    with open(os.path.join(save_path, "initial_screenshot.png"), "wb") as f:
        f.write(obs)
    history_inputs = [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": PROMPT_TEMPLATE.format(instruction=instruction, CLIENT_PASSWORD=client_password)},
            {"type": "input_image", "image_url": f"data:image/png;base64,{screenshot_b64}"},
        ],
    }]

    environment = "windows" if getattr(env, "os_type", "").lower().startswith("win") else "linux"

    response, cost = call_claude_cua(
        history_inputs,
        screen_width=screen_width,
        screen_height=screen_height,
        environment=environment,
        model=cua_model,
    )
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
                reasoning_list.append(reasoning)
                logger.info(f"[Reasoning]: {reasoning}")
            elif typ == 'message':
                if 'TERMINATE' in o.content[0].text:
                    reasoning_list.append(f"Final output: {o.content[0].text}")
                    reasoning = "My thinking process\n" + "\n- ".join(reasoning_list) + '\nPlease check the screenshot and see if it fulfills your requirements.'
                    breakflag = True
                    break
                elif 'IDK' in o.content[0].text:
                    reasoning = f"{o.content[0].text}. I don't know how to complete the task. Please check the current screenshot."
                    breakflag = True
                    break
                else:
                    logger.info(f"[Message]: {o.content[0].text}")
                    reasoning = o.content[0].text
                    reasoning_list.append(reasoning)

        if breakflag:
            break

        for action_call in calls:
            py_cmd = _cua_to_pyautogui(action_call["action"])

            # --- execute in VM ---------------------------------------------------
            obs, *_ = env.step(py_cmd, sleep_after_execution)

            # --- send screenshot back -------------------------------------------
            screenshot_b64 = base64.b64encode(obs["screenshot"]).decode("utf-8")
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

        response, cost = call_claude_cua(
            history_inputs,
            screen_width=screen_width,
            screen_height=screen_height,
            environment=environment,
            model=cua_model,
        )
        total_cost += cost
        logger.info(f"Cost: ${cost:.6f} | Total Cost: ${total_cost:.6f}")
    
    logger.info(f"Total cost for the task: ${total_cost:.4f}")
    history_inputs[0]['content'][1]['image_url'] = "<image>"
    for item in history_inputs:
        if item.get('type', None) == 'computer_call_output':
            item['output']['image_url'] = "<image>"
    return history_inputs, reasoning, total_cost