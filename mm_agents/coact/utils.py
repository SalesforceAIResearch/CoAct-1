import time
from typing import Any
from desktop_env.desktop_env import DesktopEnv
from .autogen.code_utils import PYTHON_VARIANTS


def _openaicua_to_pyautogui(action) -> str:
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
            return f"pyautogui.click({fld('x')}, {fld('y')}, button='{button}')"
        if act_type == "double_click":
            return f"pyautogui.doubleClick({fld('x')}, {fld('y')}, button='{button}')"
        
    if act_type == "scroll":
        cmd = ""
        if fld('scroll_y', 0) != 0:
            cmd += f"pyautogui.scroll({-fld('scroll_y', 0) / 110}, x={fld('x', 0)}, y={fld('y', 0)});"
        return cmd

    if act_type == "drag":
        path = fld('path', [{"x": 0, "y": 0}, {"x": 0, "y": 0}])
        cmd = f"pyautogui.moveTo({path[0]['x']}, {path[0]['y']}, _pause=False); "
        cmd += f"pyautogui.dragTo({path[1]['x']}, {path[1]['y']}, duration=1.0, button='left')"
        return cmd

    if act_type == 'move':
        return f"pyautogui.moveTo({fld('x')}, {fld('y')})"

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


def env_step(env: DesktopEnv, inputs: str, type: str, **kwargs):
    if type == "coding":
        exitcode = 1 # 1 means error, 0 means success
        logs = ""
        image = None
        lang = kwargs.get("lang", "")
        if lang in ["bash", "shell", "sh"]:
            timeout = kwargs.get("timeout", 30)
            working_dir = kwargs.get("working_dir")
            output_dict = env.controller.run_bash_script(inputs, timeout=timeout, working_dir=working_dir)
            if not output_dict:
                return -1, "bash execution returned no result", image
            if output_dict.get("status") == "success":
                exitcode = 0
                logs = output_dict.get("output", "")
            else:
                exitcode = 1
                stdout = output_dict.get("output", "")
                stderr = output_dict.get("error", "")
                logs = (stdout + ("\n" if stdout and stderr else "") + stderr).strip()
        elif lang in PYTHON_VARIANTS:
            timeout = kwargs.get("timeout", 90)
            output_dict = env.controller.run_python_script(inputs, timeout=timeout)
            if not output_dict:
                return -1, "python execution returned no result", image
            if output_dict.get("status") == "success":
                exitcode = 0
                # Prefer stdout when present; otherwise use message
                logs = output_dict.get("output") or output_dict.get("message", "")
            else:
                exitcode = 1
                stdout = output_dict.get("output", "")
                stderr = output_dict.get("error", "")
                message = output_dict.get("message", "")
                combined = stdout
                if combined and (stderr or message):
                    combined += "\n"
                combined += stderr or message
                logs = combined.strip()
        else:
            exitcode = -1
            logs = f"unknown language {lang}"
        return {
            "exitcode": exitcode,
            "logs": logs,
            "image": image
        }
    if type == "gui":
        py_cmd = _openaicua_to_pyautogui(inputs)
        output_dict, *_ = env.step(py_cmd, kwargs.get("sleep_after_execution", 0.5))
        return output_dict
