import json
import logging
import os
import random
from typing import Any, Dict, Optional
import time
import traceback
import requests

from desktop_env.actions import KEYBOARD_KEYS

logger = logging.getLogger("desktopenv.pycontroller")


class PythonController:
    def __init__(self, vm_ip: str,
                 server_port: int,
                 pkgs_prefix: str = "import pyautogui; import time; pyautogui.FAILSAFE = False; {command}"):
        self.vm_ip = vm_ip
        self.http_server = f"http://{vm_ip}:{server_port}"
        self.pkgs_prefix = pkgs_prefix  # fixme: this is a hacky way to execute python commands. fix it and combine it with installation of packages
        self.retry_times = 3
        self.retry_interval = 5

    @staticmethod
    def _is_valid_image_response(content_type: str, data: Optional[bytes]) -> bool:
        """Quick validation for PNG/JPEG payload using magic bytes; Content-Type is advisory.
        Returns True only when bytes look like a real PNG or JPEG.
        """
        if not isinstance(data, (bytes, bytearray)) or not data:
            return False
        # PNG magic
        if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
            return True
        # JPEG magic
        if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
            return True
        # If server explicitly marks as image, accept as a weak fallback (some environments strip magic)
        if content_type and ("image/png" in content_type or "image/jpeg" in content_type or "image/jpg" in content_type):
            return True
        return False

    def get_screenshot(self) -> Optional[bytes]:
        """
        Gets a screenshot from the server. With the cursor. None -> no screenshot or unexpected error.
        """

        for attempt_idx in range(self.retry_times):
            try:
                response = requests.get(self.http_server + "/screenshot", timeout=10)
                if response.status_code == 200:
                    content_type = response.headers.get("Content-Type", "")
                    content = response.content
                    if self._is_valid_image_response(content_type, content):
                        logger.info("Got screenshot successfully")
                        return content
                    else:
                        logger.error("Invalid screenshot payload (attempt %d/%d).", attempt_idx + 1, self.retry_times)
                        logger.info("Retrying to get screenshot.")
                else:
                    logger.error("Failed to get screenshot. Status code: %d", response.status_code)
                    logger.info("Retrying to get screenshot.")
            except Exception as e:
                logger.error("An error occurred while trying to get the screenshot: %s", e)
                logger.info("Retrying to get screenshot.")
            time.sleep(self.retry_interval)

        logger.error("Failed to get screenshot.")
        return None

    def set_vm_screen_size(self, width: int, height: int):
        """
        Sets the size of the vm screen.
        """
        response = requests.post(self.http_server + "/set_screen_resolution", json={"width": width, "height": height})
        if response.status_code == 200:
            logger.debug("Set screen size successfully")
            return response.json()
        else:
            logger.error("Failed to set screen size. Status code: %d", response.status_code)
            logger.debug("Retrying to set screen size.")
            return None

    def get_accessibility_tree(self) -> Optional[str]:
        """
        Gets the accessibility tree from the server. None -> no accessibility tree or unexpected error.
        """

        for _ in range(self.retry_times):
            try:
                response: requests.Response = requests.get(self.http_server + "/accessibility")
                if response.status_code == 200:
                    logger.info("Got accessibility tree successfully")
                    return response.json()["AT"]
                else:
                    logger.error("Failed to get accessibility tree. Status code: %d", response.status_code)
                    logger.info("Retrying to get accessibility tree.")
            except Exception as e:
                logger.error("An error occurred while trying to get the accessibility tree: %s", e)
                logger.info("Retrying to get accessibility tree.")
            time.sleep(self.retry_interval)

        logger.error("Failed to get accessibility tree.")
        return None

    def get_terminal_output(self) -> Optional[str]:
        """
        Gets the terminal output from the server. None -> no terminal output or unexpected error.
        """

        for _ in range(self.retry_times):
            try:
                response = requests.get(self.http_server + "/terminal")
                if response.status_code == 200:
                    logger.info("Got terminal output successfully")
                    return response.json()["output"]
                else:
                    logger.error("Failed to get terminal output. Status code: %d", response.status_code)
                    logger.info("Retrying to get terminal output.")
            except Exception as e:
                logger.error("An error occurred while trying to get the terminal output: %s", e)
                logger.info("Retrying to get terminal output.")
            time.sleep(self.retry_interval)

        logger.error("Failed to get terminal output.")
        return None

    def get_file(self, file_path: str) -> Optional[bytes]:
        """
        Gets a file from the server.
        """

        for _ in range(self.retry_times):
            try:
                response = requests.post(self.http_server + "/file", data={"file_path": file_path})
                if response.status_code == 200:
                    logger.info("File downloaded successfully")
                    return response.content
                else:
                    logger.error("Failed to get file. Status code: %d", response.status_code)
                    logger.info("Retrying to get file.")
            except Exception as e:
                logger.error("An error occurred while trying to get the file: %s", e)
                logger.info("Retrying to get file.")
            time.sleep(self.retry_interval)

        logger.error("Failed to get file.")
        return None

    def execute_python_command(self, command: str) -> None:
        """
        Executes a python command on the server.
        It can be used to execute the pyautogui commands, or... any other python command. who knows?
        """
        # command_list = ["python", "-c", self.pkgs_prefix.format(command=command)]
        command_list = ["python", "-c", self.pkgs_prefix.format(command=command)]
        payload = json.dumps({"command": command_list, "shell": False})

        for _ in range(self.retry_times):
            try:
                response = requests.post(self.http_server + "/execute", headers={'Content-Type': 'application/json'},
                                         data=payload, timeout=90)
                if response.status_code == 200:
                    logger.info("Command executed successfully: %s", response.text)
                    return response.json()
                else:
                    logger.error("Failed to execute command. Status code: %d", response.status_code)
                    logger.info("Retrying to execute command.")
            except requests.exceptions.ReadTimeout:
                break
            except Exception as e:
                logger.error("An error occurred while trying to execute the command: %s", e)
                logger.info("Retrying to execute command.")
            time.sleep(self.retry_interval)

        logger.error("Failed to execute command.")
        return None
    
    def run_python_script(self, script: str, timeout: int = 90) -> Optional[Dict[str, Any]]:
        """Execute a Python script via the server's /run_python_script endpoint."""
        payload = json.dumps({"code": script, "timeout": timeout})

        for _ in range(self.retry_times):
            try:
                response = requests.post(
                    self.http_server + "/run_python_script",
                    headers={'Content-Type': 'application/json'},
                    data=payload,
                    timeout=timeout + 10,
                )
                if response.status_code == 200:
                    return response.json()
                # Try to return server-provided error if available
                try:
                    return response.json()
                except Exception:
                    return {
                        "status": "error",
                        "message": f"Failed to execute python script (HTTP {response.status_code})",
                        "output": "",
                        "error": response.text,
                        "returncode": -1,
                    }
            except requests.exceptions.ReadTimeout:
                return {
                    "status": "error",
                    "message": "Script execution timed out",
                    "output": "",
                    "error": f"Timed out after {timeout} seconds",
                    "returncode": -1,
                }
            except Exception:
                logger.error("An error occurred while trying to execute the python script: %s", traceback.format_exc())
                logger.info("Retrying to execute command.")
            time.sleep(self.retry_interval)

        logger.error("Failed to execute python script.")
        return {
            "status": "error",
            "message": "Failed to execute command.",
            "output": "",
            "error": "Retry limit reached.",
            "returncode": -1,
        }
    
    def run_bash_script(self, script: str, timeout: int = 30, working_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Executes a bash script on the server.
        
        :param script: The bash script content (can be multi-line)
        :param timeout: Execution timeout in seconds (default: 30)
        :param working_dir: Working directory for script execution (optional)
        :return: Dictionary with status, output, error, and returncode, or None if failed
        """
        payload = json.dumps({
            "script": script,
            "timeout": timeout,
            "working_dir": working_dir
        })

        for _ in range(self.retry_times):
            try:
                response = requests.post(
                    self.http_server + "/run_bash_script", 
                    headers={'Content-Type': 'application/json'},
                    data=payload, 
                    timeout=timeout + 100  # Add buffer to HTTP timeout
                )
                if response.status_code == 200:
                    result = response.json()
                    logger.info("Bash script executed successfully with return code: %d", result.get("returncode", -1))
                    return result
                else:
                    logger.error("Failed to execute bash script. Status code: %d, response: %s", 
                                response.status_code, response.text)
                    logger.info("Retrying to execute bash script.")
            except requests.exceptions.ReadTimeout:
                logger.error("Bash script execution timed out")
                return {
                    "status": "error",
                    "output": "",
                    "error": f"Script execution timed out after {timeout} seconds",
                    "returncode": -1
                }
            except Exception as e:
                logger.error("An error occurred while trying to execute the bash script: %s", e)
                logger.info("Retrying to execute bash script.")
            time.sleep(self.retry_interval)

        logger.error("Failed to execute bash script after %d retries.", self.retry_times)
        return {
            "status": "error",
            "output": "",
            "error": f"Failed to execute bash script after {self.retry_times} retries",
            "returncode": -1
        }

    def execute_action(self, action: Dict[str, Any]):
        """
        Executes an action on the server computer.
        """
        if action in ['WAIT', 'FAIL', 'DONE']:
            return

        action_type = action["action_type"]
        parameters = action["parameters"] if "parameters" in action else {param: action[param] for param in action if param != 'action_type'}
        move_mode = random.choice(
            ["pyautogui.easeInQuad", "pyautogui.easeOutQuad", "pyautogui.easeInOutQuad", "pyautogui.easeInBounce",
             "pyautogui.easeInElastic"])
        duration = random.uniform(0.5, 1)

        if action_type == "MOVE_TO":
            if parameters == {} or None:
                self.execute_python_command("pyautogui.moveTo()")
            elif "x" in parameters and "y" in parameters:
                x = parameters["x"]
                y = parameters["y"]
                self.execute_python_command(f"pyautogui.moveTo({x}, {y}, {duration}, {move_mode})")
            else:
                raise Exception(f"Unknown parameters: {parameters}")

        elif action_type == "CLICK":
            if parameters == {} or None:
                self.execute_python_command("pyautogui.click()")
            elif "button" in parameters and "x" in parameters and "y" in parameters:
                button = parameters["button"]
                x = parameters["x"]
                y = parameters["y"]
                if "num_clicks" in parameters:
                    num_clicks = parameters["num_clicks"]
                    self.execute_python_command(
                        f"pyautogui.click(button='{button}', x={x}, y={y}, clicks={num_clicks})")
                else:
                    self.execute_python_command(f"pyautogui.click(button='{button}', x={x}, y={y})")
            elif "button" in parameters and "x" not in parameters and "y" not in parameters:
                button = parameters["button"]
                if "num_clicks" in parameters:
                    num_clicks = parameters["num_clicks"]
                    self.execute_python_command(f"pyautogui.click(button='{button}', clicks={num_clicks})")
                else:
                    self.execute_python_command(f"pyautogui.click(button='{button}')")
            elif "button" not in parameters and "x" in parameters and "y" in parameters:
                x = parameters["x"]
                y = parameters["y"]
                if "num_clicks" in parameters:
                    num_clicks = parameters["num_clicks"]
                    self.execute_python_command(f"pyautogui.click(x={x}, y={y}, clicks={num_clicks})")
                else:
                    self.execute_python_command(f"pyautogui.click(x={x}, y={y})")
            else:
                raise Exception(f"Unknown parameters: {parameters}")

        elif action_type == "MOUSE_DOWN":
            if parameters == {} or None:
                self.execute_python_command("pyautogui.mouseDown()")
            elif "button" in parameters:
                button = parameters["button"]
                self.execute_python_command(f"pyautogui.mouseDown(button='{button}')")
            else:
                raise Exception(f"Unknown parameters: {parameters}")

        elif action_type == "MOUSE_UP":
            if parameters == {} or None:
                self.execute_python_command("pyautogui.mouseUp()")
            elif "button" in parameters:
                button = parameters["button"]
                self.execute_python_command(f"pyautogui.mouseUp(button='{button}')")
            else:
                raise Exception(f"Unknown parameters: {parameters}")

        elif action_type == "RIGHT_CLICK":
            if parameters == {} or None:
                self.execute_python_command("pyautogui.rightClick()")
            elif "x" in parameters and "y" in parameters:
                x = parameters["x"]
                y = parameters["y"]
                self.execute_python_command(f"pyautogui.rightClick(x={x}, y={y})")
            else:
                raise Exception(f"Unknown parameters: {parameters}")

        elif action_type == "DOUBLE_CLICK":
            if parameters == {} or None:
                self.execute_python_command("pyautogui.doubleClick()")
            elif "x" in parameters and "y" in parameters:
                x = parameters["x"]
                y = parameters["y"]
                self.execute_python_command(f"pyautogui.doubleClick(x={x}, y={y})")
            else:
                raise Exception(f"Unknown parameters: {parameters}")

        elif action_type == "DRAG_TO":
            if "x" in parameters and "y" in parameters:
                x = parameters["x"]
                y = parameters["y"]
                self.execute_python_command(
                    f"pyautogui.dragTo({x}, {y}, duration=1.0, button='left', mouseDownUp=True)")

        elif action_type == "SCROLL":
            # todo: check if it is related to the operating system, as https://github.com/TheDuckAI/DuckTrack/blob/main/ducktrack/playback.py pointed out
            if "dx" in parameters and "dy" in parameters:
                dx = parameters["dx"]
                dy = parameters["dy"]
                self.execute_python_command(f"pyautogui.hscroll({dx})")
                self.execute_python_command(f"pyautogui.vscroll({dy})")
            elif "dx" in parameters and "dy" not in parameters:
                dx = parameters["dx"]
                self.execute_python_command(f"pyautogui.hscroll({dx})")
            elif "dx" not in parameters and "dy" in parameters:
                dy = parameters["dy"]
                self.execute_python_command(f"pyautogui.vscroll({dy})")
            else:
                raise Exception(f"Unknown parameters: {parameters}")

        elif action_type == "TYPING":
            if "text" not in parameters:
                raise Exception(f"Unknown parameters: {parameters}")
            # deal with special ' and \ characters
            # text = parameters["text"].replace("\\", "\\\\").replace("'", "\\'")
            # self.execute_python_command(f"pyautogui.typewrite('{text}')")
            text = parameters["text"]
            self.execute_python_command("pyautogui.typewrite({:})".format(repr(text)))

        elif action_type == "PRESS":
            if "key" not in parameters:
                raise Exception(f"Unknown parameters: {parameters}")
            key = parameters["key"]
            if key.lower() not in KEYBOARD_KEYS:
                raise Exception(f"Key must be one of {KEYBOARD_KEYS}")
            self.execute_python_command(f"pyautogui.press('{key}')")

        elif action_type == "KEY_DOWN":
            if "key" not in parameters:
                raise Exception(f"Unknown parameters: {parameters}")
            key = parameters["key"]
            if key.lower() not in KEYBOARD_KEYS:
                raise Exception(f"Key must be one of {KEYBOARD_KEYS}")
            self.execute_python_command(f"pyautogui.keyDown('{key}')")

        elif action_type == "KEY_UP":
            if "key" not in parameters:
                raise Exception(f"Unknown parameters: {parameters}")
            key = parameters["key"]
            if key.lower() not in KEYBOARD_KEYS:
                raise Exception(f"Key must be one of {KEYBOARD_KEYS}")
            self.execute_python_command(f"pyautogui.keyUp('{key}')")

        elif action_type == "HOTKEY":
            if "keys" not in parameters:
                raise Exception(f"Unknown parameters: {parameters}")
            keys = parameters["keys"]
            if not isinstance(keys, list):
                raise Exception("Keys must be a list of keys")
            for key in keys:
                if key.lower() not in KEYBOARD_KEYS:
                    raise Exception(f"Key must be one of {KEYBOARD_KEYS}")

            keys_para_rep = "', '".join(keys)
            self.execute_python_command(f"pyautogui.hotkey('{keys_para_rep}')")

        elif action_type in ['WAIT', 'FAIL', 'DONE']:
            pass

        else:
            raise Exception(f"Unknown action type: {action_type}")

    # Record video
    def start_recording(self):
        """
        Starts recording the screen.
        """

        for _ in range(self.retry_times):
            try:
                response = requests.post(self.http_server + "/start_recording")
                if response.status_code == 200:
                    logger.info("Recording started successfully")
                    return
                else:
                    logger.error("Failed to start recording. Status code: %d", response.status_code)
                    logger.info("Retrying to start recording.")
            except Exception as e:
                logger.error("An error occurred while trying to start recording: %s", e)
                logger.info("Retrying to start recording.")
            time.sleep(self.retry_interval)

        logger.error("Failed to start recording.")

    def end_recording(self, dest: str, connect_timeout: float = 5.0, read_timeout: float = 600.0,
                      chunk_size: int = 1024 * 1024) -> None:
        url = self.http_server.rstrip("/") + "/end_recording"
        tmp_path = dest + ".part"

        if os.path.exists(dest) and not os.path.exists(tmp_path):
            logger.info("Target file already exists, skipping download: %s", dest)
            return

        start_offset = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
        base_sleep = max(0.5, self.retry_interval)
        session = requests.Session()

        for attempt in range(1, self.retry_times + 1):
            try:
                headers = {}
                if start_offset > 0:
                    headers["Range"] = f"bytes={start_offset}-"

                logger.info("Stopping recording & fetching video (attempt %d/%d). Range: %s",
                            attempt, self.retry_times, headers.get("Range", "full"))

                with session.post(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=(connect_timeout, read_timeout)
                ) as response:
                    status = response.status_code

                    if status not in (200, 206):
                        logger.error("Failed to stop recording. HTTP %s", status)
                        raise RuntimeError(f"Unexpected HTTP status {status}")

                    content_range: Optional[str] = response.headers.get("Content-Range")
                    is_resuming = (status == 206 and content_range and content_range.startswith("bytes"))

                    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
                    with open(tmp_path, "ab" if is_resuming and start_offset > 0 else "wb") as f:
                        bytes_written_this_round = 0
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if not chunk:
                                continue
                            f.write(chunk)
                            bytes_written_this_round += len(chunk)

                        try:
                            f.flush()
                            os.fsync(f.fileno())
                        except OSError as ioe:
                            logger.error("Flush/fsync failed: %s", ioe)
                            raise
                    try:
                        if status == 206 and content_range:
                            _, rng = content_range.split(" ", 1)
                            byte_range, total_str = rng.split("/")
                            start_str, end_str = byte_range.split("-")
                            end_pos = int(end_str)
                            total = int(total_str)
                            final_size = end_pos + 1
                            actual_size = os.path.getsize(tmp_path)
                            if actual_size != final_size:
                                raise RuntimeError(
                                    f"Resume size mismatch: expected {final_size}, got {actual_size}"
                                )
                            logger.info("Resumed download ok. %d/%d bytes.", actual_size, total)
                        elif status == 200:
                            cl = response.headers.get("Content-Length")
                            if cl is not None:
                                expected = int(cl)
                                actual = os.path.getsize(tmp_path)
                                if actual != expected:
                                    raise RuntimeError(
                                        f"Size mismatch: expected {expected}, got {actual}"
                                    )
                            logger.info("Full download ok. Size=%d bytes.", os.path.getsize(tmp_path))
                    except Exception as verify_err:
                        logger.error("Verification failed: %s", verify_err)
                        raise

                    os.replace(tmp_path, dest)
                    logger.info("Recording stopped successfully. Saved to %s", dest)
                    return

            except (requests.Timeout, requests.ConnectionError) as net_err:
                logger.error("Network error: %s", net_err)
            except OSError as io_err:
                logger.error("File system error (disk full / permission?): %s", io_err)
                break
            except Exception as e:
                logger.error("Unexpected error: %s", e)

            if attempt < self.retry_times:
                sleep_s = min(base_sleep * (2 ** (attempt - 1)), 60.0)
                logger.info("Retrying in %.1f seconds...", sleep_s)
                time.sleep(sleep_s)

        logger.error("Failed to stop recording after %d attempts.", self.retry_times)

    # Additional info
    def get_vm_platform(self):
        """
        Gets the size of the vm screen.
        """
        return self.execute_python_command("import platform; print(platform.system())")['output'].strip()

    def get_vm_screen_size(self):
        """
        Gets the size of the vm screen.
        """

        for _ in range(self.retry_times):
            try:
                response = requests.post(self.http_server + "/screen_size")
                if response.status_code == 200:
                    logger.info("Got screen size successfully")
                    return response.json()
                else:
                    logger.error("Failed to get screen size. Status code: %d", response.status_code)
                    logger.info("Retrying to get screen size.")
            except Exception as e:
                logger.error("An error occurred while trying to get the screen size: %s", e)
                logger.info("Retrying to get screen size.")
            time.sleep(self.retry_interval)

        logger.error("Failed to get screen size.")
        return None

    def get_vm_window_size(self, app_class_name: str):
        """
        Gets the size of the vm app window.
        """

        for _ in range(self.retry_times):
            try:
                response = requests.post(self.http_server + "/window_size", data={"app_class_name": app_class_name})
                if response.status_code == 200:
                    logger.info("Got window size successfully")
                    return response.json()
                else:
                    logger.error("Failed to get window size. Status code: %d", response.status_code)
                    logger.info("Retrying to get window size.")
            except Exception as e:
                logger.error("An error occurred while trying to get the window size: %s", e)
                logger.info("Retrying to get window size.")
            time.sleep(self.retry_interval)

        logger.error("Failed to get window size.")
        return None

    def get_vm_wallpaper(self):
        """
        Gets the wallpaper of the vm.
        """

        for _ in range(self.retry_times):
            try:
                response = requests.post(self.http_server + "/wallpaper")
                if response.status_code == 200:
                    logger.info("Got wallpaper successfully")
                    return response.content
                else:
                    logger.error("Failed to get wallpaper. Status code: %d", response.status_code)
                    logger.info("Retrying to get wallpaper.")
            except Exception as e:
                logger.error("An error occurred while trying to get the wallpaper: %s", e)
                logger.info("Retrying to get wallpaper.")
            time.sleep(self.retry_interval)

        logger.error("Failed to get wallpaper.")
        return None

    def get_vm_desktop_path(self) -> Optional[str]:
        """
        Gets the desktop path of the vm.
        """

        for _ in range(self.retry_times):
            try:
                response = requests.post(self.http_server + "/desktop_path")
                if response.status_code == 200:
                    logger.info("Got desktop path successfully")
                    return response.json()["desktop_path"]
                else:
                    logger.error("Failed to get desktop path. Status code: %d", response.status_code)
                    logger.info("Retrying to get desktop path.")
            except Exception as e:
                logger.error("An error occurred while trying to get the desktop path: %s", e)
                logger.info("Retrying to get desktop path.")
            time.sleep(self.retry_interval)

        logger.error("Failed to get desktop path.")
        return None

    def get_vm_directory_tree(self, path) -> Optional[Dict[str, Any]]:
        """
        Gets the directory tree of the vm.
        """
        payload = json.dumps({"path": path})

        for _ in range(self.retry_times):
            try:
                response = requests.post(self.http_server + "/list_directory", headers={'Content-Type': 'application/json'}, data=payload)
                if response.status_code == 200:
                    logger.info("Got directory tree successfully")
                    return response.json()["directory_tree"]
                else:
                    logger.error("Failed to get directory tree. Status code: %d", response.status_code)
                    logger.info("Retrying to get directory tree.")
            except Exception as e:
                logger.error("An error occurred while trying to get directory tree: %s", e)
                logger.info("Retrying to get directory tree.")
            time.sleep(self.retry_interval)

        logger.error("Failed to get directory tree.")
        return None
