from typing import Any, Callable, Optional
from desktop_env.desktop_env import DesktopEnv

from .autogen.llm_config import LLMConfig
from .autogen.code_utils import PYTHON_VARIANTS
from .autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from .utils import env_step

CODER_SYSTEM_MESSAGE = """# Your role
You are the **lead programmer**. Solve the user's task step by step with a terminal (supports Python and Bash).
The terminal streams real-time execution output when you submit a code.
Your username is `user`; the sudo password is `{CLIENT_PASSWORD}`.
You should reply **only** with "TERMINATE" after the task is completed.

# Coding format
Submit **one** fenced code block **only**, labeled with its language:

```bash
# Your Bash script here
# To use sudo, follow this pattern:
echo {CLIENT_PASSWORD} | sudo -S <your commands>
```

**or**

```python
# Your Python code here
# Do not use: if __name__ == "__main__": (it will suppress output)
```

# Requirements
- **Code fence language:** Every fenced block must specify the language (`bash` or `python`); otherwise you will receive `unknown language unknown`.
- **Single block:** Wrap all code in **one** code block. Do not split your submission across multiple blocks.
- **Spreadsheets:** When editing spreadsheets, ensure **every value** is updated to the **intended cell** and **preserve the original formatting** (fonts, colors, sizes, etc.).
- **Dependencies:** Before importing or using a package, **check whether it is installed**; if not, **install it** in your submission.
- **Observability:** Print intermediate results to aid debugging, for example, the value you are modifying.
- **Final review:** Before completion, carefully inspect your result by writing test cases and confirm that **nothing outside the user's instructions has changed**.
"""


CONVERSATION_REVIEW_PROMPT = """# Programmer↔Terminal Log Summarizer (No Timeline/Env/Next Actions)

**Role:** Summarize Programmer↔Terminal logs for the **Orchestrator** so they can decide the next step immediately.  
**Orchestrator's task:** `{task}`
**Execution history:** `{chat_history}` (prompts + outputs).

## Output

**1) Summary (2-4 lines)** — task, what was tried, current status, why (cite key log lines / exit codes).

**2) Commands (deduped)**
```bash
# unique commands in run order; annotate repeats (xN)
```

**3) Terminal excerpts**
```text
# minimal evidence: head(~10) … [truncated N lines] … tail(~10)
# always include full error traces and return codes
```

**4) Artifacts / Side effects** — files/dirs changed (paths + purpose); installs/migrations.  
*Spreadsheet:* list cells/ranges edited and confirm formatting preserved.

**5) Errors / Blockers** — precise messages + exit codes; likely root cause from logs (no speculation).

**6) Verification** — what checks passed (tests, file existence, row counts); what still needs verification (e.g., reopen file and confirm cell Y).

## Rules
- **Evidence-first**, no speculation.  
- **Deterministic truncation** (head/tail; note omitted lines); always include error stacks.  
- **Call out deltas** (what changed vs intended).  
- Keep it tight: bullets > prose.
"""


class TerminalProxyAgent(MultimodalConversableAgent):
    def __init__(
            self, 
        name: str, 
        env: DesktopEnv,
        llm_config: LLMConfig = False, 
        system_message: str = "",
        human_input_mode: str = "NEVER",
        code_execution_config = {},
        is_termination_msg: Optional[Callable[[dict[str, Any]], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        default_auto_reply: Optional[str] = None,
        description: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply,
            description=description
        )
        self.env = env

    def run_code(self, code: str, lang: str = "python", **kwargs):
        output_dict = env_step(self.env, code, "coding", lang=lang, **kwargs)
        exitcode = output_dict.get("exitcode", -1)
        logs = output_dict.get("logs", "")
        image = output_dict.get("image", None)
        
        return exitcode, logs, image
