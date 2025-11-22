# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import copy
from typing import Any, Optional, Union

from ... import OpenAIWrapper
from ...code_utils import content_str
from .. import Agent, ConversableAgent
from ..contrib.img_utils import (
    gpt4v_formatter,
    message_formatter_pil_to_b64,
)

DEFAULT_MODEL = "gpt-4-vision-preview"


class MultimodalConversableAgent(ConversableAgent):
    DEFAULT_CONFIG = {
        "model": DEFAULT_MODEL,
    }

    def __init__(
        self,
        name: str,
        system_message: Optional[Union[str, list]] = None,
        is_termination_msg: str = None,
        mimic_human_return: bool = True,
        *args,
        **kwargs: Any,
    ):
        """Args:
        name (str): agent name.
        system_message (str): system message for the OpenAIWrapper inference.
            Please override this attribute if you want to reprogram the agent.
        **kwargs (dict): Please refer to other kwargs in
            [ConversableAgent](/docs/api-reference/autogen/ConversableAgent#conversableagent).
        """
        super().__init__(
            name,
            system_message,
            is_termination_msg=is_termination_msg,
            *args,
            **kwargs,
        )
        self.mimic_human_return = mimic_human_return
        self._is_termination_msg = (
            is_termination_msg
            if is_termination_msg is not None
            else (lambda x: content_str(x.get("content")) == "TERMINATE")
        )

        # Override the `generate_oai_reply`
        self.replace_reply_func(ConversableAgent.generate_oai_reply, MultimodalConversableAgent.generate_oai_reply)
        self.replace_reply_func(
            ConversableAgent.a_generate_oai_reply,
            MultimodalConversableAgent.a_generate_oai_reply,
        )

    @staticmethod
    def _message_to_dict(message: Union[dict[str, Any], list[str], str]) -> dict:
        """Convert a message to a dictionary. This implementation
        handles the GPT-4V formatting for easier prompts.

        The message can be a string, a dictionary, or a list of dictionaries:
            - If it's a string, it will be cast into a list and placed in the 'content' field.
            - If it's a list, it will be directly placed in the 'content' field.
            - If it's a dictionary, it is already in message dict format. The 'content' field of this dictionary
            will be processed using the gpt4v_formatter.
        """
        if isinstance(message, str):
            return {"content": gpt4v_formatter(message, img_format="pil")}
        if isinstance(message, list):
            return {"content": message}
        if isinstance(message, dict):
            assert "content" in message, "The message dict must have a `content` field"
            if isinstance(message["content"], str):
                message = copy.deepcopy(message)
                message["content"] = gpt4v_formatter(message["content"], img_format="pil")
            try:
                content_str(message["content"])
            except (TypeError, ValueError) as e:
                print("The `content` field should be compatible with the content_str function!")
                raise e
            return message
        raise ValueError(f"Unsupported message type: {type(message)}")

    def generate_oai_reply(
        self,
        messages: Optional[list[dict[str, Any]]] = None,
        sender: Optional[Agent] = None,
        config: Optional[OpenAIWrapper] = None,
    ) -> tuple[bool, Optional[Union[str, dict[str, Any]]]]:
        """Generate a reply using autogen.oai."""
        client = self.client if config is None else config
        if client is None:
            return False, None
        if messages is None:
            messages = self._oai_messages[sender]
        
        for message in messages:
            new_message = []
            if message['content'] is None:
                continue
            for msg in message['content']:
                if msg['type'] == 'text' and msg['text'] == "":
                    continue
                new_message.append(msg)
            message['content'] = new_message

        messages_with_b64_img = message_formatter_pil_to_b64(self._oai_system_message + messages)

        if self.mimic_human_return:
            new_messages = []
            for message in messages_with_b64_img:
                if 'tool_responses' in message:
                    for tool_response in message['tool_responses']:
                        tmp_image = None
                        tmp_list = []
                        for ctx in message['content']:
                            if ctx['type'] == 'image_url':
                                tmp_image = ctx
                        tmp_list.append({
                            'role': 'tool',
                            'tool_call_id': tool_response['tool_call_id'],
                            'content': [message['content'][0]]
                        })
                        if tmp_image:
                            tmp_list.append({
                                'role': 'user',
                                'content': [
                                    {'type': 'text', 'text': "I take a screenshot of the current OS state after GUI Operator completed its task. Please check the screenshot of the current OS state carefully and see if if fulfill the task requirements."},
                                    tmp_image
                                ]
                            })
                        new_messages.extend(tmp_list)
                else:
                    new_messages.append(message)
            messages_with_b64_img = new_messages.copy()


        # TODO: #1143 handle token limit exceeded error
        response = client.create(
            messages=messages_with_b64_img, agent=self.name
        )

        # TODO: line 301, line 271 is converting messages to dict. Can be removed after ChatCompletionMessage_to_dict is merged.
        extracted_response = client.extract_text_or_completion_object(response)[0]
        if not isinstance(extracted_response, str):
            extracted_response = extracted_response.model_dump()
        return True, extracted_response
