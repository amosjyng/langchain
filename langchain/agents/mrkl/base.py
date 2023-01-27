"""Attempt to implement MRKL systems as described in arxiv.org/pdf/2205.00445.pdf."""
from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from langchain.agents.agent import Agent, AgentExecutor
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from langchain.agents.tools import Tool
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.chains.multiple_outputs import GetMultipleOutputsChain
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.prompts.base import DictOutputParser
from langchain.schema import AgentAction

FINAL_ANSWER_ACTION = 'Action: "Final Answer"'


class ChainConfig(NamedTuple):
    """Configuration for chain to use in MRKL system.

    Args:
        action_name: Name of the action.
        action: Action function to call.
        action_description: Description of the action.
    """

    action_name: str
    action: Callable
    action_description: str


def get_action_and_input(llm_output: str) -> Tuple[str, str]:
    """Parse out the action and input from the LLM output.

    Note: if you're specifying a custom prompt for the ZeroShotAgent,
    you will need to ensure that it meets the following Regex requirements.
    The string starting with "Action:" and the following string starting
    with "Action Input:" should be separated by a newline.
    """
    regex = r"(?s)Action: \"?(.*?)\"?\nAction Input:\s*\"?(.*)\"?"
    match = re.search(regex, llm_output)
    if not match:
        raise ValueError(f"Could not parse LLM output: `{llm_output}`")
    action = match.group(1).strip()
    action_input = match.group(2)
    return action, action_input.strip().strip('"')


class OneStepParser(DictOutputParser):
    def parse(self, text: str) -> Dict[str, str]:
        action, input = get_action_and_input(text)
        regex = r"^(.*?)\"?\nAction:"
        match = re.search(regex, text)
        if match:
            thought = match.group(1).strip()
        else:
            thought = ""
        result= {
            "action": action,
            "input": input,
            "thought": thought,
        }
        return result


class ZeroShotAgent(Agent):
    """Agent for the MRKL chain."""

    one_step_inputs: bool = False

    @property
    def _agent_type(self) -> str:
        """Return Identifier of agent type."""
        return "zero-shot-react-description"

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return ""

    @classmethod
    def create_prompt(
        cls,
        tools: List[Tool],
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        input_variables: Optional[List[str]] = None,
    ) -> PromptTemplate:
        """Create prompt in the style of the zero shot agent.

        Args:
            tools: List of tools the agent will have access to, used to format the
                prompt.
            prefix: String to put before the list of tools.
            suffix: String to put after the list of tools.
            input_variables: List of input variables the final prompt will expect.

        Returns:
            A PromptTemplate with the template assembled from the pieces here.
        """
        tools = tools + [
            Tool(
                name="Final Answer",
                func=lambda _: "This is only a dummy tool",
                description=(
                    "Useful for when you have figured out the final answer. The input "
                    "should be the answer, phrased as a sentence, in string form."
                ),
            )
        ]
        tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        tool_names = ", ".join(["\"" + tool.name + "\"" for tool in tools])
        format_instructions = FORMAT_INSTRUCTIONS.format(tool_names=tool_names)
        template = "\n\n".join([prefix, tool_strings, format_instructions, suffix])
        if input_variables is None:
            input_variables = ["input", "agent_scratchpad"]
        return PromptTemplate(template=template, input_variables=input_variables)

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLLM,
        tools: List[Tool],
        callback_manager: Optional[BaseCallbackManager] = None,
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        input_variables: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools."""
        cls._validate_tools(tools)
        prompt = cls.create_prompt(
            tools, prefix=prefix, suffix=suffix, input_variables=input_variables
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        return cls(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)

    @classmethod
    def _validate_tools(cls, tools: List[Tool]) -> None:
        for tool in tools:
            if tool.description is None:
                raise ValueError(
                    f"Got a tool {tool.name} without a description. For this agent, "
                    f"a description must always be provided."
                )

    def _extract_tool_and_input(self, text: str) -> Optional[Tuple[str, str]]:
        return get_action_and_input(text)

    def _get_next_action(self, full_inputs: Dict[str, str]) -> AgentAction:
        selected_inputs = {
            k: full_inputs[k] for k in self.llm_chain.prompt.input_variables
        }
        prefix = self.llm_chain.prompt.format(**selected_inputs)
        variables = OrderedDict(
            Thought="thought",
            Action="action"
        )
        variables["Action Input"] = "input"
        chain = GetMultipleOutputsChain(
            llm=self.llm_chain.llm,
            prefix=prefix,
            variables=variables,
            one_step=self.one_step_inputs,
            output_parser=OneStepParser(),
            callback_manager=self.llm_chain.callback_manager,
            verbose=self.llm_chain.verbose,
            stop="Observation:"
        )
        return self._result_to_action(chain)

    def _result_to_action(self, chain: GetMultipleOutputsChain) -> AgentAction:
        results = chain({})
        action = results["action"]
        input = results["input"]
        return AgentAction(tool=action, tool_input=input, log=chain.log())


class MRKLChain(AgentExecutor):
    """Chain that implements the MRKL system.

    Example:
        .. code-block:: python

            from langchain import OpenAI, MRKLChain
            from langchain.chains.mrkl.base import ChainConfig
            llm = OpenAI(temperature=0)
            prompt = PromptTemplate(...)
            chains = [...]
            mrkl = MRKLChain.from_chains(llm=llm, prompt=prompt)
    """

    @classmethod
    def from_chains(
        cls, llm: BaseLLM, chains: List[ChainConfig], **kwargs: Any
    ) -> AgentExecutor:
        """User friendly way to initialize the MRKL chain.

        This is intended to be an easy way to get up and running with the
        MRKL chain.

        Args:
            llm: The LLM to use as the agent LLM.
            chains: The chains the MRKL system has access to.
            **kwargs: parameters to be passed to initialization.

        Returns:
            An initialized MRKL chain.

        Example:
            .. code-block:: python

                from langchain import LLMMathChain, OpenAI, SerpAPIWrapper, MRKLChain
                from langchain.chains.mrkl.base import ChainConfig
                llm = OpenAI(temperature=0)
                search = SerpAPIWrapper()
                llm_math_chain = LLMMathChain(llm=llm)
                chains = [
                    ChainConfig(
                        action_name = "Search",
                        action=search.search,
                        action_description="useful for searching"
                    ),
                    ChainConfig(
                        action_name="Calculator",
                        action=llm_math_chain.run,
                        action_description="useful for doing math"
                    )
                ]
                mrkl = MRKLChain.from_chains(llm, chains)
        """
        tools = [
            Tool(name=c.action_name, func=c.action, description=c.action_description)
            for c in chains
        ]
        agent = ZeroShotAgent.from_llm_and_tools(llm, tools)
        return cls(agent=agent, tools=tools, **kwargs)
