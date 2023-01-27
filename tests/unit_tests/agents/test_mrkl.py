"""Test MRKL functionality."""

from typing import List, Optional

import pytest

from langchain.agents.mrkl.base import (
    FINAL_ANSWER_ACTION,
    MRKLChain,
    ZeroShotAgent,
    get_action_and_input,
)
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from langchain.agents.tools import Tool
from langchain.prompts import PromptTemplate
from tests.unit_tests.llms.fake_llm import FakeLLM


class MultipleAnswersAtOnceFakeLLM(FakeLLM):
    n: int = 0
    finish_after: int = 3

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        self.n += 1
        if self.n >= self.finish_after:
            return (
                'Thought: "Answer"\n'
                'Action: "Final Answer"\n'
                'Action Input: "News that is not real"'
            )
        return 'Thought: "bar"\nAction: "bar"\nAction Input: "bar"'


class OneAnswerAtATimeFakeLLM(FakeLLM):
    n: int = 0
    finish_after: int = 7

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        self.n += 1
        if self.n >= self.finish_after:
            return "Final Answer"
        return "bar"


def test_get_action_and_input() -> None:
    """Test getting an action from text."""
    llm_output = (
        'Thought: "I need to search for NBA"\n'
        'Action: "Search"\n'
        'Action Input: "NBA"'
    )
    action, action_input = get_action_and_input(llm_output)
    assert action == "Search"
    assert action_input == "NBA"


def test_get_action_and_input_whitespace() -> None:
    """Test getting an action from text."""
    llm_output = (
        'Thought: "I need to search for NBA"\n'
        'Action: "Search "\n'
        'Action Input: "NBA"'
    )
    action, action_input = get_action_and_input(llm_output)
    assert action == "Search"
    assert action_input == "NBA"


def test_get_final_answer() -> None:
    """Test getting final answer."""
    llm_output = (
        'Thought: "I can now answer the question"\n'
        'Action: "Final Answer"\n'
        'Action Input: "1994"'
    )
    action, action_input = get_action_and_input(llm_output)
    assert action == "Final Answer"
    assert action_input == "1994"


def test_get_final_answer_new_line() -> None:
    """Test getting final answer."""
    llm_output = (
        'Observation: "founded in 1994"\n'
        'Thought: "I can now answer the question"\n'
        'Action: "Final Answer"\n'
        'Action Input:\n"1994"'
    )
    action, action_input = get_action_and_input(llm_output)
    assert action == "Final Answer"
    assert action_input == "1994"


def test_get_final_answer_multiline() -> None:
    """Test getting final answer that is multiline."""
    llm_output = (
        'Thought: "I can now answer the question"\n'
        'Action: "Final Answer"\n'
        'Action Input: "1994\n1993"'
    )
    action, action_input = get_action_and_input(llm_output)
    assert action == "Final Answer"
    assert action_input == "1994\n1993"


def test_bad_action_input_line() -> None:
    """Test handling when no action input found."""
    llm_output = (
        'Thought: "I need to search for NBA"\n' 'Action: "Search"\n' "Thought: NBA"
    )
    with pytest.raises(ValueError):
        get_action_and_input(llm_output)


def test_bad_action_line() -> None:
    """Test handling when no action input found."""
    llm_output = (
        "Thought: I need to search for NBA\n" "Thought: Search\n" "Action Input: NBA"
    )
    with pytest.raises(ValueError):
        get_action_and_input(llm_output)


def test_from_chains() -> None:
    """Test initializing from chains."""
    chain_configs = [
        Tool(name="foo", func=lambda x: "foo", description="foobar1"),
        Tool(name="bar", func=lambda x: "bar", description="foobar2"),
    ]
    agent = ZeroShotAgent.from_llm_and_tools(FakeLLM(), chain_configs)
    # imo it's good to have an explicit representation of the result we want, so that
    # there are no hidden surprises in the code that constructs the expected output
    expected_template = """
Answer the following questions as best you can. You have access to the following tools:

foo: foobar1
bar: foobar2
Final Answer: Useful for when you have figured out the final answer. The input should be the answer, phrased as a sentence, in string form.

Use the following format:

Question: the input question you must answer
Thought: "you should always think about what to do"
Action: the action to take, should be one of ["foo", "bar", "Final Answer"]
Action Input: "the input to the action"
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Action: "Final Answer"
Action Input: "the final answer to the question"

Begin!

Question: {input}
{agent_scratchpad}
""".strip()
    prompt = agent.llm_chain.prompt
    print(prompt.template)
    assert isinstance(prompt, PromptTemplate)
    assert prompt.template == expected_template


def test_run_mrkl_one_step() -> None:
    fake_llm = MultipleAnswersAtOnceFakeLLM()
    fake_tools = [
        Tool(name="bar", func=lambda x: "bar", description="foobar2"),
    ]
    agent = ZeroShotAgent.from_llm_and_tools(llm=fake_llm, tools=fake_tools, one_step_inputs=True)
    executor = MRKLChain.from_agent_and_tools(
        agent=agent,
        tools=fake_tools,
        max_iterations=5,
        verbose=True,
    )
    assert executor.run("What is fake news?").strip() == "News that is not real"


def test_run_mrkl_multiple_steps() -> None:
    fake_llm = OneAnswerAtATimeFakeLLM()
    fake_tools = [
        Tool(name="bar", func=lambda x: "bar", description="foobar2"),
    ]
    agent = ZeroShotAgent.from_llm_and_tools(llm=fake_llm, tools=fake_tools)
    executor = MRKLChain.from_agent_and_tools(
        agent=agent,
        tools=fake_tools,
        max_iterations=5,
        verbose=True,
        one_step_inputs=False,
    )
    assert executor.run("What is fake news?").strip() == "Final Answer"
