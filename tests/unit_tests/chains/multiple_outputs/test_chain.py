"""Test that GetMultipleOutputsChain can run successfully."""

from langchain.chains.multiple_outputs.base import GetMultipleOutputsChain
from tests.unit_tests.llms.fake_llm import FakeLLM
from tests.unit_tests.prompts.fake_parser import FakeDictParser


def test_multiple_outputs_can_run() -> None:
    """Test that GetMultipleOutputsChain can run successfully with multiple steps."""
    chain = GetMultipleOutputsChain(
        llm=FakeLLM(),
        prefix="Figure out what to do next.\n\n",
        variables={"Action": "tool", "Action Input": "tool_input"},
    )
    assert chain({}) == {
        "tool": "bar",
        "tool_input": "bar",
    }


def test_multiple_outputs_can_run_in_one_step() -> None:
    """Test that GetMultipleOutputsChain can run successfully in a single step."""
    chain = GetMultipleOutputsChain(
        llm=FakeLLM(),
        prefix="Figure out what to do next.\n\n",
        variables={"Action": "tool", "Action Input": "tool_input"},
        one_step=True,
        output_parser=FakeDictParser(),
    )
    assert chain({}) == {
        "tool": "Fake tool",
        "tool_input": "Fake input",
    }
