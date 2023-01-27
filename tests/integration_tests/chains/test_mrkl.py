"""Integration test for ZeroShotAgent and SerpAPI"""
from langchain.agents import load_tools
from langchain.agents.mrkl.base import MRKLChain, ZeroShotAgent
from langchain.llms.openai import OpenAI


def test_mrkl() -> None:
    """Test functionality on a prompt."""
    llm = OpenAI(temperature=0, model_name="text-davinci-002")
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    agent = ZeroShotAgent.from_llm_and_tools(llm=llm, tools=tools, one_step_inputs=False)
    executor = MRKLChain.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
    assert (
        "raised to the 0.23 power is 2."
        in executor.run(
            "Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 "
            "power?"
        ).strip()
    )
