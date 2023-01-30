# flake8: noqa
PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: "you should always think about what to do"
Action: the action to take, should be one of [{tool_names}]
Action Input: "the input to the action"
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Action: "Final Answer"
Action Input: "the final answer to the question"
""".strip()
SUFFIX = """Begin!

Question: {input}
{agent_scratchpad}"""
