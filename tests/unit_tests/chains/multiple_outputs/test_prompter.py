"""Test step-by-step prompting for GetMultipleOutputsChain."""

from langchain.chains.multiple_outputs import MultipleOutputsPrompter, VariableConfig
from tests.unit_tests.prompts.fake_parser import FakeDictParser


def test_prompter_with_defaults() -> None:
    """Test templating with the simplest defaults."""
    prompter = MultipleOutputsPrompter(
        prefix="Figure out what to do next.\n\n",
        variables={
            "Action": "tool",
            "Action Input": "tool_input",
        },
    )

    first_prompt_template = prompter.prompt_template_for_variable_at(0)
    assert first_prompt_template.output_parser is None
    first_prompt = first_prompt_template.format()
    assert (
        first_prompt
        == """
Figure out what to do next.

Action: "
    """.strip()
    )

    second_prompt_template = prompter.prompt_template_for_variable_at(1)
    assert second_prompt_template.output_parser is None
    second_prompt = second_prompt_template.format(tool="Search")
    assert (
        second_prompt
        == """
Figure out what to do next.

Action: "Search"
Action Input: "
    """.strip()
    )


def test_prompter_without_auto_suffixing() -> None:
    """Test templating without automatic suffixes applied."""
    prompter = MultipleOutputsPrompter(
        prefix="Figure out what to do next.\n\n",
        variables={
            'Action: "': "tool",
            'Action Input: "': "tool_input",
        },
        auto_suffix_variable_display=False,
        # also test that output parser doesn't leak into templates by default
        output_parser=FakeDictParser(),
    )

    first_prompt_template = prompter.prompt_template_for_variable_at(0)
    assert first_prompt_template.output_parser is None
    first_prompt = first_prompt_template.format()
    assert (
        first_prompt
        == """
Figure out what to do next.

Action: "
    """.strip()
    )

    second_prompt_template = prompter.prompt_template_for_variable_at(1)
    assert second_prompt_template.output_parser is None
    second_prompt = second_prompt_template.format(tool="Search")
    assert (
        second_prompt
        == """
Figure out what to do next.

Action: "Search"
Action Input: "
    """.strip()
    )


def test_prompter_with_specified_variables() -> None:
    """Test templating when variables are specified instead of a dict."""
    prompter = MultipleOutputsPrompter(
        prefix="Write some code.\n\n",
        variable_configs=[
            VariableConfig(display="File Path", output_key="path"),
            VariableConfig(display="Code", output_key="code", stop="```"),
            VariableConfig(display="Tests", output_key="test_code", stop="```"),
        ],
        # also test that output parser doesn't leak into templates by default
        output_parser=FakeDictParser(),
    )

    first_prompt_template = prompter.prompt_template_for_variable_at(0)
    assert first_prompt_template.output_parser is None
    first_prompt = first_prompt_template.format()
    assert (
        first_prompt
        == """
Write some code.

File Path: "
    """.strip()
    )

    second_prompt_template = prompter.prompt_template_for_variable_at(1)
    assert second_prompt_template.output_parser is None
    second_prompt = second_prompt_template.format(path="my_module/functionality.py")
    assert (
        second_prompt
        == """
Write some code.

File Path: "my_module/functionality.py"
Code: ```
""".strip()
    )

    third_prompt_template = prompter.prompt_template_for_variable_at(2)
    third_prompt = third_prompt_template.format(
        path="my_module/functionality.py",
        code="""\ndef do_something():\n    print("Hello world!")\n""",
    )
    assert (
        third_prompt
        == """
Write some code.

File Path: "my_module/functionality.py"
Code: ```
def do_something():
    print("Hello world!")
```
Tests: ```
""".strip()
    )


def test_prompter_with_fully_specified_variables() -> None:
    """Test templating when all variables are fully specified.

    Now we want to start the code blocks with ```python, but end them with ``` still.
    """
    prompter = MultipleOutputsPrompter(
        prefix="Write some code.\n\n",
        variable_configs=[
            VariableConfig(
                display="File Path",
                display_suffix=': "',
                output_key="path",
                stop='"',
            ),
            VariableConfig(
                display="Code",
                display_suffix=": ```python\n",
                output_key="code",
                stop="```",
            ),
            VariableConfig(
                display="Tests",
                display_suffix=": ```python\n",
                output_key="test_code",
                stop="```",
            ),
        ],
        auto_suffix_variable_display=False,
        # also test that output parser doesn't leak into templates by default
        output_parser=FakeDictParser(),
    )

    first_prompt_template = prompter.prompt_template_for_variable_at(0)
    assert first_prompt_template.output_parser is None
    first_prompt = first_prompt_template.format()
    assert (
        first_prompt
        == """
Write some code.

File Path: "
    """.strip()
    )

    second_prompt_template = prompter.prompt_template_for_variable_at(1)
    assert second_prompt_template.output_parser is None
    second_prompt = second_prompt_template.format(path="my_module/functionality.py")
    assert (
        second_prompt
        == """
Write some code.

File Path: "my_module/functionality.py"
Code: ```python
""".lstrip()
    )

    third_prompt_template = prompter.prompt_template_for_variable_at(2)
    third_prompt = third_prompt_template.format(
        path="my_module/functionality.py",
        code="""def do_something():\n    print("Hello world!")\n""",
    )
    assert (
        third_prompt
        == """
Write some code.

File Path: "my_module/functionality.py"
Code: ```python
def do_something():
    print("Hello world!")
```
Tests: ```python
""".lstrip()
    )


def test_prompter_full_input() -> None:
    """Test prompt templating when asking for all outputs at once.

    The prompt should be exactly the same as the prompt for the first step of the
    sequential chain. The only differences should exist in the presence of an
    output_parser, and in the lack of a stop (because now the stop would be for the
    whole chain, not just per-variable).
    """
    prompter = MultipleOutputsPrompter(
        prefix="Figure out what to do next.\n\n",
        variables={
            "Action": "tool",
            "Action Input": "tool_input",
        },
        output_parser=FakeDictParser(),
    )

    full_input_template = prompter.prompt_template_for_full_input()
    assert full_input_template.output_parser is prompter.output_parser
    full_input_prompt = full_input_template.format()
    assert (
        full_input_prompt
        == """
Figure out what to do next.

Action: "
    """.strip()
    )


def test_prompter_log() -> None:
    """Test reconstruction of the full LLM output from this interaction."""
    prompter = MultipleOutputsPrompter(
        prefix="Figure out what to do next.\n\n",
        variables={
            "Action": "tool",
            "Action Input": "tool_input",
        },
    )

    assert (
        prompter.log(
            {
                "tool": "Search",
                "tool_input": "Olivia Wilde boyfriend",
            }
        )
        == """
Search"
Action Input: "Olivia Wilde boyfriend"
    """.strip()
    )
