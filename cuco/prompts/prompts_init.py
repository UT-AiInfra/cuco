INIT_SYSTEM_MSG = """You are an expert programmer.
Your goal is to write an initial program that solves a given task.
"""

INIT_USER_MSG = """Please write an initial program for the following task. The program should be written in {language}.

The task is as follows:
{task_description}

You MUST repond using a short summary name, description and the full code:

<NAME>
A shortened name summarizing the code you are proposing. Lowercase, no spaces, underscores allowed.
</NAME>

<DESCRIPTION>
A description of the initial code you are proposing.
</DESCRIPTION>

<CODE>
```{language}
# The new initial program here.
```
</CODE>
"""
