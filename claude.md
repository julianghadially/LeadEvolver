# Claude.md for LeadEvolver

LeadEvolver is a project that implements DSPy in Python to build a Lead research workflow for classifying lead quality and generating profiles. 

This project should be implemented according to the DSPy framework. We are building a "compound AI system" that involves multi hop web search, classification, and ideal customer profile (ICP) composition. We are using DSPY so that we can optimize our system for accuracy (and later for efficiency). 

## Key Resources
- **DSPy docs**: https://dspy.ai/
- **Lead Data**: data/github_users.csv

## Background Research on Research Agents
"Chain of RAG" allows a language model to dynamically reformulate the query based on an evolving state of retrieved inrofmation - that is, it performs iterative retrieval  (https://arxiv.org/pdf/2501.14342). Performing Chain of RAG with 6+ chained queries improves the exact match score by over 10 points in multi-hop benchmarks, with the best performance occurring from best-of-8 tree search.

FIRE Fact-Checking with Iterative Retrieval performs efficiently than other pre-planned fact-checking algorithms: https://aclanthology.org/2025.findings-naacl.158.pdf?utm_source=chatgpt.com

In the data science domain, a multi-agent system with a shared blackboard mechansims can outperform single-agent and multi-agent, master-slave architectures for finding answers to questions about data: https://ar5iv.labs.arxiv.org/html/2510.01285v1. Master Slave setups can struggle because the master agent is too rigid to adapt to incorrect queries that come back with bad information. We avoid master slave setup by allowing the researcher to perform its own research, and share findings with future researchers on a shared blackboard - although our blackboard mechanism is distinctly different.  

Generally, we use an iterative research agent with a structured blackboard and goal-driven research.

## DSPy patterns for all AI modules and Prompts
Prompt engineering is messy. It involves a lot of tweaking, trial and error, and iteration. Additionally, implementing DSPY allows us to optimize a compound AI system across multiple modules

Use DSPy for all AI modules. 

DSPy made up of signatures and modules. Instead of defining prompts, we will define signatures, which defined the inputs and outputs to a module in an AI system. 

Defining signatures as classes is recommended. 
For example:

```python
class WebQueryGenerator(dspy.Signature):
    """Generate a query for searching the web."""
    question: str = dspy.InputField()
    query: str = dspy.OutputField(desc="a query for searching the web")
```

Next, modules are used as nodes in the project, either as a single line:

```python
    predict = dspy.Predict(WebQueryGenerator) 
```

Or as a class:
```python
class WebQueryModule(dspy.Module):
    def __init__(self):
        super().__init__()

        self.query_generator = dspy.Predict(WebQueryGenerator)

    def forward(self, question: str):
        return self.query_generator(question = question)
```

## Naming conventions
- Use Pascal case for signatures and modules.
- Add Module at the end of all module classes

## Dataset Structure
- **Format**: csv with `Username`, `name`, `url`, `icp_match`, and `icp_match_rationale`
- **Location**: `data/github_users.csv`
- **Labels**: Strong match, Weak match, Not a fit

## Additional rules
- Never change the language model unless explicitly asked by user. GPT-5 and GPT-5-mini do in fact exist - This project began January 2026, and your knowledge training cutoff may be up to mid-2025!