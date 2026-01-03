'''
ICP profile and offering

These variables are loaded directly into the LeadEvolverPipeline.
'''

offering = '''
What is the offering?
1. Generally, we provide a prompt optimization product that uses an LLM to write the optimal prompt for you, when your system outcome is a “ground truth” outcome. Additionally, we are helpful when:
When you have an AI workflow that needs high reliability
When you prefer to use a non-dspy framework for a given project
When you have ground truth outcomes such as a classification outcome variable, or a specific short answer that is verifiable with semantic matching. See ground truth below.
2. We also provide prompt optimization technology/development services for the following use cases which we are planning to package into a product in the future: 
When you do not have ground truth outcomes
When you need multiple outcome metrics
When you want to scale up your evaluation set efficiently with only a dozen manual rows. 
'''

icp_profile = '''
Who is the ICP for this project?
- Python developer who is building AI workflows (not agents) and who would benefit from self improving prompts - like dspy or other prompt optimizers. They may already be using dspy, but want to use a different framework for another project.
- AI developer in a company where the developer (or the developer’s boss) has decision making power over the technology to implement 
- Developer that is part of a small AI native company or in a AI team within a broader company. 
Not a fit:
- People who are part of a competitor team (e.g., dspy) 
- Companies that have long / multi-stakeholder enterprise sales processes (i.e.,  series C and Beyond)
- Companies that are big tech or big AI houses (Openai, Anthropic, Google, Amazon, Apple, etc.)
- People in 3rd world countries - (can be a weak match if they are particularly strong)
- Developers who do not work in python
Bonus: 
- People with youtube channels or other networks they control can be future partners

Ground Truth definition for outcome variables
A ground truth outcome variable is an outcome variable that has 1 verifiable right answer. Verification can be conducted with multiple techniques, including string matching or (preferably) semantic matching. Semantic matching is much more useful inbroader and uses language model embedding vectors to encode and compare the meaning of two different sets of words.
E.g., for the question, what is the capital of France? “Paris, France” and “Paris” would have roughly the same semantic meaning  using  semantic matching with embedding vector comparison,  but would be considered different answers with string matching
'''