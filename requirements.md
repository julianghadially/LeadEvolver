# LeadEvolver Bot Requirements (For Claude)
LeadEvolver is a project that implements DSPy in Python to build a sales lead research workflow for classifying lead quality and generating profiles. 

This project should be implemented according to the DSPy framework. We are building a "compound AI system" that involves multi hop web search, classification, and ideal customer profile (ICP) composition. We are using DSPY so that we can optimize our system for accuracy (and later for efficiency). 

## Key Objectives: 
1. We seek a GitHub lead research workflow that can be used in sales pipelines
2. We seek to run experiments on the effectiveness of prompt optimization in the settings marked under experiments

## Experiments
1. **Scaling up human-scored labels for a ground-truth judge:** How well does a ground truth LLM judge perform for tuning a web research agent + classification system, when human-scored labels are limited? Use a judge with 10 examples, including input (context) and output (ground_truth and rationale).
2. **Performance of a non-ground truth judge with a scoring rubric:** How well does a non-ground truth LLM judge perform for tuning a profile drafting system, when armed with a scoring rubric? 
3. **Efficiency training and cost penalties:** How well can the system teach itself efficiency, and what cost penalty should be used?
4. **Context embellishment on human-scored labels:**  When creating human labeled data for a research- or rag-supported task, it is generally cumbersome to aggregate the correct context alongside ground truth labels. The context is often long, messy, and unclear, and if a human prepares it, it can often simplify the context or inject judgements into the context. How well can the system teach its own context so that a user can focus on providing ground truth and rationale, without detailed context?

## Assessment
**Assessment for Experiment 1:** We will run the research + classification system on the leads in github_users.csv. We will create a language model judge that is armed with 10+ example classifications, including context, ground truth, and rationale. For training set examples, we will compare the system output to LLM-judged classifications. Meanwhile, test set examples will be compared to human-reviewed examples. See `training_set` and `judge_example` column in github_users.csv. 

## System Part 1
- **Lead Classifier:** LeadClassifier classifies leads as strong fit, weak fit, or not a fit, and uses iterative investigation to figure out if more research is needed. (see src/LeadEvolver/signatures/LeadClassifier.py)
- **Researcher:** Outputs a list of PageFindings objects that answer the research goal. Each PageFindings object contains the findings from the page and interesting links that could be explored further. (see src/LeadEvolver/signatures/Researcher.py). The researcher should use a dspy react module that can iterate over multiple search and scrape tool calls. It should be iterative through recursion on through tool calls. Use a max character limit of 10k per page and max 20 pges so we dont over crowd context. Save interesting links to the blackboard in the PageFindings object.
- **Blackboard:** The blackboard contains past research findings. All PageFindings objects are saved to the blackboard (see src/data_schema/page_finding.py for the PageFindings class)

## System Part 2
The second part of the system will research leads further and use a LeadProfiler module that composes an ICP sales profile for the lead