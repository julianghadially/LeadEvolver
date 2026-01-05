
profile_rubric = """
Profile scoring 100 total points.

I. Accuracy 60%:
Did the profile accurately capture what we know about the user?
- Faithfulness (facts): When facts are presented, are those facts found in the research? 
- Reasonable (judgments): when judgments are made, are they A) represented as judgement (and not facts) and B) reasonable judgments based on the research?

II. Succinct – 10%:
- Too short <800
- Succinct = 800-2000 characters
- Long: 2000-3000 
- Very long: 3000

III. Relevant – 10%:
- Are the facts relevant to the offering and determining whether the lead is a fit with the ideal customer profile?

IV. Complete – 10%:
Profile contains past experiences, either their past employment, or thought leadership, or work posted in public portfolios like github

V. Contains contact details – 10%:
Maximum 10 points for this section
First, determine the highest tier of contact type:
- Tier 1: Email or phone: 5 points
- Tier 2: Linkedin, twitter, or instagram, or platform where DM is possible: 4 points
- Tier 3: personal website or other, where contact is difficult (e.g., unlikely response from contact form): 3 points
Next, add points for each contact type that is present:
- Contains email: +1 point
- Contains phone: +1 point 
- Contains twitter: +1 point
- Contains linkedin: + 1 point
- Contains personal website: +1 point

V. Persona / storytelling – 10%
- In the description section, does the profile characterize the persona immediately? 
- Does the "what they might need" section describe what the persona might need?

"""

profile_response_format = """
ACCURACY: [score]/60 - [brief rationale]
SUCCINCT: [score]/10 - [brief rationale]
RELEVANT: [score]/10 - [brief rationale]
COMPLETE: [score]/10 - [brief rationale]
CONTACT: [score]/10 - [brief rationale]
PERSONA: [score]/10 - [brief rationale]
TOTAL: [total]/100"""