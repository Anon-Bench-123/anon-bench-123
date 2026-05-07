CRITERION_JUDGE_PROMPT = """You are an expert evaluator for AI-generated responses in second language (L2) education.
Your job is to judge whether an AI response meets a single, specific scoring criterion.

## Original Task
{task_text}

## AI Response
{ai_response}

## Criterion
{criterion_description}

## Definitions

PASS (true): The AI response meets the core requirement of the specific criterion. It may not be 
perfect in all aspects of the task, but for this specific criterion, the required content, accuracy, 
or pedagogical skill is clearly demonstrated.

FAIL (false): The AI response fails to meet the core requirement of the criterion. This includes 
cases where the information is factually incorrect (especially pedagogical content), the required 
element is entirely missing, or the level of support is inappropriate for the target learners 
specified in the criterion.

## Examples

### Example 1: PASS
Task: Create teaching materials to present and practise modal verbs ('must', 'mustn't', 'have to', 'don't have to') for an A2 class.
AI Response: "...Remember: 'Mustn't' means something is forbidden (against the rules), like 'You mustn't smoke in the hospital.' On the other hand, 'Don't have to' means it is not necessary, but you can do it if you want, like 'You don't have to wear a suit to the party, but you can if you like.'..."
Criterion: The explanation of the difference in meaning between 'mustn't' and 'don't have to' is correct.
Critique: The AI response correctly identifies the semantic difference between the two modals. It 
explicitly states that 'mustn't' refers to prohibition ('forbidden') and 'don't have to' refers 
to a lack of necessity ('not necessary'). The examples provided ('smoking in hospital' vs 'wearing 
a suit') clearly illustrate this distinction for an A2 learner. The pedagogical explanation is 
accurate and meets the criterion.
Result: 
true

### Example 2: FAIL
Task: Create teaching materials to present and practise modal verbs ('must', 'mustn't', 'have to', 'don't have to') for an A2 class.
AI Response: "...Important rule: 'Mustn't' and 'Don't have to' both mean that you should not do something. For example, 'You mustn't run in the hall' and 'You don't have to run in the hall' are both rules for school safety..."
Criterion: The explanation of the difference in meaning between 'mustn't' and 'don't have to' is correct.
Critique: The AI response provides a factually incorrect explanation. It treats 'mustn't' and 
'don't have to' as synonymous ways to express prohibition. In reality, 'don't have to' expresses a 
lack of obligation, which is a critical distinction at the A1/A2 level. By equating them, the 
response creates significant learner confusion on the very point the criterion is evaluating. 
The pedagogical content is incorrect.
Result:
false

### Example 3: PASS (borderline)
Task: Create teaching materials to present and practise modal verbs ('must', 'mustn't', 'have to', 'don't have to') for an A2 class.
AI Response: "...Activity 1: Fill in the blanks with 'must' or 'have to'. Activity 2: Match the rules to the signs. *Teacher's Note: For students who finish early, ask them to write three rules for their own bedroom.*"
Criterion: The materials are suitable for a mixed-ability A2 class.
Critique: The materials are basic, and the primary activities don't explicitly differentiate 
between levels within the A2 range. However, the inclusion of a 'Teacher's Note' for fast finishers 
provides a simple form of differentiation (extension task), which addresses the 'mixed-ability' 
aspect of the criterion. While the scaffolding for lower-ability students could be more explicit, 
the presence of an extension task for stronger students means the criterion is met at a minimum 
acceptable level for this context.
Result: 
true

## Instructions

First, write a detailed critique that:
- Identifies whether the specific requirement of the criterion is present.
- Cites specific evidence from the AI response to support your judgement.

After your critique, output your final verdict on a new line.
Output ONLY 'true' if the criterion IS met, or ONLY 'false' if the criterion IS NOT met.
Do not output any other text after your verdict — it will NOT be parsed correctly.
"""