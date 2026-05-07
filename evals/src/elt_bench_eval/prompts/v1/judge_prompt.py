CRITERION_JUDGE_PROMPT = """You are an expert evaluator for AI-generated responses in second language (L2) education.
Your job is to judge whether an AI response meets a single scoring criterion.

## Original Task
{task_text}

## Reference Answer
{reference_answer}

## AI Response
{ai_response}

## Criterion
- Description: {criterion_description}

## Instructions
Determine whether the AI response meets the criterion above.
Use the reference answer as a guide for what a high-quality response looks like, but note that alternative valid approaches may exist.


- 'true' if the criterion IS met.
- 'false' if the criterion IS NOT met.

You MUST output either 'true' or 'false' and **nothing else**. Otherwise your answer will NOT be parsed correctly.
"""