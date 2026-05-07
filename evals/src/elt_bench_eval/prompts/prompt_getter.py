from elt_bench_eval.prompts.retry import CRITERION_JUDGE_RETRY_SUFFIX


def get_judge_prompt(version: str) -> str:
    """Load the judge prompt for a specific version.
    
    Parameters
    ----------
    version : str
        The version of the prompt to load (e.g. 'v1').
        
    Returns
    -------
    str
        The prompt text.
        
    Raises
    ------
    ValueError
        If the version cannot be loaded or the attribute is missing.
    """
    import importlib
    
    try:
        module_path = f"elt_bench_eval.prompts.{version}.judge_prompt"
        module = importlib.import_module(module_path)
        return getattr(module, "CRITERION_JUDGE_PROMPT")
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Failed to fetch prompt for version '{version}': {e}")

def get_retry_prompt()->str:
    return CRITERION_JUDGE_RETRY_SUFFIX