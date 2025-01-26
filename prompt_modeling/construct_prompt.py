import inspect
pg_task_prompt = (
        "Paraphrase generation task is about generating a paraphrase of the text from the source. "
        "In the example shown below, The source corresponds to the text that needs to be paraphrased; "
        "The target is the correct paraphrase for the source; "
        "the hypothesis is the predicated paraphrase from the model."
    )

example_prompt_no_src = (
    "Example:"
    "target: !tgt!"
    "hypothesis: !hyp!"
    )


hallucination_task_prompt = (
        "Your task is to answer whether the hypothesis from the example contains any hallucination "
        "(e.g., incorrect semantic information unsupported or inconsistent with the source) and explain why. "
        "The target is inferred from source without any hallucination. "
        "You should consider both source and target before making the judgement on the hypothesis. "
    )

general_task_prompt = (
    "Your task is to determine whether the hypothesis contains any hallucinations based on the target. "
    "(e.g., incorrect semantic information) and explain why. "
    "Only consider target and hypothesis when making the judgement. "
    "Your answer must start with 'Yes' or 'No'. "
)


pg_prompt = (
    "Your task is to determine whether the hypothesis is a paraphrase of the source and explain why "
    "Your answer must start with 'Yes' or 'No'. "
)

general_pipeline_prompt = general_task_prompt + example_prompt_no_src


pg_intermediate_prompt = pg_task_prompt + example_prompt_no_src