def build_prompt(inputs):
    return f"""
        You are a helpful assistant answering questions based ONLY on the provided context.

        If the answer is not in the context, say "I don't know".

        Chat History:
        {inputs.get("chat_history", "")}

        Context:
        {inputs["context"]}

        Question:
        {inputs["question"]}

        Answer:
    """