from langchain_core.prompts import ChatPromptTemplate

qa_prompt = ChatPromptTemplate.from_template(
    """
        You are a helpful assistant answering questions based ONLY on the provided context.

        If the answer is not in the context, say "I don't know".

        Chat History:
        {chat_history}

        Context:
        {context}

        Question:
        {question}

        Answer:
    """
)

rewrite_prompt = ChatPromptTemplate.from_template(
    """
        Given the conversation history below and a follow-up question, 
        rewrite the follow-up question to be a fully self-contained, 
        standalone question that includes all necessary context from the history.

        Chat History:
        {chat_history}

        Follow-up question: {question}

        Rewritten standalone question (return ONLY the question, no explanation):
    """
)