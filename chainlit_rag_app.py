import chainlit as cl
from src.rags.reliable_rag_w_convo_ret_1_qa_few_shot import (
    ReliableRAGWithConvoRetAndQueryAtomizationAndFewShot,
)

# Path to your PDF directory and vectorstore
PDF_DIR = "data/pdfs"


@cl.on_chat_start
async def on_chat_start():
    # Initialize RAG pipeline
    rag = ReliableRAGWithConvoRetAndQueryAtomizationAndFewShot()
    # Try to load existing vectorstore, else build from scratch
    try:
        rag.load_existing_vectorstore()
    except Exception:
        rag.load_documents(PDF_DIR)
        rag.process_documents()
        rag.build_index()
    # Store RAG instance in user session
    cl.user_session.set("rag", rag)


@cl.on_message
async def on_message(message: cl.Message):
    rag = cl.user_session.get("rag")
    if rag is None:
        await cl.Message(
            content="RAG system not initialized. Please restart the chat."
        ).send()
        return

    # Create message for streaming
    msg = cl.Message(content="")
    await msg.send()

    # Stream the response using async method for improved performance
    try:
        async for token in rag.run_complete_pipeline_streaming_async(message.content):
            await msg.stream_token(token)
        await msg.update()
    except Exception as e:
        await msg.stream_token(f"[Error: {str(e)}]")
        await msg.update()
