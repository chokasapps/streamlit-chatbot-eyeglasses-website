from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import MarkdownHeaderTextSplitter, CharacterTextSplitter
from langchain.schema.document import Document
import streamlit as st
from langchain.agents.agent_toolkits import create_retriever_tool
import os

DOCS_FOLDER = "uploads/crawled"


class Wrapper:
    def __init__(self, wrapped_class):
        self.wrapped_class = wrapped_class

    def __getattr__(self, attr):
        original_func = getattr(self.wrapped_class, attr)

        def wrapper(*args, **kwargs):
            # print(f"Calling function: {attr}")
            # print(f"Arguments: {kwargs}")
            result = original_func(*args, **kwargs)
            return result

        return wrapper


st.set_page_config(page_title="Conversational Chatbot for Eyeglasses.com")

openai_api_key = st.sidebar.text_input(
    "OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY")
)


@st.cache_resource  #
def get_executer(_memory):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=openai_api_key,
        streaming=True,
        max_tokens=300,
    )
    llm.client = Wrapper(llm.client)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    text_loader_kwargs = {"autodetect_encoding": True}
    loader = DirectoryLoader(
        DOCS_FOLDER,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
        loader_kwargs=text_loader_kwargs,
    )
    docs = loader.load()
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, chunk_overlap=10
    )
    split_docs = []
    for doc in docs:
        texts = text_splitter.split_text(doc.page_content)
        for md_header_split in texts:
            split_docs.append(
                Document(
                    page_content=md_header_split,
                    metadata={"source": "local"},
                )
            )
    vectorstore = FAISS.from_documents(documents=split_docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "search_state_of_union",
        "Searches and returns documents regarding the state-of-the-union.",
    )
    tools = [retriever_tool]
    chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
    executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        memory=_memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
    return executor


msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs,
    return_messages=True,
    memory_key="chat_history",
    output_key="output",
)
if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")
    st.session_state.steps = {}

avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.status(
                f"**{step[0].tool}**: {step[0].tool_input}", state="complete"
            ):
                st.write(step[0].log)
                st.write(step[1])
        st.write(msg.content)

if prompt := st.chat_input(placeholder=""):
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    executor = get_executer(memory)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = executor(prompt, callbacks=[st_cb])
        st.write(response["output"])
        st.session_state.steps[str(len(msgs.messages) - 1)] = response[
            "intermediate_steps"
        ]
