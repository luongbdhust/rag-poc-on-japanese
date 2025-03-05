import streamlit as st
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain.schema import HumanMessage

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
embeddings = PineconeEmbeddings(
    model=st.secrets["EMBEDDING_MODEL"],
    pinecone_api_key=st.secrets["PINECONE_API_KEY"]
)

docsearch = PineconeVectorStore.from_existing_index(
    index_name=st.secrets["PINECONE_INDEX_NAME"],
    embedding=embeddings,
    namespace=st.secrets["PINECONE_NAMESPACE"],
)
retriever = docsearch.as_retriever()


def create_llm(model_name):
    return ChatOpenAI(
        base_url=st.secrets["BASE_URL_API_KEY"],
        openai_api_key=st.secrets["OPENAI_API_KEY"],
        model_name=model_name,
        temperature=0.0,
        streaming=True
    )


def create_retrieval(llm):
    combine_docs_chain = create_stuff_documents_chain(
        llm, retrieval_qa_chat_prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)

# llm = ChatOpenAI(
#     base_url=st.secrets["BASE_URL_API_KEY"],
#     openai_api_key=st.secrets["OPENAI_API_KEY"],
#     model_name=st.secrets["LLM_MODEL"],
#     temperature=0.0,
#     streaming=True
# )

# combine_docs_chain = create_stuff_documents_chain(
#     llm, retrieval_qa_chat_prompt
# )
# retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)


if "selected_model" not in st.session_state:
    st.session_state.selected_model = "deepseek/deepseek-r1"
    st.session_state.llm = create_llm(st.session_state.selected_model)
    st.session_state.retrieval_chain = create_retrieval(st.session_state.llm)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(layout="wide")
st.header("PoC ứng dụng hỏi đáp trên tập dữ liệu riêng")
st.write("Ứng dụng sử dụng tài liệu [bài phân tích thị trường nhựa tái chế ở việt nam](https://raw.githubusercontent.com/luongbdhust/rag-poc-on-japanese/main/phan-tich-nhua-tai-che-vietnam-japanese.pdf) để làm kiến thức bổ sung cho các llm model khi thực hiện hỏi đáp. Tài liệu này là ngôn ngữ tiếng nhật(Được dịch từ [bản tiếng việt](https://raw.githubusercontent.com/luongbdhust/rag-poc-on-japanese/main/phan-tich-nhua-tai-che-vietnam-vietnamese.pdf) bằng google dịch). Bạn có thể click vào link để tải về xem.")

selected_model = st.selectbox(
    "Lựa chọn model bạn muốn sử dụng?",
    ("deepseek/deepseek-r1",
     "openai/gpt-4o-mini", "openai/o3-mini",),
    index=("deepseek/deepseek-r1", "openai/gpt-4o-mini",
           "openai/o3-mini").index(st.session_state.selected_model),
)

if selected_model != st.session_state.selected_model:
    st.session_state.selected_model = selected_model
    st.session_state.llm = create_llm(selected_model)
    st.session_state.retrieval_chain = create_retrieval(st.session_state.llm)

for message in st.session_state.messages:
    if "user" == message["role"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    elif "assistant" == message["role"]:
        leftColumn, rightColumn = st.columns(2)
        with leftColumn:
            st.caption("Có sử dụng dữ liệu trong file")
            with st.chat_message(message["role"]):
                st.markdown(message["rag_content"])
                if isinstance(message["rag_context"], list) and len(message["rag_context"]) > 0:
                    with st.expander("Chi tiết các dữ liệu bổ sung được sử dụng khi hỏi đáp"):
                        for doc in message["rag_context"]:
                            st.header(
                                f"{int(doc.metadata['page'])}/{int(doc.metadata['total_pages'])} pages. {doc.metadata['source']}", divider="gray")
                            st.text(doc.page_content)
        with rightColumn:
            st.caption("Không sử dụng dữ liệu trong file")
            with st.chat_message(message["role"]):
                st.markdown(message["llm_content"])

if prompt := st.chat_input("Nhập câu truy vấn tại đây?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    leftColumn, rightColumn = st.columns(2)
    rag_context = None
    rag_response = ""
    llm_response = ""
    with leftColumn:
        st.caption("Có sử dụng dữ liệu trong file")
        with st.chat_message("assistant"):
            placeholder = st.empty()
            for chunk in st.session_state.retrieval_chain.stream({"input": prompt}):
                if "answer" in chunk:
                    rag_response += chunk["answer"]
                    placeholder.write(rag_response)
                elif "context" in chunk:
                    rag_context = chunk["context"]
            if isinstance(rag_context, list) and len(rag_context) > 0:
                with st.expander("Chi tiết các dữ liệu bổ sung được sử dụng khi hỏi đáp"):
                    for doc in rag_context:
                        st.header(
                            f"{int(doc.metadata['page'])}/{int(doc.metadata['total_pages'])} pages. {doc.metadata['source']}", divider="gray")
                        st.text(doc.page_content)
    with rightColumn:
        st.caption("Không sử dụng dữ liệu trong file")
        with st.chat_message("assistant"):
            placeholder = st.empty()
            for chunk in st.session_state.llm.stream([HumanMessage(content=prompt)]):
                llm_response += chunk.content
                placeholder.write(llm_response)

    st.session_state.messages.append(
        {"role": "assistant", "rag_content": rag_response, "llm_content": llm_response, "rag_context": rag_context})
