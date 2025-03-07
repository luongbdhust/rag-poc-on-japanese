import streamlit as st
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain.prompts import ChatPromptTemplate

embeddings = PineconeEmbeddings(
    model=st.secrets["EMBEDDING_MODEL"],
    pinecone_api_key=st.secrets["PINECONE_API_KEY"]
)

docsearch = PineconeVectorStore.from_existing_index(
    index_name=st.secrets["PINECONE_INDEX_NAME"],
    embedding=embeddings,
    namespace=st.secrets["PINECONE_NAMESPACE"],
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={
    "k": 5,
})

def create_llm(model_name):
    return ChatOpenAI(
        base_url=st.secrets["BASE_URL_API_KEY"],
        openai_api_key=st.secrets["OPENAI_API_KEY"],
        model_name=model_name,
        temperature=1,
        streaming=True
    )

def create_retrieval(llm):
    custom_prompt = ChatPromptTemplate.from_messages([
        ("human",
         "Reference: {context}\n\nQuestion: {input}. reply in Vietnamese")
    ])
    combine_docs_chain = create_stuff_documents_chain(
        llm, custom_prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)

deepseek_r1_retrieval_chain = create_retrieval(
    create_llm("deepseek/deepseek-r1"))

gpt_4o_mini_retrieval_chain = create_retrieval(
    create_llm("openai/gpt-4o-mini"))

gpt_o3_mini_retrieval_chain = create_retrieval(create_llm("openai/o3-mini"))

if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(layout="wide")
st.header("PoC ứng dụng hỏi đáp trên tập dữ liệu riêng")
st.write("Ứng dụng sử dụng tài liệu [bài phân tích thị trường nhựa tái chế ở việt nam](https://raw.githubusercontent.com/luongbdhust/rag-poc-on-japanese/main/phan-tich-nhua-tai-che-vietnam-japanese.pdf) để làm kiến thức bổ sung cho các llm model khi thực hiện hỏi đáp. Tài liệu này là ngôn ngữ tiếng nhật(Được dịch từ [bản tiếng việt](https://raw.githubusercontent.com/luongbdhust/rag-poc-on-japanese/main/phan-tich-nhua-tai-che-vietnam-vietnamese.pdf) bằng google dịch). Bạn có thể click vào link để tải về xem.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "user" == message["role"]:
            st.markdown(message["content"])
        else:
            st.markdown(message["rag_content"])
            st.caption(message["model"])
            if isinstance(message["rag_context"], list) and len(message["rag_context"]) > 0:
                with st.expander("Chi tiết các dữ liệu bổ sung được sử dụng khi hỏi đáp"):
                    for doc in message["rag_context"]:
                        st.header(
                            f"{int(doc.metadata['page'])}/{int(doc.metadata['total_pages'])} pages. {doc.metadata['source']}", divider="gray")
                        st.text(doc.page_content)

if prompt := st.chat_input("Nhập câu truy vấn tại đây?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    rag_context = None
    rag_response = ""

    model_name = "openai/gpt-4o-mini"
    with st.chat_message("assistant"):
        placeholder = st.empty()
        st.caption(model_name)
        for chunk in gpt_4o_mini_retrieval_chain.stream({"input": prompt}):
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
    st.session_state.messages.append(
        {"role": "assistant", "rag_content": rag_response, "rag_context": rag_context, "model": model_name})

    model_name = "openai/gpt-o3-mini"
    with st.chat_message("assistant"):
        placeholder = st.empty()
        st.caption(model_name)
        for chunk in gpt_o3_mini_retrieval_chain.stream({"input": prompt}):
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
    st.session_state.messages.append(
        {"role": "assistant", "rag_content": rag_response, "rag_context": rag_context, "model": model_name})

    model_name = "deepseek/deepseek-r1"
    with st.chat_message("assistant"):
        placeholder = st.empty()
        st.caption(model_name)
        for chunk in deepseek_r1_retrieval_chain.stream({"input": prompt}):
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
    st.session_state.messages.append(
        {"role": "assistant", "rag_content": rag_response, "rag_context": rag_context, "model": model_name})
