import streamlit as st
from agent import create_gemini_agent, process_uploaded_file, get_api_key
from agent import set_vector_store, get_vector_store, clear_vector_store

# 1. Page Configuration
st.set_page_config(page_title="Assistant D", layout="centered")

st.title("AI Assistant D")
st.markdown("### An AI agent powered by Google Gemini 3.6 Flash")
st.markdown("---")

# 2. Initialize Agent in Session State
if "agent_executor" not in st.session_state:
    with st.spinner("Initializing Assistant D..."):
        st.session_state.agent_executor = create_gemini_agent()
        st.session_state.config = {"configurable": {"thread_id": "streamlit_session_v1"}}

# 3. Sidebar - Document Upload for RAG
with st.sidebar:
    st.header("Upload Documents")
    st.caption("Upload a file and ask questions about it")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "md", "csv"],
        help="Supported formats: PDF, TXT, MD, CSV",
    )
    if uploaded_file is not None:
        file_key = f"processed_{uploaded_file.name}_{uploaded_file.size}"
        if file_key not in st.session_state:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                api_key = get_api_key()
                vector_store, message = process_uploaded_file(uploaded_file, api_key)
                if vector_store is not None:
                    existing = get_vector_store()
                    if existing is not None:
                        existing.merge_from(vector_store)
                        set_vector_store(existing)
                        st.success(f"Added! {message}")
                    else:
                        set_vector_store(vector_store)
                        st.success(message)
                else:
                    st.error(message)
                st.session_state[file_key] = True
        else:
            st.info(f"'{uploaded_file.name}' already processed.")

    if get_vector_store() is not None:
        st.divider()
        st.caption("Documents are loaded and ready for questions!")
        if st.button("Clear Documents"):
            clear_vector_store()
            for key in list(st.session_state.keys()):
                if key.startswith("processed_"):
                    del st.session_state[key]
            st.rerun()

# 4. Chat History Management
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. User Input & Agent Logic
if prompt := st.chat_input("How may I assist you?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Assistant D is thinking..."):
            try:
                response = st.session_state.agent_executor.invoke(
                    {"messages": [("user", prompt)]},
                    config=st.session_state.config,
                )

                raw = response["messages"][-1].content
                if isinstance(raw, list):
                    output = "".join(
                        block["text"] for block in raw if isinstance(block, dict) and "text" in block
                    )
                else:
                    output = raw

                st.markdown(output)
                st.session_state.messages.append(
                    {"role": "assistant", "content": output}
                )

            except Exception as e:
                error_msg = f"Something went wrong: {e}"
                st.error(error_msg)
