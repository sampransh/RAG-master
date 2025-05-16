import streamlit as st
import base64
import RAG_app as rag

st.set_page_config(page_title="Analyze your data via chat")
# Define a custom theme
def set_theme():
    st.markdown(
        """
       
        """,
        unsafe_allow_html=True,
    )


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background: url("data:image/png;base64,%s") no-repeat center;
    background-size: cover;

    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


def model_selection_view():
    st.subheader("Model Providers")
    llm_chooser = st.radio(
        "  ",
        rag.list_LLM_providers,
        captions=[
            "Under development",
            "Working almost FREE",
            "Under development",
        ],

    )

    st.divider()
    st.subheader("API Key")
    if llm_chooser == rag.list_LLM_providers[0]:
        rag.expander_model_parameters(
            LLM_provider="OpenAI",
            text_input_API_key="Your OpenAI API Key - [Get New API key](https://platform.openai.com/account/api-keys)",
            list_models=[
                "gpt-3.5-turbo-0125",
                "gpt-3.5-turbo",
                "gpt-4-turbo-preview",
            ],
        )

    if llm_chooser == rag.list_LLM_providers[1]:
        rag.expander_model_parameters(
            LLM_provider="Google",
            text_input_API_key="Your Google API Key - [Get New API key](https://makersuite.google.com/app/apikey)",
            list_models=["gemini-2.0-flash-001"],
        )
    if llm_chooser == rag.list_LLM_providers[2]:
        rag.expander_model_parameters(
            LLM_provider="HuggingFace",
            text_input_API_key="Your HuggingFace API key - [Get New API key](https://huggingface.co/settings/tokens)",
            list_models=["mistralai/Mistral-7B-Instruct-v0.2",
                         "meta-llama/Meta-Llama-3-8B",
                         "mistralai/Mistral-7B-Instruct-v0.1",
                         "HuggingFaceH4/zephyr-7b-beta",
                         "nkgwh/mistralai_Mistral-7B-Instruct-v0.2-CITPRED-ABSTRACT"],
        )
    # Assistant language
    st.write("")
    st.session_state.assistant_language = "english"

    st.divider()
    st.subheader("Retrievers")
    retrievers = rag.list_retriever_types
    if st.session_state.selected_model == "gpt-3.5-turbo":
        # for "gpt-3.5-turbo", we will not use the vectorstore backed retriever
        # there is a high risk of exceeding the max tokens limit (4096).
        retrievers = rag.list_retriever_types[:-1]

    st.session_state.retriever_type = st.selectbox(
        f"Select retriever type", retrievers
    )
    st.write("")
    if st.session_state.retriever_type == rag.list_retriever_types[0]:  # Cohere
        st.session_state.cohere_api_key = st.text_input(
            "Cohere API Key - [Get an API key](https://dashboard.cohere.com/api-keys)",
            type="password",
            placeholder="insert your API key",
        )

    st.write("\n\n")

def chat_view():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("                                       🤖 Talk with Your Document....\n\n")
    with col2:
        st.button("Clear Chat History", on_click=rag.clear_chat_history)

    st.write("\n\n\n\n\n\n\n\n\n")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": rag.dict_welcome_message[st.session_state.assistant_language],
            }
        ]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Type your question here....."):
        if (
                not st.session_state.openai_api_key
                and not st.session_state.google_api_key
                and not st.session_state.hf_api_key
        ):
            st.info(
                f"Please insert your {st.session_state.LLM_provider} API key to continue."
            )
            st.stop()

        st.write("\n")
        with st.spinner("Talking with the Model, Please Wait......"):
            st.write("\n\n")
            st.chat_message("user").write(prompt)
            st.write("\n\n")
            rag.get_response_from_LLM(prompt=prompt)

def create_vectorstore_view():
    st.subheader("Create a new Vector Store.")
    # 1. Select documnets
    st.session_state.uploaded_file_list = st.file_uploader(
        label=" ",
        accept_multiple_files=True,
        type=(["pdf", "txt", "docx", "csv"]),
        label_visibility='collapsed',
    )
    # 2. Process documents
    st.session_state.vector_store_name = st.text_input(
        label=" ",
        placeholder="Vectorstore name",
        label_visibility='collapsed',
    )
    # 3. Add a button to process documnets and create a Chroma vectorstore

    st.button("Create Vectorstore", on_click=rag.chain_RAG_blocks)
    try:
        if st.session_state.error_message != "":
            st.warning(st.session_state.error_message)
    except:
        pass
    st.write("\n\n\n\n\n")

def open_vectorstore_view():
    # Open a saved Vectorstore
    # https://github.com/streamlit/streamlit/issues/1019
    st.subheader("Please select a Vectorstore:")
    import tkinter as tk
    from tkinter import filedialog

    clicked = st.button("Vectorstore chooser")
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)  # Make dialog appear on top of other windows

    st.session_state.selected_vectorstore_name = ""

    if clicked:
        # Check inputs
        error_messages = []
        if (
                not st.session_state.openai_api_key
                and not st.session_state.google_api_key
                and not st.session_state.hf_api_key
        ):
            error_messages.append(
                f"insert your {st.session_state.LLM_provider} API key"
            )

        if (
                st.session_state.retriever_type == rag.list_retriever_types[0]
                and not st.session_state.cohere_api_key
        ):
            error_messages.append(f"insert your Cohere API key")

        if len(error_messages) == 1:
            st.session_state.error_message = "Please " + error_messages[0] + "."
            st.warning(st.session_state.error_message)
        elif len(error_messages) > 1:
            st.session_state.error_message = (
                    "Please "
                    + ", ".join(error_messages[:-1])
                    + ", and "
                    + error_messages[-1]
                    + "."
            )
            st.warning(st.session_state.error_message)

        # if API keys are inserted, start loading Chroma index, then create retriever and ConversationalRetrievalChain
        else:
            selected_vectorstore_path = filedialog.askdirectory(master=root)

            if selected_vectorstore_path == "":
                st.info("Please select a valid path.")

            else:
                with st.spinner("Loading vectorstore..."):
                    st.session_state.selected_vectorstore_name = (
                        selected_vectorstore_path.split("/")[-1]
                    )
                    try:
                        # 1. load Chroma vectorestore
                        embeddings = rag.select_embeddings_model()
                        st.session_state.vector_store = rag.Chroma(
                            embedding_function=embeddings,
                            persist_directory=selected_vectorstore_path,
                        )

                        # 2. create retriever
                        st.session_state.retriever = rag.create_retriever(
                            vector_store=st.session_state.vector_store,
                            embeddings=embeddings,
                            retriever_type=st.session_state.retriever_type,
                            base_retriever_search_type="similarity",
                            base_retriever_k=16,
                            compression_retriever_k=20,
                            cohere_api_key=st.session_state.cohere_api_key,
                            cohere_model="rerank-multilingual-v3.0",
                            cohere_top_n=10,
                        )

                        # 3. create memory and ConversationalRetrievalChain
                        (
                            st.session_state.chain,
                            st.session_state.memory,
                        ) = rag.create_ConversationalRetrievalChain(
                            retriever=st.session_state.retriever,
                            chain_type="stuff",
                            language=st.session_state.assistant_language,
                        )

                        # 4. clear chat_history
                        rag.clear_chat_history()

                        st.info(
                            f"**{st.session_state.selected_vectorstore_name}** is loaded successfully."
                        )

                    except Exception as e:
                        st.error(e)

# Main app layout
if __name__ == "__main__":
    set_theme()  # Apply the custom theme
    set_background('./images/bg1.png')

    with st.sidebar:
        st.header(
            " GPT – DATA ANALYST"
        )

        tab_model_selection, tab_create_vectorstore, tab_open_vectorstore = st.tabs(
            ["Select Model", "Create New Vectorstore", "Open Saved Vectorstore"]
        )
        with tab_model_selection:
            model_selection_view()

        with tab_create_vectorstore:
            create_vectorstore_view()

        with tab_open_vectorstore:
            open_vectorstore_view()

    chat_view()