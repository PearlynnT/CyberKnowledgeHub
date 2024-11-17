import logging
import os
import shutil
import sys
import traceback
import uuid
from typing import List, Optional

import streamlit as st
import yaml
from streamlit.runtime.uploaded_file_manager import UploadedFile

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from typing import Optional

from enterprise_knowledge_retriever.src.document_retrieval import DocumentRetrieval
from utils.events.mixpanel import MixpanelEvents
from utils.visual.env_utils import are_credentials_set, env_input_fields, initialize_env_variables, save_credentials

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')
PERSIST_DIRECTORY = os.path.join(kit_dir, f'data/my-vector-db')

logging.basicConfig(level=logging.INFO)
logging.info('URL: http://localhost:8501')


def handle_userinput(user_question: str) -> None:
    if user_question:
        try:
            with st.spinner('Processing...'):
                response = st.session_state.conversation.invoke({'question': user_question})
            print(f"user question: {user_question}")
            st.session_state.chat_history.append(user_question)
            print(f"response: (response['answer']")
            st.session_state.chat_history.append(response['answer'])

            sources = set([f'{sd.metadata["filename"]}' for sd in response['source_documents']])
            sources_text = ''
            for index, source in enumerate(sources, start=1):
                source_link = source
                sources_text += f'<font size="2" color="grey">{index}. {source_link}</font>  \n'
            st.session_state.sources_history.append(sources_text)
        except Exception as e:
            st.error(f'An error occurred while processing your question: {str(e)}')
            logging.error("Error processing question", exc_info=True)
            print(f"Detailed error: {traceback.format_exc()}")

    for ques, ans, source in zip(
        st.session_state.chat_history[::2],
        st.session_state.chat_history[1::2],
        st.session_state.sources_history,
    ):
        with st.chat_message('user'):
            st.write(f'{ques}')

        with st.chat_message(
            'ai',
            avatar='https://img.freepik.com/premium-vector/hacker-vector-concept-unknown-man-stealing-data-from-email-while-using-laptop_505557-4212.jpg',
        ):
            st.write(f'{ans}')
            if st.session_state.show_sources:
                with st.expander('Sources'):
                    st.markdown(
                        f'<font size="2" color="grey">{source}</font>',
                        unsafe_allow_html=True,
                    )


def initialize_document_retrieval(prod_mode: bool) -> Optional[DocumentRetrieval]:
    if prod_mode:
        sambanova_api_key = st.session_state.SAMBANOVA_API_KEY
    else:
        if 'SAMBANOVA_API_KEY' in st.session_state:
            sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY') or st.session_state.SAMBANOVA_API_KEY
        else:
            sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY')
    if are_credentials_set():
        try:
            return DocumentRetrieval(sambanova_api_key=sambanova_api_key)
        except Exception as e:
            st.error(f'Failed to initialize DocumentRetrieval: {str(e)}')
            return None
    return None


def main() -> None:
    with open(CONFIG_PATH, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    prod_mode = config.get('prod_mode', False)
    conversational = config['retrieval'].get('conversational', False)
    default_collection = 'ekr_default_collection'

    initialize_env_variables(prod_mode)

    st.set_page_config(
        page_title='Cyber Knowledge Hub',
        page_icon='https://img.freepik.com/premium-vector/hacker-vector-concept-unknown-man-stealing-data-from-email-while-using-laptop_505557-4212.jpg',
    )

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'show_sources' not in st.session_state:
        st.session_state.show_sources = True
    if 'sources_history' not in st.session_state:
        st.session_state.sources_history = []
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'input_disabled' not in st.session_state:
        st.session_state.input_disabled = True
    if 'document_retrieval' not in st.session_state:
        st.session_state.document_retrieval = None
    if 'st_session_id' not in st.session_state:
        st.session_state.st_session_id = str(uuid.uuid4())
    if 'mp_events' not in st.session_state:
        st.session_state.mp_events = MixpanelEvents(
            os.getenv('MIXPANEL_TOKEN'),
            st_session_id=st.session_state.st_session_id,
            kit_name='enterprise_knowledge_retriever',
            track=prod_mode,
        )
        st.session_state.mp_events.demo_launch()

    page_by_img = """
    <style>
    [data-testid="stAppViewContainer"] {
    background-image: url("https://www.shutterstock.com/shutterstock/videos/1079786111/thumb/12.jpg?ip=x480");
    background-size: cover;
    }
    </style>
    """

    st.markdown(page_by_img, unsafe_allow_html=True)
    st.markdown(
        """
        <h1 style='color: white; text-align: center;'>Cyber Knowledge Hub</h1>
        """,
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.title('Setup')

        # Callout to get SambaNova API Key
        st.markdown('Get your SambaNova API key [here](https://cloud.sambanova.ai/apis)')

        if not are_credentials_set():
            url, api_key = env_input_fields()
            if st.button('Save Credentials', key='save_credentials_sidebar'):
                message = save_credentials(url, api_key, prod_mode)
                st.session_state.mp_events.api_key_saved()
                st.success(message)
                st.rerun()
        else:
            st.success('Credentials are set')
            if st.button('Clear Credentials', key='clear_credentials'):
                save_credentials('', '', prod_mode)  # type: ignore
                st.rerun()

        if are_credentials_set():
            if st.session_state.document_retrieval is None:
                st.session_state.document_retrieval = initialize_document_retrieval(prod_mode)

        if st.session_state.document_retrieval is not None:
            st.markdown('**1. Connect to Pinata**')
            if st.button('Connect to Pinata'):
                with st.spinner('Connecting...'):
                    try:
                        pinata_files = st.session_state.document_retrieval.pinata_client.list_files()
                        st.success(f'Connected to Pinata. Found {len(pinata_files)} files.')
                        st.session_state.pinata_connected = True
                    except Exception as e:
                        st.error(f'Failed to connect to Pinata: {str(e)}')

            if st.session_state.get('pinata_connected', False):
                st.markdown('**2. Process documents and create vector store**')
                if st.button('Process'):
                    with st.spinner('Processing'):
                        try:
                            text_chunks = st.session_state.document_retrieval.parse_doc('')
                            embeddings = st.session_state.document_retrieval.load_embedding_model()
                            vectorstore = st.session_state.document_retrieval.create_vector_store(text_chunks, embeddings)
                            st.session_state.vectorstore = vectorstore
                            st.session_state.document_retrieval.init_retriever(vectorstore)
                            st.session_state.conversation = st.session_state.document_retrieval.get_qa_retrieval_chain(
                                conversational=conversational
                            )
                            st.toast(f'Documents processed! Go ahead and ask some questions', icon='🎉')
                            st.session_state.input_disabled = False
                        except Exception as e:
                            st.error(f'An error occurred while processing: {str(e)}')
                            
            st.markdown('**3. Ask questions about your data!**')

            with st.expander('Additional settings', expanded=True):
                st.markdown('**Interaction options**')
                st.markdown('**Note:** Toggle these at any time to change your interaction experience')
                show_sources = st.checkbox('Show sources', value=True, key='show_sources')

                st.markdown('**Reset chat**')
                st.markdown('**Note:** Resetting the chat will clear all conversation history')
                if st.button('Reset conversation'):
                    st.session_state.chat_history = []
                    st.session_state.sources_history = []
                    if not st.session_state.input_disabled:
                        st.session_state.conversation = st.session_state.document_retrieval.get_qa_retrieval_chain(
                            conversational=conversational
                        )
                    st.toast('Conversation reset. The next response will clear the history on the screen')
                    logging.info('Conversation reset')

    user_question = st.chat_input('Ask questions about your data', disabled=st.session_state.input_disabled)
    if user_question is not None:
        st.session_state.mp_events.input_submitted('chat_input')
        handle_userinput(user_question)


if __name__ == '__main__':
    main()