import json
import os
import re
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from typing import Any, Callable, Generator, List, Optional, Type, Union

import pandas
import streamlit
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import StructuredTool, Tool
from matplotlib.figure import Figure
from PIL import Image

from financial_insights.src.function_calling import FunctionCalling
from financial_insights.src.tools import get_conversational_response
from financial_insights.src.tools_database import create_stock_database, query_stock_database
from financial_insights.src.tools_filings import retrieve_filings
from financial_insights.src.tools_pdf_generation import pdf_rag
from financial_insights.src.tools_stocks import (
    get_historical_price,
    get_stock_info,
    retrieve_symbol_list,
    retrieve_symbol_quantity_list,
)
from financial_insights.src.tools_yahoo_news import scrape_yahoo_finance_news
from financial_insights.streamlit.constants import *

# tool mapping of available tools
TOOLS = {
    'get_stock_info': get_stock_info,
    'get_historical_price': get_historical_price,
    'retrieve_symbol_list': retrieve_symbol_list,
    'retrieve_symbol_quantity_list': retrieve_symbol_quantity_list,
    'scrape_yahoo_finance_news': scrape_yahoo_finance_news,
    'get_conversational_response': get_conversational_response,
    'retrieve_filings': retrieve_filings,
    'create_stock_database': create_stock_database,
    'query_stock_database': query_stock_database,
    'pdf_rag': pdf_rag,
}


@contextmanager
def st_capture(output_func: Callable[[Any], Any]) -> Generator[None, None, None]:
    """
    Context manager to catch stdout and send it to an output function.

    Args:
        output_func (Callable[[str], None]): Function to write terminal output to.

    Yields:
        None: A generator that redirects stdout to the output_func.
    """
    stdout = StringIO()
    with redirect_stdout(stdout):
        try:
            yield
        finally:
            output_func(stdout.getvalue())


def attach_tools(
    tools: Optional[List[str]] = None,
    default_tool: Optional[Union[StructuredTool, Tool, Type[BaseModel]]] = None,
) -> None:
    """
    Attach the tools to the streamit session for the LLM to use.

    Args:
        tools (list): list of tools to be used
    """
    if tools is not None:
        set_tools = [TOOLS[name] for name in tools]
    else:
        set_tools = [default_tool]
    streamlit.session_state.fc = FunctionCalling(tools=set_tools, default_tool=default_tool)


def handle_userinput(user_question: Optional[str], user_query: Optional[str]) -> Optional[Any]:
    """
    Handle user input and generate a response, also update chat UI in streamlit app.

    Args:
        user_question (str): The user's question or input.
    """
    output = streamlit.empty()

    with streamlit.spinner('Processing...'):
        with st_capture(output.code):
            tool_messages, response = streamlit.session_state.fc.function_call_llm(
                query=user_query,
                debug=True,
            )

    streamlit.session_state.chat_history.append(user_question)
    streamlit.session_state.chat_history.append(response)

    with streamlit.chat_message('user'):
        streamlit.write(f'{user_question}')

    stream_response_object(response, stream_response=True)

    return response


def stream_chat_history() -> None:
    for question, answer in zip(
        streamlit.session_state.chat_history[::2],
        streamlit.session_state.chat_history[1::2],
    ):
        with streamlit.chat_message('user'):
            streamlit.write(question)

        stream_response_object(answer, stream_response=True)


def stream_response_object(response: Any, stream_response: bool = False) -> Any:
    if isinstance(response, (str, float, int, Figure, pandas.DataFrame)):
        return stream_complex_response(response, stream_response)

    elif isinstance(response, list) or isinstance(response, dict):
        return stream_complex_response(response, stream_response)

    elif isinstance(response, tuple):
        json_response = list()
        for item in response:
            json_response.append(stream_complex_response(item, stream_response))
        return json.dumps(json_response)
    else:
        return


def stream_complex_response(response: Any, stream_response: bool = False) -> Any:
    if isinstance(response, (str, float, int, Figure, pandas.DataFrame)):
        if stream_response:
            stream_single_response(response)

    elif isinstance(response, list):
        if stream_response:
            for item in response:
                stream_single_response(item)

    elif isinstance(response, dict):
        if stream_response:
            for key, value in response.items():
                if isinstance(value, str):
                    stream_single_response(key + ': ' + value)
                elif isinstance(value, list):
                    # If all values are strings
                    stream_single_response(key + ': ' + ', '.join([str(item) for item in value]) + '.')

    try:
        return json.dumps(response)
    except:
        return ''


def stream_single_response(response: Any) -> None:
    """Streamlit chatbot response."""
    # If response is a string
    if isinstance(response, (str, float, int)):
        with streamlit.chat_message(
            'ai',
            avatar='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
        ):
            if isinstance(response, str):
                if not response.endswith('.png'):
                    # Ploat the last path of the png file
                    streamlit.write(response)
                else:
                    # Load the image after extracting the last path of the png file
                    image = Image.open(response.split(' ')[-1])

                    # Display the image
                    streamlit.image(image, use_column_width=True)

                png_paths = extract_png_paths(response)
                for path in png_paths:
                    if path != response:
                        # Extract the last part of path
                        relative_path = extract_path_after('ai-starter-kit', path)
                        assert isinstance(relative_path, str), 'Path should be a string'
                        # Load the image
                        image = Image.open(relative_path)
                        # Display the image
                        streamlit.image(image, use_column_width=True)

    # If response is a figure
    elif isinstance(response, Figure):
        # Display the image
        # streamlit.image(response, use_column_width=True)
        # Show the figure
        streamlit.pyplot(response)

    # If response is a dataframe, display its head
    elif isinstance(response, pandas.DataFrame):
        streamlit.write(response.head())


def extract_png_paths(sentence: str) -> List[str]:
    png_pattern = r'\b\S+\.png\b'
    png_paths: List[str] = []
    matches = re.findall(png_pattern, sentence)
    return matches


def extract_path_after(directory: str, path: str) -> Optional[str]:
    # Normalize paths to avoid issues with different path representations
    norm_directory = os.path.normpath(directory)
    norm_path = os.path.normpath(path)

    # Find the directory in the path
    try:
        index = norm_path.index(norm_directory)
    except ValueError:
        return None  # The directory is not in the path

    # Extracting the part after the directory
    start_pos = index + len(norm_directory)

    # Ensure that the directory is found in the path at the end of a segment
    if start_pos >= len(norm_path) or norm_path[start_pos] != os.sep:
        return None  # The directory is not in the path

    # Return the substring from the character after the directory onwards
    return norm_path[start_pos + 1 :]
