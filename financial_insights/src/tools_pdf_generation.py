import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import streamlit
from fpdf import FPDF
from fpdf.fpdf import Align
from langchain.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain_text_splitters import CharacterTextSplitter

from financial_insights.streamlit.constants import *

BACKGROUND_COLOUR = (255, 229, 180)
L_MARGIN = 15
T_MARGIN = 20
LOGO_WIDTH = 25


FONT_SIZE = 'helvetica'


class PDFReport(FPDF):  # type: ignore
    def header(self, title_name: str = 'Financial Report') -> None:
        self.set_text_color(SAMBANOVA_ORANGE)

        # Rendering logo:
        self.image(
            'https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
            self.w - self.l_margin - LOGO_WIDTH,
            self.t_margin - self.t_margin / 2,
            LOGO_WIDTH,
        )

        # Setting font: helvetica bold 15
        self.set_font(FONT_SIZE, 'B', 16)
        # Printing title:
        self.cell(0, 10, title_name, align=Align.C)
        self.ln(20)
        self.set_font(FONT_SIZE, '', 10)
        self.cell(
            0,
            0,
            'Powered by SambaNova',
            align=Align.R,
        )
        # Performing a line break:
        self.ln(5)

    def footer(self) -> None:
        # Position cursor at 1.5 cm from bottom:
        self.set_y(-15)
        self.set_text_color(128)
        # Setting font: helvetica italic 8
        self.set_font(FONT_SIZE, 'I', 8)
        # Printing page number:
        self.cell(0, 5, f'Page {self.page_no()}/{{nb}}', align='C')

    def chapter_title(self, title: str) -> None:
        self.set_text_color(SAMBANOVA_ORANGE)
        self.set_font(FONT_SIZE, 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_summary(
        self,
        body: str,
    ) -> None:
        self.set_text_color((0, 0, 0))
        self.set_font(FONT_SIZE, 'I', 11)
        self.multi_cell(0, 5, body)
        self.ln(5)

    def chapter_body(
        self,
        body: str,
    ) -> None:
        self.set_text_color((0, 0, 0))
        self.set_font(FONT_SIZE, '', 11)
        self.multi_cell(0, 5, body)
        self.ln(5)

    def add_figure(self, figure_path: str) -> None:
        # Calculate the desired width of the figure (90% of the page width)
        page_width = self.w - 2 * self.l_margin
        figure_width = page_width * 0.9

        # Place the image on the PDF with the calculated width
        self.image(figure_path, x=self.l_margin + (page_width - figure_width) / 2, w=figure_width)
        self.ln(10)


def read_txt_files(directory: str) -> List[str]:
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r') as file:
                texts.append(file.read())
    return texts


def parse_documents(documents: List[str]) -> list[tuple[str, Any]]:
    report_content = []
    figure_regex = re.compile(r'financial_insights/*[^\s]+\.png')

    for doc in documents:
        parts = doc.split('\n\n\n\n')
        for part in parts:
            # Search for figure paths
            figure_matches = figure_regex.findall(part)
            cleaned_part = figure_regex.sub('', part)  # remove figure paths from text

            if figure_matches:
                for figure_path in figure_matches:
                    report_content.append((cleaned_part.strip(), figure_path))
            else:
                report_content.append((cleaned_part.strip(), None))

    return report_content


# Define your desired data structure.
class SectionTitleSummary(BaseModel):
    title: str = Field(description='Title of the section.')
    summary: str = Field(description='Summary of the section.')


class SectionTitle(BaseModel):
    title: str = Field(description='Title of the section.')


def generate_pdf(
    report_content: List[Tuple[str, Optional[str]]],
    output_file: str,
    title_name: str,
    include_summary: bool = False,
) -> None:
    pdf = PDFReport()
    pdf.set_font('Helvetica')
    pdf.set_page_background(BACKGROUND_COLOUR)
    pdf.set_margins(L_MARGIN, T_MARGIN)

    title_generation_template = (
        'Generate a json formatted concise title (less than 10 words) '
        + 'that captures and summarizes the main idea or theme of following paragraphs.'
        + '\nParagraphs:{text}.'
        + '\n{format_instructions}'
    )

    pdf.add_page()
    content_list: List[Dict[str, str]] = list()
    for idx, content in enumerate(report_content):
        text, figure_path = content
        if (text is None and figure_path is None) or (text is not None and len(text) == 0 and figure_path is None):
            continue

        content_list.append(
            {
                'text': text,
                'figure_path': figure_path if figure_path is not None else '',
            }
        )

    # Parse and split the documents
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=MAX_CHUNK_SIZE, chunk_overlap=0)
    docs = [Document(page_content=content['text']) for content in content_list if len(content['text']) > 0]
    split_docs = text_splitter.split_documents(docs)

    # Add a summary at the beginning
    if include_summary:
        progress_text = f'Summarising {len(content_list)} queries...'
        summary_bar = streamlit.progress(0, text=progress_text)

        intermediate_summaries, intermediate_titles, final_summary, abstract = summarize_text(split_docs)
        pdf.chapter_title('Abstract')
        pdf.chapter_summary(abstract)

        pdf.chapter_title('Summary')
        pdf.chapter_summary(final_summary)

        for idx, item in enumerate(content_list):
            time.sleep(0.01)
            summary_bar.progress(idx + 1, text=progress_text)
            pdf.chapter_title(intermediate_titles[idx])
            pdf.chapter_summary(intermediate_summaries[idx])
            if item['text'] is not None:
                pdf.chapter_body(item['text'])
            if item['figure_path'] is not None and len(item['figure_path']) > 0:
                pdf.add_figure(item['figure_path'])
        time.sleep(0.01)
        summary_bar.empty()
    else:
        for idx, item in enumerate(content_list):
            pdf.chapter_title('Query ' + str(idx))
            if item['text'] is not None:
                pdf.chapter_body(item['text'])
            if item['figure_path'] is not None:
                pdf.add_figure(item['figure_path'])

    pdf.output(output_file)


class Summary(BaseModel):
    title: str = Field(description='Title of the document')
    summary: str = Field(description='Extracted summary of the document')


class SummariesList(BaseModel):
    items: List[Summary] = Field(description='List of titles and summaries')


class ReduceSummary(BaseModel):
    summary: str = Field(description='Final concise summary of the documents')


def summarize_text(split_docs: List[Document]) -> Tuple[List[str], List[str], str, str]:
    # Extract the LLM
    llm = streamlit.session_state.fc.llm

    # Map
    map_template = """The following is a set of documents
        {docs}
        Based on this list of docs, please identify the main themesx§\n.'
        '{format_instructions}'
        """

    map_parser = PydanticOutputParser(pydantic_object=SummariesList)  # type: ignore

    map_prompt = PromptTemplate(
        template=map_template,
        input_variables=['docs'],
        partial_variables={'format_instructions': map_parser.get_format_instructions()},
    )

    map_chain = map_prompt | llm | map_parser

    # Extract intermediate titles and summaries for each document in the split docs
    intermediate_results = map_chain.invoke(split_docs).items
    intermediate_summaries = [item.summary for item in intermediate_results]
    intermediate_titles = [item.title for item in intermediate_results]

    # Reduce
    reduce_template = """The following is set of summaries:
        {intermediate_summaries}
        Take these and distill it into a final, consolidated summary of the main themes.\n'
        '{format_instructions}'
        """
    reduce_parser = PydanticOutputParser(pydantic_object=ReduceSummary)  # type: ignore
    reduce_prompt = PromptTemplate(
        template=reduce_template,
        input_variables=['intermediate_summaries'],
        partial_variables={'format_instructions': reduce_parser.get_format_instructions()},
    )

    reduce_chain = reduce_prompt | llm | reduce_parser

    # Run chain
    final_summary = reduce_chain.invoke('\n'.join(intermediate_summaries)).summary

    # Abstract
    abstract_template = """Write a concise summary of the following:
        {final_summary}.\n
        {format_instructions}
        """
    abstract_parser = PydanticOutputParser(pydantic_object=ReduceSummary)  # type: ignore

    abstact_prompt = PromptTemplate(
        template=abstract_template,
        input_variables=['final_summary'],
        partial_variables={'format_instructions': abstract_parser.get_format_instructions()},
    )
    abstract_chain = abstact_prompt | llm | abstract_parser

    # Run chain
    abstract = abstract_chain.invoke(final_summary).summary

    return intermediate_summaries, intermediate_titles, final_summary, abstract


class PDFRAGInput(BaseModel):
    """Use the provided PDF file to answer the user query using RAG."""

    pdf_file: str = Field('The path to the PDF file for RAG.')


@tool(args_schema=PDFRAGInput)
def pdf_rag(pdf_file: str) -> str:
    loader = PyPDFLoader(pdf_file)

    return ''
