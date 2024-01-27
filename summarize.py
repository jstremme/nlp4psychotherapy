"""
Transcript summarizer based on:
https://www.koyeb.com/tutorials/use-langchain-deepgram-and-mistral7b-to-build-a-youtube-video-summarization-app
"""

import os
import time
from langchain.chains import MapReduceDocumentsChain, LLMChain, ReduceDocumentsChain, StuffDocumentsChain
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def summarize_transcript(transcript_str, config, transcript_topic, wrt, show_input_text=False):

    # Convert transcript to document
    docs = [Document(page_content=transcript_str, metadata={"source": "local"})]
    if show_input_text:
        print("Input text:")
        print(docs[0].page_content)

    # Load LLM
    llm = CTransformers(
        model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        model_file="mistral-7b-instruct-v0.1.Q6_K.gguf", # very large, extremely low quality loss
        config=config,
        threads=os.cpu_count()
    )

    # Map template and chain
    map_template = """<s>[INST] The following is a section from a {transcript_topic} transcript:
    {docs}
    Identify the main points of the section.
    Answer:  [/INST] </s>"""
    map_prompt = PromptTemplate.from_template(map_template, partial_variables={"transcript_topic": transcript_topic})
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce template and chain
    reduce_template = """<s>[INST] The following is set of section summaries from the {transcript_topic} transcript:
    {doc_summaries}
    Distill these into a final, consolidated summary with respect to {wrt}.
    Answer:  [/INST] </s>"""
    reduce_prompt = PromptTemplate.from_template(reduce_template, partial_variables={"transcript_topic": transcript_topic, "wrt": wrt})
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries"
    )
    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=4000,
    )
    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=True,
    )

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, chunk_overlap=0
    )
    split_docs = text_splitter.split_documents(docs)

    # Run the chain
    start_time = time.time()
    result = map_reduce_chain.__call__(split_docs, return_only_outputs=True)
    print(f"Time taken: {(time.time() - start_time) / 60 } minutes.")

    return result['output_text']

if __name__ == "__main__":

    with open("transcript.txt", 'rb') as f:
        transcript = f.read()

    config = {'max_new_tokens': 256, 'temperature': 0.7, 'context_length': 4096}
    transcript_topic = 'book'
    wrt = 'the plot'

    summary = summarize_transcript(
        transcript,
        config=config,
        transcript_topic=transcript_topic,
        wrt=wrt
    )
    
    print("Summary:")
    print(summary)
