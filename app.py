import os

## Identify our requests to websites (some sites, e.g. Wikipedia, block generic user agents)
USER_AGENT = "LangChainSummarizer/1.0 (educational project)"
os.environ.setdefault("USER_AGENT", USER_AGENT)

import validators, streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import YoutubeLoader, WebBaseLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

## Keep the prompt within the model's context window
MAX_CONTENT_CHARS = 20000


## Streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')


## Get the Huggingface API Token and url(YT or website) to be summarized
with st.sidebar:
    hf_api_key = st.text_input("Huggingface API Token", value=os.getenv("hf_token", ""), type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}

"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])


def get_llm(token):
    ## Llama 3.1 via Huggingface Inference Providers
    repo_id = "meta-llama/Llama-3.1-8B-Instruct"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="conversational",
        max_new_tokens=512,
        temperature=0.7,
        huggingfacehub_api_token=token,
    )
    return ChatHuggingFace(llm=llm)


if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not hf_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It may be a YT video url or website url")

    else:
        try:
            with st.spinner("Waiting..."):
                ## loading the website or yt video data
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=False)
                else:
                    loader = WebBaseLoader(generic_url, header_template={"User-Agent": USER_AGENT})
                docs = loader.load()

                content = "\n\n".join(doc.page_content for doc in docs).strip()
                if not content:
                    st.error("Could not extract any text from this URL (YouTube videos need captions).")
                    st.stop()

                ## Chain For Summarization
                chain = prompt | get_llm(hf_api_key.strip()) | StrOutputParser()
                output_summary = chain.invoke({"text": content[:MAX_CONTENT_CHARS]})

                st.success(output_summary)
        except Exception as e:
            st.exception(e)
