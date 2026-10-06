# 🦜 LangChain URL Summarizer: YouTube & Websites

A Streamlit web app that summarizes any **YouTube video** or **website** in about 300 words. It uses **LangChain** and **Meta Llama 3.1 8B Instruct**, served through **Hugging Face Inference Providers**.

Paste a link, click a button, get a summary.

**🔗 Live demo:** [Open the app on Streamlit]-https://huggingface-langchain-kasjtg4xszqwy2gz5synk2.streamlit.app/

---

## ✨ Features

- **YouTube summarization**: pulls the video transcript (`youtube.com` and `youtu.be` links) and summarizes it
- **Website summarization**: scrapes and cleans the page text, then summarizes it
- **Input validation**: checks the URL and token before any API call
- **Context-window safety**: cuts very long content to fit the model's context
- **Clear error handling**: shows readable errors for videos without captions, blocked pages and API failures
- **Secure token handling**: the API token goes in a password field or loads from a local `.env` file

## 🏗️ How it works

```
URL ──► Validate ──► Loader ──────────────► Documents ──► Prompt ──► Llama 3.1 8B ──► Summary
                     ├─ YoutubeLoader (transcript)              (LCEL chain via
                     └─ WebBaseLoader (HTML → text)              Hugging Face)
```

The summarization chain uses **LCEL (LangChain Expression Language)**:

```python
chain = prompt | ChatHuggingFace(llm=HuggingFaceEndpoint(...)) | StrOutputParser()
summary = chain.invoke({"text": content})
```

## 🧰 Tech stack

| Layer | Tools |
|---|---|
| LLM | Meta Llama 3.1 8B Instruct (Hugging Face Inference Providers) |
| Framework | LangChain (`langchain-core`, `langchain-huggingface`, `langchain-community`) |
| Data loading | `YoutubeLoader` (youtube-transcript-api), `WebBaseLoader` (BeautifulSoup) |
| UI | Streamlit |
| Utilities | `validators`, `python-dotenv` |

## 🚀 Run locally

```bash
# 1. Clone
git clone https://github.com/aditsharma14/HuggingFace-LangChain.git
cd HuggingFace-LangChain

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) add your Hugging Face token to a .env file
echo hf_token=hf_xxxxxxxxxxxxxxxx > .env

# 5. Run
streamlit run app.py
```

Get a free Hugging Face token at <https://huggingface.co/settings/tokens>. A **Read** token with the *"Make calls to Inference Providers"* permission is enough.

## 📁 Project structure

```
├── app.py              # Streamlit summarizer app
├── experiments.ipynb   # Experiments: HF chat models, prompt templates, LCEL chains, BGE embeddings
├── requirements.txt
└── readme.md
```

### Experiments notebook

`experiments.ipynb` covers what led up to the app:
- Calling **Llama 3.1 8B** and **Gemma 3 12B** through `HuggingFaceEndpoint` + `ChatHuggingFace`
- Chain-of-thought style prompting with `PromptTemplate`
- Building chains with the LCEL pipe (`|`) operator
- Generating text embeddings locally with **BAAI/bge-small-en** (`HuggingFaceEmbeddings`)

## ⚠️ Known limitations

- YouTube videos need captions or a transcript to be summarized.
- Some websites block scrapers or load their content with JavaScript, so little or no text can be pulled from them.
- YouTube may block transcript requests from cloud servers, so YouTube links can fail on the hosted demo even when they work locally.
- Content longer than ~20,000 characters is cut, so the summary covers only the beginning.

## 🔮 Future improvements

- Map-reduce summarization for long videos and articles
- Model picker (Llama, Gemma, Mistral, Qwen)
- Streaming output
- PDF upload support

## 👤 Author

**Adit Sharma**: [GitHub @aditsharma14](https://github.com/aditsharma14)
