"""
Scope: Document assistant for Langchain website documentation

Documentation sources:

https://api.python.langchain.com/en/v0.1/langchain_api_reference.html

https://python.langchain.com/docs/introduction/

https://langchain-ai.github.io/langgraph/

In order to download the web contents with all links, use the following command on cmd line for each source:
wget -r -A.html -P langgraph-docs https://langchain-ai.github.io/langgraph/

"""

from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain import hub
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from rank_bm25 import BM25Okapi
from typing import List, Dict, Optional
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from chromadb.config import Settings
from langchain_chroma import Chroma
from pydantic import Field
import msgspec
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

nltk.download('punkt')
nltk.download('stopwords')

stop_words_en = set(stopwords.words('english'))
stop_words_tr = set(stopwords.words('turkish'))

import warnings
from langsmith.utils import LangSmithMissingAPIKeyWarning
warnings.filterwarnings("ignore", category=LangSmithMissingAPIKeyWarning)

class Data(msgspec.Struct):
    documents: list[str]
    metadata: list[dict]

#--------------------------------------------------------------------------------------------------
#Custom BM25Retriever in order to utilize if all keywords are inside a text chunk, give higher score to that chunk in ranking with scores

class CustomBM25Retriever(BaseRetriever):
    k1: float = Field(default=1.2)
    b: float = Field(default=0.75)
    phrase_boost: float = Field(default=1.5)
    k: int = Field(default=10)  # Number of top documents to retrieve
    documents: List[Document] = Field(default_factory=list)
    tokenized_docs: List[List[str]] = Field(default_factory=list)
    bm25: Optional[BM25Okapi] = None

    def __init__(self, documents: List[Document], k: int = 10, k1: float = 1.2, b: float = 0.75, phrase_boost: float = 1.5):
        super().__init__()
        self.documents = documents
        self.k = k
        self.k1 = k1
        self.b = b
        self.phrase_boost = phrase_boost
        self.tokenized_docs = [self.tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=self.k1, b=self.b)

    @classmethod
    def from_texts(cls, texts: List[str], metadatas: Optional[List[Dict]] = None, k: int = 10, bm25_params: Optional[Dict] = None):
        """ Factory method to initialize from raw texts and metadata (similar to LangChain's BM25Retriever) """
        bm25_params = bm25_params or {"k1": 1.2, "b": 0.75, "phrase_boost": 1.0}
        documents = [Document(page_content=text.lower(), metadata=meta or {}) for text, meta in zip(texts, metadatas or [{}] * len(texts))]
        return cls(documents=documents, k=k, **bm25_params)


    def tokenize(self, doc: Document):
        """ Tokenizes both content and metadata for retrieval """
        metadata_str = " ".join(f"{key}: {value}" for key, value in doc.metadata.items())
        full_text = f"{doc.page_content} {metadata_str}"
        return full_text.split()

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        """ Retrieve and rank documents using BM25 with metadata and phrase boosting """
        if not self.bm25:
            raise ValueError("BM25 model not initialized. Call __init__ first.")

        query = query.lower()
        query = re.sub(r'[^\w\s.,%@()*-+/!&_?#|]', '', query) 

        tokens = word_tokenize(query)
        filtered_query_tokens = [word for word in tokens if word not in stop_words_en]

        scores = self.bm25.get_scores(filtered_query_tokens)

        # Boost score if query appears as a phrase

        boosted_scores = []
        for i, doc in enumerate(self.documents):
            full_text = f"{doc.page_content} " + " ".join(f"{key}: {value}" for key, value in doc.metadata.items())
            phrase_bonus = self.phrase_boost if query.lower() in full_text.lower() else 1.0           
            boosted_scores.append(scores[i] * phrase_bonus)

        # Rank documents by boosted BM25 score
        ranked_docs = sorted(zip(self.documents, boosted_scores), key=lambda x: x[1], reverse=True)
        return [doc[0] for doc in ranked_docs[:self.k]]  # Return only top_k documents

# Initialize the Chroma client (PersistentClient for persistent storage)
from langchain import hub
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


import warnings
from langsmith.utils import LangSmithMissingAPIKeyWarning
warnings.filterwarnings("ignore", category=LangSmithMissingAPIKeyWarning)

#----------------------------------------------------------
# Get the semantic retriever from the Chroma vectorstore
llm = ChatOllama(model="cogito:14b", temperature=0.1, repeat_penalty=1.1, disable_streaming=False)

embeddings = OllamaEmbeddings(model="bge-m3")

vector_store = Chroma(
    collection_name = "my-doc-assistant-db",
    embedding_function=embeddings,
    persist_directory="D:\\Langchain-Langgraph-Doc-WebSites\\__Databases\\langchain-docs-vectordb"
    )


retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)


search_as = { "k": 5, "lambda_mult": 0.8,  "score_threshold": 0.1, "fetch_k": 10} 
semantic_retriever = vector_store.as_retriever(search_type="mmr",search_kwargs=search_as)


#----------------------------------------------------------
#Read the document chunks and metadata as a file and put them into the BM25REtriever -- heavy RAM load
folder = "D:\\Langchain-Langgraph-Doc-WebSites\\__Databases\\langchain-docs-bm25-db\\"
with open(folder + 'bm25_data_docs_and_metadata.json', 'rb') as file:
    data = msgspec.json.decode(file.read(), type=Data)

print("bm25_data_docs_and_metadata.json read...")

# Initialize Custom BM25 Retriever instead of default BM25Retriever
bm25_retriever = CustomBM25Retriever.from_texts(
                                            data.documents, 
                                            data.metadata, 
                                            k=10, 
                                            bm25_params={"k1": 1.0, "b": 0.5, "phrase_boost": 1.5}
                                            )

print(f"length of data bm25: {len(data.documents)}")
#-----------------------------------------------------------
#Create the ensemble retriever with BM25 retriever and semantic retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever], 
    weights=[0.5, 0.5]
)


retrieval_chain = create_retrieval_chain(
    retriever = ensemble_retriever,
    combine_docs_chain = combine_docs_chain,
)

import warnings
from langsmith.utils import LangSmithMissingAPIKeyWarning
warnings.filterwarnings("ignore", category=LangSmithMissingAPIKeyWarning)


query = "why use langgraph?"

response = retrieval_chain.invoke(input={"input": query})


print("Response: ")
for chunk in response["answer"]:
    print(chunk, end="", flush=True)

print("\nSources: ")
sources = [doc.metadata["source"] for doc in response["context"]]
for i, chunk in zip(range(len(sources)), sources):
    print(f"{i} - {chunk}\n", end="", flush=True)  
