from langchain_ollama import OllamaLLM

model = OllamaLLM(model='deepseekmini')

from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch

class MiniLM(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**inputs)

        embedding = self.mean_pooling(model_output, inputs["attention_mask"])
        return embedding[0].cpu().numpy().tolist()
    
embedding_fn = MiniLM(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda"  # or "cpu"
)

from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

qdrant = QdrantVectorStore(
    client=client,
    collection_name="wikipedia",
    embedding=embedding_fn,
    retrieval_mode=RetrievalMode.DENSE,
)

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Create a prompt template that includes context
prompt_template = """Use the following pieces of context to answer the question. If you don't know the answer based on the context, just say you don't know.

Context: {context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=qdrant.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

query = "whats nga taonga sound?"
result = qa_chain.invoke({"query": query})

print(f"{result['result']}")