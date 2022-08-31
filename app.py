from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Init is ran on server startup
# Load your model to GPU as a global variable here.
def init():
    global model
    global tokenizer
    model = SentenceTransformer('msmarco-distilbert-base-tas-b')
    # conditionally load to GPU
    if device == "cuda:0":
        print("loading to GPU...")
        model.cuda()
        print("done")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    vectorised_chunks = model_inputs.get("vectorised_chunks", None)
    query = model_inputs.get("query", None)
    print(query)
    query_embedding = model.encode(query)
    doc_embeddings = np.array([np.array(json.loads(chunk['vector'])).astype(np.float32) for chunk in vectorised_chunks])
    hits = util.semantic_search(torch.from_numpy(query_embedding), torch.from_numpy(doc_embeddings), top_k=5)
    print(hits)
    return [vectorised_chunks[hit['corpus_id']] for hit in hits[0]]