import torch
import torch.nn.functional as F
import numpy as np
import string
import nltk
import faiss
from langdetect import detect
import os
# from dotenv import load_dotenv
# load_dotenv()
from typing import Dict, List, Tuple
from flask import Flask, jsonify, request, json


class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        # допишите ваш код здесь
        return np.exp(-(x - self.mu)**2 / (2 * (self.sigma ** 2)))


class KNRM(torch.nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, mlp, freeze_embeddings: bool = True, kernel_num: int = 21,
                 sigma: float = 0.1, exact_sigma: float = 0.001,
                 out_layers: List[int] = [10, 5]):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze_embeddings,
            padding_idx=0
        )

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers
        self.kernels = self._get_kernels_layers()
        self.mlp = mlp
        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        kernels = torch.nn.ModuleList()
        # допишите ваш код здесь
        mus = []
        step = 2 / (self.kernel_num - 1)
        mu = -((self.kernel_num // 2) - 0.5) * step
        while mu < 1:
            mus.append(mu)
            kernels.append(GaussianKernel(mu, self.sigma))
            mu += step
        kernels.append(GaussianKernel(1, self.exact_sigma))
        mus.append(1)
        # print(mus)
        return kernels

    def forward(self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        query = self.embeddings(query)
        doc = self.embeddings(doc)
        return torch.einsum('bik, bjk -> bij', F.normalize(query, p=2, dim=2), F.normalize(doc, p=2, dim=2))

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        # shape = [Batch, Left], [Batch, Right]
        query, doc = inputs['query'], inputs['document']

        # shape = [Batch, Left, Right]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape = [Batch, Kernels]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape = [Batch]
        out = self.mlp(kernels_out)
        return out


class Solution:
    def __init__(self, emb_path_glove=os.environ['EMB_PATH_GLOVE'],
                 emb_path_knrm=os.environ['EMB_PATH_KNRM'],
                 vocab_path=os.environ['VOCAB_PATH'],
                 mlp_path=os.environ['MLP_PATH'],
                 num_cluster_centroids=8192,
                 num_candidates=50):
        # self.glove_vectors_path = os.environ['EMB_PATH_GLOVE']
        # time.sleep(10)
        self.embs = self._read_glove_embeddings(emb_path_glove)
        self.krnm_embs = torch.load(emb_path_knrm)
        self.index = None
        self.text_vocab = {}
        with open(vocab_path) as f:
            self.vocab = json.load(f)
        self.krnm_mpl = torch.load(mlp_path)
        self.krnm_model = KNRM(self.krnm_embs['weight'], mlp=self.krnm_mpl)
        self.num_candidates = num_candidates
        self.num_cluster_centroids = num_cluster_centroids

    def _read_glove_embeddings(self, file_path: str) -> Dict[str, List[str]]:
        # допишите ваш код здесь
        embs = {}
        with open(f"{file_path}", "r", encoding='utf-8') as f:
            for line in f:
                fields = line.split()
                embs[fields[0]] = np.float32(fields[1:])
        self.len_embeddings = len(fields[1:])
        return embs

    def handle_punctuation(self, inp_str: str) -> str:
        for s in string.punctuation:
            inp_str = inp_str.replace(s, ' ')
        return inp_str

    def preproc_func(self, inp_str: str) -> List[str]:
        # допишите ваш код здесь
        inp_str = self.handle_punctuation(inp_str).lower()
        return nltk.word_tokenize(inp_str)

    def _vectorize_text(self, text: str) -> np.array:
        tokens = self.preproc_func(text)
        if len(tokens) == 0:
            return np.zeros(50, dtype=np.float32)
        else:
            return np.array([self.embs.get(token, np.zeros(50)) for token in tokens]).mean(axis=0).astype(np.float32)

    def vectorize_corpus(self, documents: Dict[str, str]) -> Tuple[np.array, Dict[int, str]]:
        text_vector = []
        text_vocab = {}
        for idx, (key, value) in enumerate(documents.items()):
             text_vector.append(self._vectorize_text(value))
             text_vocab[idx] = [key, value]
        return np.array(text_vector), text_vocab

    def train_faiss_index(self, documents: Dict[str, str]):
        text_vector, self.text_vocab = self.vectorize_corpus(documents)
        # print(text_vector.shape)
        self.index = faiss.IndexFlatL2(text_vector.shape[1])
        self.index.train(text_vector)
        self.index.add(text_vector)
        del text_vector

    def find_kmins(self, doc: str, k: int):
        vectorized_doc = self._vectorize_text(doc).reshape(1, -1)
        l2, idxs = self.index.search(vectorized_doc, k=k)
        # return idxs
        mask = idxs != -1
        candidates = [self.text_vocab[idx] for idx in idxs[mask]]
        return np.array(candidates)

    def _tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        # допишите ваш код здесь
        return [self.vocab.get(word, self.vocab['OOV']) for word in tokenized_text]

    def _convert_text_to_token_idxs(self, text: str, max_len: int = 30):
        text = self.preproc_func(text)[:max_len]
        text = self._tokenized_text_to_index(text)
        return text

    def create_ranking_batch(self,  question: str, candidates: np.array(str)):
        # candidates = self.find_kmeans(doc=question, k=num_candidates)
        preproc_candidates = [self._convert_text_to_token_idxs(candidate) for candidate in candidates]
        preproc_question = self._convert_text_to_token_idxs(question)
        padded_candidates = []
        questions = [preproc_question] * len(candidates)
        c_len_max = 0
        for elem in preproc_candidates:
            c_len_max = max(c_len_max, len(elem))
        for elem in preproc_candidates:
            c_len = len(elem)
            pad_len_c = max(c_len_max-c_len, 0)
            padded_candidates.append(elem + [0] * pad_len_c)
        batch = {'query': torch.LongTensor(questions),
                 'document': torch.LongTensor(padded_candidates)
                 }
        return batch

    def rank(self, query: List[str]):
        queries = query
        dct_answer = {
                      'query': [],
                      'same_queries': [],
                      'lang_check': []
                      }
        with torch.no_grad():
            for question in queries:
                lang_check = detect(question) == "en"
                if lang_check:
                    candidates = self.find_kmins(doc=question, k=self.num_candidates)
                    batch = self.create_ranking_batch(question, candidates[:, 1])
                    scores = self.krnm_model.predict(batch)
                    top_docs_idx = torch.argsort(scores.reshape(-1), descending=True).numpy()[:10]
                    # print(candidates[top_docs_idx])
                    dct_answer['query'].append(question)
                    dct_answer['same_queries'].append(candidates[top_docs_idx].tolist())
                    dct_answer['lang_check'].append(detect(question) == "en")
                else:
                    dct_answer['query'].append(question)
                    dct_answer['same_queries'].append(None)
                    dct_answer['lang_check'].append(detect(question) == "en")
        return dct_answer


app = Flask(__name__)

s = Solution()

@app.route('/ping')
def ping():
    return jsonify(status='ok')

@app.route('/query', methods=['POST'])
def query():
    global s
    if isinstance(s.index, type(None)):
        return jsonify(status='FAISS is not initialized!')
    else:
        queries = json.loads(request.json)
        # print(request.json['queries'])
        answer = s.rank(queries['queries'])
        return jsonify(lang_check=answer['lang_check'],
                       suggestions=answer['same_queries'])

@app.route('/update_index', methods=['POST'])
def update_index():
    global s
    documents = json.loads(request.json)['documents']
    s.train_faiss_index(documents)
    del documents
    return jsonify(status='ok',
                   index_size=s.index.ntotal
                   )


if __name__ == '__main__':
    app.run(port=11000)