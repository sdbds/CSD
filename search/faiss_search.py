import logging
import faiss
import numpy as np

module_logger = logging.getLogger(__name__)


class FaissIndex(object):
    def __init__(self, index_func=faiss.IndexFlatIP, embedding_size=512*512):
        self.index = index_func(embedding_size)
        # Enable GPU support
        # self.index_gpu = faiss.index_cpu_to_all_gpus(self.index)

    def build_index(self, nodes):
        # Ensure nodes is a NumPy array of type float32
        if not isinstance(nodes, np.ndarray):
            nodes = np.array(nodes, dtype=np.float32)
        elif nodes.dtype != np.float32:
            nodes = nodes.astype(np.float32)
        self.index.add(nodes)
        # Enable GPU support
        # self.index_gpu.add(nodes)

    def search_nns(self, embeddings, n):
        # Ensure embeddings is a NumPy array of type float32
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)
        elif embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        # Enable GPU support
        # return self.index_gpu.search(embeddings, n)
        return self.index.search(embeddings, n)
