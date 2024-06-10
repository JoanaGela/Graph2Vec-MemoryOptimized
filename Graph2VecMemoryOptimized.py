######### Graph2Vec-MemoryOptimized #############

import numpy as np
import networkx as nx
from typing import List
from karateclub.estimator import Estimator
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from karateclub.utils.treefeatures import WeisfeilerLehmanHashing
import zlib
import pickle

class Graph2VecMemoryOptimized(Estimator):
    def __init__(
        self,
        wl_iterations: int = 2,
        attributed: bool = False,
        dimensions: int = 128,
        workers: int = 4,
        down_sampling: float = 0.0001,
        epochs: int = 10,
        learning_rate: float = 0.025,
        min_count: int = 5,
        seed: int = 42,
        erase_base_features: bool = False,
    ):
        self.wl_iterations = wl_iterations
        self.attributed = attributed
        self.dimensions = dimensions
        self.workers = workers
        self.down_sampling = down_sampling
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed
        self.erase_base_features = erase_base_features

    def fit(self, base_graph: nx.DiGraph, modified_attributes: List[List[int]]):
        """
        Fitting a Graph2VecMemoryOptimized model.

        Arg types:
            * **base_graph** *(NetworkX graph)* - The base graph for all graphs.
            * **modified_attributes** *(List of lists)* - The list of attributes to be modified.
              Each inner list contains the indexes of nodes whose 'features' attribute will be modified to '1'. They are the diagnoses each patient has
        """
        self._set_seed()
        comp_documents = []

        for node_indexes in modified_attributes:
            graph = base_graph.copy()
            for node_index in node_indexes:
                graph.nodes[node_index]['feature'] = '1' # modifes the 'feature' attribute of the specified nodes
            
            graph = self._check_graph(graph)
            wl_hashing = WeisfeilerLehmanHashing(
                graph, self.wl_iterations, self.attributed, self.erase_base_features
            )
            compressed_wl_hashing = zlib.compress(pickle.dumps(wl_hashing)) 
            comp_documents.append(compressed_wl_hashing)
            
        documents = []

        for i, compressed_doc in enumerate(comp_documents):

            decompressed_doc = zlib.decompress(compressed_doc)
            doc = pickle.loads(decompressed_doc)

            processed_features = doc.get_graph_features()
            tag = str(i)

            tagged_doc = TaggedDocument(words=processed_features, tags=[tag])
            documents.append(tagged_doc)
            

        del comp_documents

        
        self.model = Doc2Vec(
            documents,
            vector_size=self.dimensions,
            window=0,
            min_count=self.min_count,
            dm=0,
            sample=self.down_sampling,
            workers=self.workers,
            epochs=self.epochs,
            alpha=self.learning_rate,
            seed=self.seed,
        )

        #self._embedding = [self.model.docvecs[str(i)] for i, _ in enumerate(documents)] codi original
        self._embedding = []
        c=0
        for i, _ in enumerate(documents):
            embedding = self.model.docvecs[str(i)]
            self._embedding.append(embedding)
            c += 1 
            if c % 100 == 0:
                print(f'c:{c}') #Veuré quants embeddings faig

    def get_embedding(self) -> np.array:
        """
        Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)

