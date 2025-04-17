import faiss
import numpy as np
D = 384
index = faiss.IndexFlatIP(D)
index.add(np.random.random((10, D)).astype('float32'))
print("FAISS index works, number of vectors:", index.ntotal)
