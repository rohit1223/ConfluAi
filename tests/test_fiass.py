import faiss
import numpy as np
d = 384
index = faiss.IndexFlatIP(d)
x = np.random.random((10, d)).astype('float32')
index.add(x)
print("FAISS index works, number of vectors:", index.ntotal)
