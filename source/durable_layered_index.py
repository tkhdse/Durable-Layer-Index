import asyncio
from bin import Bin
from query import Query
import numpy as np
import faiss


class DurableLayeredIndex:
    def __init__(self, num_bins = 1, grouping_threshold: float = 0.9):
        #Maps vectors to a bin
        self._vector_bin_index = faiss.IndexFlatL2(384)
        self._vector_bin_map = {}
        self._add_bin()

    def _add_bin(self):
        #TODO: Using random vector for now, ideally this should be a param
        bin_vector = np.random.randn(1, 384).astype('float32')
        self._vector_bin_index.add(bin_vector)
        bin_index = self._vector_bin_index.ntotal - 1
        self._vector_bin_map[bin_index] = Bin()

    async def query(self, query_text: str):
        future = asyncio.get_running_loop().create_future()
        new_query = Query(query_text, future)
        await self._add_query_to_bin(new_query)
        return await future
    
    async def _add_query_to_bin(self, new_query):
        emb = new_query.get_embedding().astype('float32').reshape(1, -1)
        distances, indices = self._vector_bin_index.search(emb, k=1)
        bin_index = indices[0][0]
        selected_bin = self._vector_bin_map[bin_index]
        await selected_bin.add_query(new_query)



    async def close(self):
        for bin in self._vector_bin_map.values():
            await bin.shutdown()


async def main():
    dli = DurableLayeredIndex()
    queries = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
    tasks = [asyncio.create_task(dli.query(q)) for q in queries]
    results = await asyncio.gather(*tasks)
    print(results)
    await dli.close()

asyncio.run(main())
