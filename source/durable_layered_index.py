import asyncio
from query import Query

class DurableLayeredIndex:
    def __init__(self, max_batch_size: int = 10,
                 max_wait_secs: float = 0.5,
                 grouping_threshold: float = 0.9):
        self.max_batch_size = max_batch_size
        self.max_wait_secs = max_wait_secs
        self.grouping_threshold = grouping_threshold

        self._queue = []
        self._lock = asyncio.Lock()
        self._task = asyncio.create_task(self._batch_loop())

    async def query(self, query_text: str):
        future = asyncio.get_running_loop().create_future()
        q = Query(query_text, future)

        async with self._lock:
            self._queue.append(q)
            if len(self._queue) >= self.max_batch_size:
                # cancel the waiting task so we flush now
                self._task.cancel()
                # start a new task for next batch
                self._task = asyncio.create_task(self._batch_loop())

        return await future

    async def _batch_loop(self):
        try:
            await asyncio.sleep(self.max_wait_secs)
        except asyncio.CancelledError:
            # we cancelled early because of batch_size
            pass
        await self._process_batch()

    async def _process_batch(self):
        async with self._lock:
            batch = self._queue
            self._queue = []
        if not batch:
            return

        texts = [q.get_text() for q in batch]
        embs = [q.get_embedding() for q in batch]
        futures = [q.get_future() for q in batch]

        # TODO: Add logic for reordering queries

        results = await self._batch_vector_db_query(texts, embs)

        for fut, res in zip(futures, results):
            if not fut.done():
                fut.set_result(res)

    async def _batch_vector_db_query(self, texts, embs):
        #TODO: Add code for accessing database here, just adding artificial wait
        # for demo purposes
        await asyncio.sleep(0.1)
        return [f"Result for '{t}'" for t in texts]

async def main():
    dli = DurableLayeredIndex(max_batch_size=5, max_wait_secs=0.3)
    queries = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
    tasks = [asyncio.create_task(dli.query(q)) for q in queries]
    results = await asyncio.gather(*tasks)
    print(results)

asyncio.run(main())
