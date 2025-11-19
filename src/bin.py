import asyncio

class Bin:
    def __init__(self, max_batch_size: int = 6,
                 max_wait_secs: float = 0.5):
        self._queue = []
        self._shutdown = False
        self._lock = asyncio.Lock()
        self._flush_event = asyncio.Event()
        self._max_batch_size = max_batch_size
        self._max_wait_secs = max_wait_secs
        self._task = asyncio.create_task(self._batch_loop())

    async def set_batch_size(self, batch_size):
        self._max_batch_size = batch_size
        #TODO: Add logic that forces batch to be processed if num items > new batch size

    async def set_wait_secs(self, wait_secs):
        self._max_wait_secs = wait_secs
        #TODO: Ditto


    async def shutdown(self):
        self._task.cancel()
        self._flush_event.set()
        self._shutdown = True
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    async def add_query(self, query):
        async with self._lock:
            self._queue.append(query)
        self._flush_event.set()  # signal loop to process immediately

    async def _batch_loop(self):
        while not self._shutdown:
            try:
                await asyncio.wait_for(self._flush_event.wait(), timeout=2)
            except asyncio.TimeoutError:
                pass
            self._flush_event.clear()
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

        results = await self._batch_vector_db_query(texts, embs)

        for fut, res in zip(futures, results):
            if not fut.done():
                fut.set_result(res)

    async def _batch_vector_db_query(self, texts, embs):
        #TODO: Add code for accessing database here, just adding artificial wait
        # for demo purposes
        await asyncio.sleep(0.1)
        return [f"Result for '{t}'" for t in texts]