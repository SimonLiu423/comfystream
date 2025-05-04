class ImageChunkBuffer:
    def __init__(self, total_chunks: int = 10):
        self.total_chunks = total_chunks
        self.chunks = []

    def add_chunk(self, chunk_data: str):
        self.chunks.append(chunk_data)
        if len(self.chunks) == self.total_chunks:
            return True
        return False

    def get_complete_image(self):
        return "".join(self.chunks)

    def clear(self):
        self.chunks = []
