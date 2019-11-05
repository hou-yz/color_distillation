class BufferSizeCounter(object):
    def __init__(self):
        self.size = 0

    def reset(self):
        self.size = 0

    def update(self, new_buffer_size):
        self.size += new_buffer_size
