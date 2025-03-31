class CSVReader:
    def __init__(self, path: str, has_header: bool=True):
        self.path = path
        self.next_item = None
        self.has_header = has_header

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.next_item is not None:
            item = self.next_item
            self.next_item = self.__advance()
            return item
        else:
            return self.__advance()

    def __peek__(self):
        if self.next_item is not None:
            return self.next_item
        else:
            self.next_item = self.__advance()
            return self.next_item
    
    def __advance(self) -> list[str]:
        line = next(self.file_iter)
        if line:
            return line.strip().split(',')
        else:
            raise StopIteration
    
    def __enter__(self):
        self.open()

        # Skip the header, if applicable
        if self.has_header:
            self.__advance()

        return self
    
    def __exit__(self, exc_type, exc_value, tb):
        self.close()
    
    def open(self):
        self.inner_file = open(self.path, 'r')
        self.file_iter = iter(self.inner_file)
    
    def close(self):
        self.inner_file.close()


def read_csv(path: str, has_header: bool=True):
    return CSVReader(path, has_header=has_header)


def peek(peekable):
    return peekable.__peek__()