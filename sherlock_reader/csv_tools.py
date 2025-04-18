#===============================================================================================
# Project: Predicting Commercial Flight Trajectories Using Transformers for CS 555
# Author(s): Luke, Alex
# Description: Lightweight CSV reader with optional header skipping and peeking support.
#===============================================================================================


class CSVReader:
    """
    A simple CSV file reader that supports iteration, header skipping, and peeking the next row.
    """

    def __init__(self, path: str, has_header: bool=True):
        """
        Initializes the CSVReader.

        Args:
            path (str): Path to the CSV file.
            has_header (bool): If True, the first line of the file will be skipped.
        """
        self.path = path
        self.next_item = None
        self.has_header = has_header

    def __iter__(self):
        return self
    
    def __next__(self):
        """
        Retrieves the next line in the file, splitting it into a list of strings.

        Returns:
            list[str]: The next parsed line.

        Raises:
            StopIteration: When the end of the file is reached.
        """
        if self.next_item is not None:
            item = self.next_item
            self.next_item = self.__advance()
            return item
        else:
            return self.__advance()

    def __peek__(self):
        """
        Peeks at the next line in the file without advancing the iterator.

        Returns:
            list[str]: The next parsed line.
        """
        if self.next_item is not None:
            return self.next_item
        else:
            self.next_item = self.__advance()
            return self.next_item
    
    def __advance(self) -> list[str]:
        """
        Reads and returns the next line from the file.

        Returns:
            list[str]: The next line split by commas.

        Raises:
            StopIteration: When the end of the file is reached.
        """
        line = next(self.file_iter)
        if line:
            return line.strip().split(',')
        else:
            raise StopIteration
    
    def __enter__(self):
        """
        Context manager entry. Opens the file and skips the header if applicable.

        Returns:
            CSVReader: Self, with file opened.
        """
        self.open()

        # Skip the header, if applicable
        if self.has_header:
            self.__advance()

        return self
    
    def __exit__(self, exc_type, exc_value, tb):
        """
        Context manager exit. Closes the file.
        """
        self.close()
    
    def open(self):
        """
        Opens the CSV file for reading.
        """
        self.inner_file = open(self.path, 'r')
        self.file_iter = iter(self.inner_file)
    
    def close(self):
        """
        Closes the CSV file.
        """
        self.inner_file.close()


def read_csv(path: str, has_header: bool=True):
    """
    Factory function to create a CSVReader.

    Args:
        path (str): Path to the CSV file.
        has_header (bool): Whether the CSV has a header row to skip.

    Returns:
        CSVReader: A new CSVReader instance.
    """
    return CSVReader(path, has_header=has_header)


def peek(peekable):
    """
    Peeks at the next element of a CSVReader without advancing it.

    Args:
        peekable (CSVReader): The reader to peek from.

    Returns:
        list[str]: The next row.
    """
    return peekable.__peek__()