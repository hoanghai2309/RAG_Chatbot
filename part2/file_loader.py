from typing import Union, List, Literal
import glob
from tqdm import tqdm
import multiprocessing
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(pdf_file: str) -> list:
    documents = PyPDFLoader(pdf_file, extract_images=True).load()
    return documents



class PDFLoader():
    def __init__(self):
        super().__init__()

    def __call__(self, pdf_files: List[str], **kwargs):
        loaded_docs = []
        total_files = len(pdf_files)
        with tqdm(total=total_files, desc="Loading PDFs", unit="file") as pbar:
            for pdf_file in pdf_files:
                result = load_pdf(pdf_file)
                loaded_docs.extend(result)
                pbar.update(1)
        return loaded_docs

class TextSplitter:
    def __init__(self, separators: List[str] = ['\n\n', '\n', ' ', ''], chunk_size: int = 300, chunk_overlap: int = 0):
        self.splitter = RecursiveCharacterTextSplitter(separators=separators, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def __call__(self, documents: list):
        return self.splitter.split_documents(documents)

class Loader:
    def __init__(self, file_type: str = Literal["pdf"], split_kwargs: dict = {"chunk_size": 300, "chunk_overlap": 0}):
        assert file_type in ["pdf"], "file_type must be pdf"
        self.file_type = file_type
        if file_type == "pdf":
            self.doc_loader = PDFLoader()
        else:
            raise ValueError("file_type must be pdf")
        self.doc_splitter = TextSplitter(**split_kwargs)

    def load(self, pdf_files: Union[str, List[str]], workers: int = 1):
        if isinstance(pdf_files, str):
            pdf_files = [pdf_files]
        loaded_docs = self.doc_loader(pdf_files, workers=workers)
        split_docs = self.doc_splitter(loaded_docs)
        return split_docs

    def load_dir(self, dir_path: str, workers: int = 1):
        if self.file_type == "pdf":
            files = glob.glob(f"{dir_path}/*.pdf")
            assert len(files) > 0, f"No {self.file_type} files found in {dir_path}"
        else:
            raise ValueError("file_type must be pdf")
        return self.load(files, workers=workers)
