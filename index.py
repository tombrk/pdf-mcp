import warnings
import arguably
from pathlib import Path
from sys import stderr
import time


@arguably.command
def main(
    *files: Path,
    state: Path = Path.home() / ".local" / "share" / "pdf-mcp",
    model: str = "nomic-embed-text",
):
    warnings.filterwarnings("ignore")

    from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex

    start = time.time()
    from llama_index.embeddings.ollama import OllamaEmbedding

    embed_model = OllamaEmbedding(model)
    embed_dim = len(embed_model.get_text_embedding("abc"))
    print(f"load: ollama {time.time() - start}", file=stderr)

    start = time.time()
    from llama_index.vector_stores.milvus import MilvusVectorStore

    state.mkdir(parents=True, exist_ok=True)
    vector_store = MilvusVectorStore(
        str(state / "milvus.db"), overwrite=False, dim=embed_dim
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    print(f"load: milvus {time.time() - start}", file=stderr)

    start = time.time()
    from llama_index.node_parser.docling import DoclingNodeParser
    from llama_index.readers.docling import DoclingReader

    pdf_reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
    print(f"load: docling {time.time() - start}", file=stderr)

    input_files = []
    for f in files:
        if not vector_store.client.query(
            vector_store.collection_name,
            limit=1,
            output_fields=["file_name"],
            filter=f'file_path == "{f.absolute()}"',
        ):
            input_files.append(f)
        else:
            print(f"skip {f}, already exist", file=stderr)

    if not input_files:
        return

    dir_reader = SimpleDirectoryReader(
        input_files=input_files,
        file_extractor={".pdf": pdf_reader},
        required_exts=[".pdf"],
    )

    VectorStoreIndex.from_documents(
        documents=dir_reader.load_data(),
        transformations=[DoclingNodeParser()],
        embed_model=embed_model,
        show_progress=True,
        storage_context=storage_context,
    )


if __name__ == "__main__":
    arguably.run()
