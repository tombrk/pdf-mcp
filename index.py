import warnings
import arguably
from pathlib import Path
import sys


@arguably.command
def main(
    *files: Path,
    state: Path = Path.home() / ".local" / "share" / "pdf-mcp",
    model: str = "BAAI/bge-small-en-v1.5",
):
    warnings.filterwarnings("ignore")

    from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
    from llama_index.node_parser.docling import DoclingNodeParser
    from llama_index.readers.docling import DoclingReader
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.milvus import MilvusVectorStore

    embed_model = HuggingFaceEmbedding(model_name=model)
    embed_dim = len(embed_model.get_text_embedding(""))
    state.mkdir(parents=True, exist_ok=True)
    vector_store = MilvusVectorStore(
        str(state / "milvus.db"), overwrite=False, dim=embed_dim
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    pdf_reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)

    input_files = []
    for f in files:
        if not vector_store.client.query(
            vector_store.collection_name,
            limit=1,
            output_fields=["file_name"],
            filter=f"file_path == {f.absolute()}",
        ):
            input_files.append(f)
        else:
            print(f"skip {f}, already exist", file=sys.stderr)

    dir_reader = SimpleDirectoryReader(
        recursive=False,
        file_extractor={".pdf": pdf_reader},
        input_files=input_files,
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
