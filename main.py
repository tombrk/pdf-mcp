from llama_index.core.schema import Document
import arguably
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
import json
import asyncio
from fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path


class Passage(BaseModel):
    id: str
    text: str = Field(description="Plain text extracted from the PDF")
    file: str = Field(description="Original PDF file this came from")
    page: int = Field(description="Page in the PDF")
    distance: Optional[float] = Field(
        description="Vector distance of the query to this match"
    )
    section: str = Field(description="The subsection this passage appeared under")
    ref: str = Field(exclude=True)

    def from_chunk(chunk: dict):
        meta = chunk["doc_items"][0]
        return Passage(
            file=chunk["file_name"],
            text=chunk["text"],
            distance=chunk.get("distance") or None,
            id=chunk["id"],
            section=",".join(chunk["headings"]),
            page=meta["prov"][0]["page_no"],
            ref=meta["self_ref"],
        )


@arguably.command
def main(dir: str,
         state: Path = Path.home() / ".local" / "share" / "pdf-mcp",
         model: str = "BAAI/bge-small-en-v1.5",
    ):
    embed_model = HuggingFaceEmbedding(model_name=model)
    embed_dim = len(embed_model.get_text_embedding(""))

    state.mkdir(parents=True, exist_ok=True)
    vector_store = MilvusVectorStore(str(state / "milvus.db"), overwrite=False, dim=embed_dim)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    coll = vector_store.client.query(
        vector_store.collection_name, limit=16384, output_fields=["file_name"]
    )

    exclude = list({x["file_name"] for x in coll})
    pdf_reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)

    try:
        dir_reader = SimpleDirectoryReader(
            recursive=True,
            file_extractor={".pdf": pdf_reader},
            input_dir=dir,
            exclude=exclude,
            required_exts=[".pdf"],
        )
        docs = dir_reader.load_data(num_workers=4)
    except ValueError as e:
        if not str(e).startswith("No files found in"):
            raise
        docs: list[Document] = []


    VectorStoreIndex.from_documents(
        documents=docs,
        transformations=[DoclingNodeParser()],
        embed_model=embed_model,
        show_progress=True,
        storage_context=storage_context,
    )

    mcp = FastMCP("PDF Papers")

    @mcp.tool
    def pdf_search(query: str, top_k: int = 10) -> list[Passage]:
        """Search for all queries in pdf sources using vector similarity matching"""
        embeddings = embed_model.get_text_embedding_batch([query])
        res = vector_store.client.search(
            vector_store.collection_name,
            embeddings,
            limit=top_k,
            output_fields=["text", "file_name", "doc_items", "headings"],
        )
        return list(map(Passage.from_chunk, res[0]))

    @mcp.tool
    def pdf_section(section: str) -> list[Passage]:
        """Read an entire section by name as previously seen in the search output"""
        res = vector_store.client.query(
            vector_store.collection_name, filter=f'headings[0] == "{section}"'
        )
        return sorted(list(map(Passage.from_chunk, res)), key=lambda x: x.ref)

    mcp.run()


if __name__ == "__main__":
    arguably.run()
