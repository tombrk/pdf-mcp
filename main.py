import arguably
from llama_index.core import StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from fastmcp import FastMCP
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
import uvicorn
import pymupdf
import io
import asyncio
import hashlib


class Passage(BaseModel):
    id: str
    text: str = Field(description="Plain text extracted from the PDF")
    file: str = Field(description="Original PDF file this came from")
    path: str = Field(exclude=True)
    page: int = Field(description="Page in the PDF")
    distance: Optional[float] = Field(
        description="Vector distance of the query to this match"
    )
    section: str = Field(description="The subsection this passage appeared under")
    ref: str = Field(exclude=True)
    bbox: dict = Field(exclude=True)

    def from_chunk(chunk: dict):
        print(chunk)
        meta = chunk["doc_items"][0]
        return Passage(
            file=chunk["file_name"],
            path=chunk["file_path"],
            text=chunk["text"],
            distance=chunk.get("distance") or None,
            id=chunk["id"],
            section=",".join(chunk["headings"]),
            page=meta["prov"][0]["page_no"],
            bbox=meta["prov"][0]["bbox"],
            ref=meta["self_ref"],
        )


@arguably.command
def main(
    docdir: Path,
    model: str = "nomic-embed-text",
):
    # Operate on a document directory; store state in ./.mcp within it
    state = (docdir / ".mcp").absolute()

    embed_model = OllamaEmbedding(model)
    embed_dim = len(embed_model.get_text_embedding("abc"))

    state.mkdir(parents=True, exist_ok=True)
    vector_store = MilvusVectorStore(
        str(state / "milvus.db"), overwrite=False, dim=embed_dim
    )
    StorageContext.from_defaults(vector_store=vector_store)
    vector_store.client.query(
        vector_store.collection_name, limit=1, output_fields=["file_name"]
    )

    mcp = FastMCP("PDF Papers")

    @mcp.tool
    def search(query: str, top_k: int = 10) -> list[Passage]:
        """Search for all queries in pdf sources using vector similarity matching"""
        embeddings = embed_model.get_text_embedding_batch([query])
        res = vector_store.client.search(
            vector_store.collection_name,
            embeddings,
            limit=top_k,
            output_fields=[
                "text",
                "file_name",
                "file_path",
                "distance",
                "doc_items",
                "headings",
            ],
        )
        return list(map(Passage.from_chunk, res[0]))

    @mcp.tool
    def read_section(section: str) -> list[Passage]:
        """Read an entire section by name as previously seen in the search output"""
        res = vector_store.client.query(
            vector_store.collection_name, filter=f'headings[0] == "{section}"'
        )
        return sorted(list(map(Passage.from_chunk, res)), key=lambda x: x.ref)

    @mcp.tool
    def read_file(file_name: str) -> list[Passage]:
        """Read an entire document by its file name, as previously seen in the search output"""
        res = vector_store.client.query(
            vector_store.collection_name, filter=f'file_name == "{file_name}"'
        )
        return sorted(list(map(Passage.from_chunk, res)), key=lambda x: x.ref)

    mcp_app = mcp.http_app(path="/")
    app = FastAPI(title="PDF MCP", lifespan=mcp_app.lifespan)
    app.mount("/mcp", mcp_app)

    def resolve_path(p: Path) -> Path:
        try:
            return p.resolve()
        except FileNotFoundError:
            # If the file does not exist yet, fall back to absolute()
            return p.absolute()

    def is_under_docdir(path: Path) -> bool:
        try:
            return resolve_path(path).is_relative_to(resolve_path(docdir))
        except AttributeError:
            # Python < 3.9 compatibility guard (is_relative_to not available)
            try:
                resolve_path(path).relative_to(resolve_path(docdir))
                return True
            except Exception:
                return False

    def iter_pdf_files_recursively(root: Path):
        for file_path in root.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() == ".pdf":
                yield file_path

    def file_row_id(file_path: Path) -> str:
        h = hashlib.md5(str(resolve_path(file_path)).encode("utf-8")).hexdigest()
        return f"row-{h}"

    def is_indexed(file_path: Path) -> bool:
        abs_path = str(resolve_path(file_path))
        res = vector_store.client.query(
            vector_store.collection_name,
            limit=1,
            output_fields=["file_name"],
            filter=f'file_path == "{abs_path}"',
        )
        return bool(res)

    def render_row_html(file_path: Path) -> str:
        row_id = file_row_id(file_path)
        rel = str(resolve_path(file_path).relative_to(resolve_path(docdir))) if is_under_docdir(file_path) else str(resolve_path(file_path))
        indexed = is_indexed(file_path)
        status_html = (
            '<span style="color: #16a34a;">Indexed</span>'
            if indexed
            else '<span style="color: #dc2626;">Not indexed</span>'
        )
        btn_label = "Reindex" if indexed else "Index"
        action_html = (
            f"<button hx-post=\"/index-file\" hx-vals='{{\"path\": \"{str(resolve_path(file_path))}\"}}' "
            f"hx-target=\"#{row_id}\" hx-swap=\"outerHTML\">{btn_label}</button>"
        )
        return (
            f"<tr id=\"{row_id}\">"
            f"<td style=\"padding: 6px 10px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace;\">{rel}</td>"
            f"<td style=\"padding: 6px 10px;\">{status_html}</td>"
            f"<td style=\"padding: 6px 10px;\">{action_html}</td>"
            f"</tr>"
        )

    def delete_entries_for_file(file_path: Path) -> None:
        abs_path = str(resolve_path(file_path))
        try:
            vector_store.client.delete(
                vector_store.collection_name,
                filter=f'file_path == "{abs_path}"',
            )
        except Exception:
            # If delete is unsupported, ignore and proceed (may duplicate)
            pass

    async def index_single_file(file_path: Path, *, force: bool = False):
        if is_indexed(file_path) and not force:
            return
        from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
        from llama_index.node_parser.docling import DoclingNodeParser
        from llama_index.readers.docling import DoclingReader

        pdf_reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        dir_reader = SimpleDirectoryReader(
            input_files=[file_path],
            file_extractor={".pdf": pdf_reader},
            required_exts=[".pdf"],
        )

        # Run the heavy indexing work in a thread to avoid blocking the event loop
        def _do_index():
            if force:
                delete_entries_for_file(file_path)
            VectorStoreIndex.from_documents(
                documents=dir_reader.load_data(),
                transformations=[DoclingNodeParser()],
                embed_model=embed_model,
                show_progress=False,
                storage_context=storage_context,
            )

        await asyncio.to_thread(_do_index)

    @app.get("/", response_class=HTMLResponse)
    def ui_home():
        rows_html = "\n".join(
            render_row_html(p) for p in sorted(iter_pdf_files_recursively(docdir), key=lambda x: str(x).lower())
        )
        return (
            "<!doctype html>"
            "<html><head>"
            "<meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<title>PDF Index Manager</title>"
            "<script src=\"https://unpkg.com/htmx.org@1.9.12\" integrity=\"sha384-+DR3Eo8Vd7i8xWZt0+kq5L9TxlGmSg1H5dV3r0C6i8qA3wqV+Q3xLrLPB0hKq1uE\" crossorigin=\"anonymous\"></script>"
            "<style>body{font-family: ui-sans-serif, system-ui, -apple-system, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, \"Noto Sans\", \"Apple Color Emoji\", \"Segoe UI Emoji\"; padding:20px;}"
            "table{border-collapse: collapse; width:100%;} th,td{border-bottom:1px solid #e5e7eb;} th{text-align:left; color:#374151;} thead tr{background:#f9fafb;} button{background:#2563eb;color:#fff;border:none;border-radius:6px;padding:6px 10px;cursor:pointer;} button:hover{background:#1d4ed8;} </style>"
            "</head><body>"
            f"<h2 style=\"margin: 0 0 10px;\">PDF Index Manager</h2>"
            f"<div style=\"color:#6b7280; margin-bottom: 16px;\">Root: {resolve_path(docdir)}</div>"
            "<table>"
            "<thead><tr><th style=\"padding: 8px 10px;\">File</th><th style=\"padding: 8px 10px;\">Status</th><th style=\"padding: 8px 10px;\">Action</th></tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            "</table>"
            "</body></html>"
        )

    @app.post("/index-file", response_class=HTMLResponse)
    async def index_file(request: Request):
        # Accept JSON, x-www-form-urlencoded, or query param
        path: Optional[str] = None
        try:
            data = await request.json()
            if isinstance(data, dict):
                path = data.get("path")
        except Exception:
            path = None
        if not path:
            try:
                body_bytes = await request.body()
                if body_bytes:
                    from urllib.parse import parse_qs
                    params = parse_qs(body_bytes.decode("utf-8"))
                    path = params.get("path", [None])[0]
            except Exception:
                path = None
        if not path:
            path = request.query_params.get("path")
        if not path:
            raise HTTPException(status_code=400, detail="Missing 'path'")

        file_path = Path(path)
        if not is_under_docdir(file_path):
            raise HTTPException(status_code=400, detail="Path is outside of document root")
        if not file_path.exists() or file_path.suffix.lower() != ".pdf":
            raise HTTPException(status_code=404, detail="PDF not found")

        await index_single_file(file_path, force=True)
        return render_row_html(file_path)

    @app.get("/{id}.json")
    def passage(id: str):
        res = vector_store.client.get(vector_store.collection_name, ids=[id])
        if not len(res) == 1:
            raise HTTPException(status_code=404)
        return {"llm": Passage.from_chunk(res[0]), "doc": res[0]}

    @app.get("/{id}")
    def inspect(id: str):
        res = vector_store.client.get(vector_store.collection_name, ids=[id])
        if not len(res) == 1:
            raise HTTPException(status_code=404)
        p = Passage.from_chunk(res[0])

        pdf = pymupdf.open(p.path)

        page_height = pdf[p.page - 1].rect.height
        x0 = p.bbox["l"]
        x1 = p.bbox["r"]
        y0 = page_height - p.bbox["t"]
        y1 = page_height - p.bbox["b"]
        rect = pymupdf.Rect(x0, y0, x1, y1)

        color = (1.0, 0.0, 0.0)

        out_buf = io.BytesIO()
        out_pdf = pymupdf.open()
        out_pdf.insert_pdf(pdf, from_page=p.page - 1, to_page=p.page - 1)
        out_page = out_pdf[0]
        out_page.draw_rect(rect, color=color, width=0.7)
        out_pdf.save(out_buf)
        out_pdf.close()
        pdf.close()
        out_buf.seek(0)
        return Response(content=out_buf.getvalue(), media_type="application/pdf")

    uvicorn.run(app, host="0.0.0.0", port=7777)


if __name__ == "__main__":
    arguably.run()
