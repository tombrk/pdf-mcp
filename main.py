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
        return p.resolve()

    def is_under_docdir(path: Path) -> bool:
        return resolve_path(path).is_relative_to(resolve_path(docdir))

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
        checkbox_html = (
            ""
            if indexed
            else (
                f"<input type=\"checkbox\" class=\"pdf-checkbox\" name=\"path\" value=\"{str(resolve_path(file_path))}\" />"
            )
        )
        action_html = (
            f"<button class=\"btn btn-success\" disabled>Indexed</button>"
            if indexed
            else (
                f"<button class=\"btn\" hx-post=\"/index-file\" hx-vals='{{\"path\": \"{str(resolve_path(file_path))}\"}}' "
                f"hx-target=\"#progress\" hx-swap=\"innerHTML\">Index</button>"
            )
        )
        return (
            f"<tr id=\"{row_id}\">"
            f"<td class=\"checkbox-col\">{checkbox_html}</td>"
            f"<td class=\"path\">{rel}</td>"
            f"<td>{action_html}</td>"
            f"</tr>"
        )

    async def index_single_file(file_path: Path):
        # Do not reindex; if present, skip
        if is_indexed(file_path):
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
            VectorStoreIndex.from_documents(
                documents=dir_reader.load_data(),
                transformations=[DoclingNodeParser()],
                embed_model=embed_model,
                show_progress=False,
                storage_context=storage_context,
            )

        await asyncio.to_thread(_do_index)

    # Single active indexing job state
    job_lock = asyncio.Lock()
    current_job: dict | None = None

    def render_progress_card_html() -> str:
        nonlocal current_job
        if not current_job:
            return ""
        total = current_job.get("total", 0)
        done = current_job.get("done", 0)
        percent = int((done / total) * 100) if total else 0
        remaining = total - done
        # Progress card fixed at bottom-right
        return (
            "<div style=\"position: fixed; right: 16px; bottom: 16px; width: 320px; background: #111827; color: #e5e7eb; border: 1px solid #374151; border-radius: 10px; box-shadow: 0 10px 30px rgba(0,0,0,0.35);\">"
            "<div style=\"padding: 12px 14px; border-bottom: 1px solid #374151; display:flex; align-items:center; justify-content:space-between;\">"
            "<div style=\"font-weight: 600;\">Indexing PDFs</div>"
            f"<div style=\"font-size: 12px; color:#9ca3af;\">{done}/{total}</div>"
            "</div>"
            "<div style=\"padding: 12px 14px;\">"
            f"<div style=\"height: 8px; background:#1f2937; border-radius: 9999px; overflow:hidden;\"><div style=\"height: 100%; width: {percent}%; background: linear-gradient(90deg,#60a5fa,#22d3ee);\"></div></div>"
            f"<div style=\"margin-top: 8px; font-size: 12px; color:#9ca3af;\">{remaining} remaining</div>"
            "</div>"
            "</div>"
        )

    def render_progress_with_oob_row_updates() -> str:
        nonlocal current_job
        if not current_job:
            return ""
        # Card
        html = render_progress_card_html()
        # OOB updates for any newly finished rows
        emitted = current_job.setdefault("emitted_done", set())
        newly_done = [p for p in current_job.get("done_paths", []) if p not in emitted]
        for p in newly_done:
            emitted.add(p)
            row_html = render_row_html(Path(p))
            # Mark row to swap out-of-band
            # We need to add hx-swap-oob to the tr
            # Inject attribute into the opening <tr>
            row_html_oob = row_html.replace("<tr ", "<tr hx-swap-oob=\"outerHTML\" ", 1)
            html += row_html_oob
        return html

    async def run_index_job(paths: list[Path]):
        nonlocal current_job
        async with job_lock:
            current_job = {
                "total": len(paths),
                "done": 0,
                "paths": [str(p) for p in paths],
                "done_paths": [],
                "emitted_done": set(),
                "status": "running",
            }
            try:
                for p in paths:
                    await index_single_file(p)
                    current_job["done"] += 1
                    current_job["done_paths"].append(str(resolve_path(p)))
            finally:
                # Mark complete; keep current_job so progress endpoint can clear it
                current_job["status"] = "done"

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
            "<script src=\"https://unpkg.com/htmx.org@2.0.6\"></script>"
            "<style>"
            ":root{--bg:#ffffff;--text:#111827;--muted:#6b7280;--border:#e5e7eb;--thead-bg:#f9fafb;--primary:#2563eb;--primary-hover:#1d4ed8;--secondary:#374151;--secondary-hover:#303846;--success:#16a34a;--card-bg:#111827;--card-text:#e5e7eb;--card-border:#374151;--track:#1f2937;}"
            "*{box-sizing:border-box;}"
            "body{font-family: ui-sans-serif, system-ui, -apple-system, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, \"Noto Sans\", \"Apple Color Emoji\", \"Segoe UI Emoji\"; padding:20px; color:var(--text); background:var(--bg);}"
            "h2{margin:0 0 10px;}"
            ".muted{color:var(--muted);}"
            ".root-note{margin-bottom:12px;}"
            ".toolbar{display:flex; gap:8px; align-items:center; margin-bottom:10px;}"
            "table{border-collapse:collapse; width:100%;}"
            "th,td{border-bottom:1px solid var(--border); padding:8px 10px;}"
            "th{text-align:left; color:#374151; background:var(--thead-bg);}"
            ".checkbox-col{width:28px;}"
            ".path{font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace;}"
            ".btn{background:var(--primary); color:#fff; border:none; border-radius:6px; padding:6px 10px; cursor:pointer;}"
            ".btn:hover{background:var(--primary-hover);}"
            ".btn-secondary{background:var(--secondary);}"
            ".btn-secondary:hover{background:var(--secondary-hover);}"
            ".btn-success{background:var(--success); cursor:default; opacity:.9;}"
            ".btn-success[disabled]{cursor:default;}"
            ".progress-card{position:fixed; right:16px; bottom:16px; width:320px; background:var(--card-bg); color:var(--card-text); border:1px solid var(--card-border); border-radius:10px; box-shadow:0 10px 30px rgba(0,0,0,0.35);}"
            ".progress-header{padding:12px 14px; border-bottom:1px solid var(--card-border); display:flex; align-items:center; justify-content:space-between;}"
            ".progress-title{font-weight:600;}"
            ".progress-count{font-size:12px; color:#9ca3af;}"
            ".progress-body{padding:12px 14px;}"
            ".progress-bar{height:8px; background:var(--track); border-radius:9999px; overflow:hidden;}"
            ".progress-bar-fill{height:100%; background:linear-gradient(90deg,#60a5fa,#22d3ee);}"
            ".progress-remaining{margin-top:8px; font-size:12px; color:#9ca3af;}"
            "</style>"
            "</head><body>"
            f"<h2>PDF Index Manager</h2>"
            f"<div class=\"muted root-note\">Root: {resolve_path(docdir)}</div>"
            "<div class=\"toolbar\">"
            "<button class=\"btn btn-secondary\" hx-post=\"/index-files\" hx-include=\".pdf-checkbox:checked\" hx-target=\"#progress\" hx-swap=\"innerHTML\">Index selected</button>"
            "</div>"
            "<table>"
            "<thead><tr><th class=\"checkbox-col\"></th><th>File</th><th>Action</th></tr></thead>"
            f"<tbody id=\"table-body\">{rows_html}</tbody>"
            "</table>"
            "<div id=\"progress\" hx-get=\"/index-progress\" hx-trigger=\"load, every 700ms\" hx-swap=\"innerHTML\"></div>"
            "</body></html>"
        )

    @app.post("/index-file", response_class=HTMLResponse)
    async def index_file(request: Request):
        # Accept only form-encoded values from htmx
        form = await request.form()
        path = form.get("path")
        if not path:
            raise HTTPException(status_code=400, detail="Missing 'path'")
        file_path = Path(path)
        if not is_under_docdir(file_path):
            raise HTTPException(status_code=400, detail="Path is outside of document root")
        if not file_path.exists() or file_path.suffix.lower() != ".pdf":
            raise HTTPException(status_code=404, detail="PDF not found")
        if job_lock.locked():
            # Return current progress if busy
            return render_progress_card_html()
        # Start a single-file job
        asyncio.create_task(run_index_job([file_path]))
        return render_progress_card_html()

    @app.post("/index-files", response_class=HTMLResponse)
    async def index_files(request: Request):
        # Accept only form-encoded values from htmx with repeated 'path'
        form = await request.form()
        paths = form.getlist("path") if hasattr(form, "getlist") else []
        clean_paths: list[Path] = []
        for p in paths:
            fp = Path(p)
            if is_under_docdir(fp) and fp.exists() and fp.suffix.lower() == ".pdf" and not is_indexed(fp):
                clean_paths.append(fp)
        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for p in clean_paths:
            s = str(resolve_path(p))
            if s not in seen:
                seen.add(s)
                unique_paths.append(p)
        if not unique_paths:
            # Nothing to do
            return ""
        if job_lock.locked():
            return render_progress_card_html()
        asyncio.create_task(run_index_job(unique_paths))
        return render_progress_card_html()

    @app.get("/index-progress", response_class=HTMLResponse)
    def index_progress():
        if not job_lock.locked():
            # Clear card
            return ""
        return render_progress_with_oob_row_updates()

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
