import arguably
from llama_index.core import StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from fastmcp import FastMCP
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import Response, HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field
from typing import Optional, Annotated
from pathlib import Path
import uvicorn
import sys
import threading
import webbrowser
import pymupdf
import io
import asyncio
import zotero


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
    title: Optional[str] = Field(default=None, description="Document title")
    zotero_item: Optional[str] = Field(default=None, description="Zotero item key")

    def from_chunk(chunk: dict):
        meta = chunk["doc_items"][0]
        zot = chunk.get("zotero") or {}
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
            title=chunk.get("title"),
            zotero_item=(zot or {}).get("item"),
        )

class File(BaseModel):
    key: str
    name: str
    abstract: str

@arguably.command
def main(
    state: Path = Path.home() / ".local" / "share" / "pdf-mcp",
    model: str = "nomic-embed-text",
    zotero_api: str = "http://localhost:23119/api/users/0",
):
    state = state.absolute()

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
                "title",
                "zotero",
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
    def read_file(item_key: str) -> list[Passage]:
        """Read an entire document by its Zotero item key, as returned by list_files"""
        res = vector_store.client.query(
            vector_store.collection_name, filter=f'zotero["item"] == "{item_key}"'
        )
        return sorted(list(map(Passage.from_chunk, res)), key=lambda x: x.ref)

    @mcp.tool
    def list_files() -> list[File]:
        """List all indexed files with metadata"""
        coll = vector_store.client.query(
            vector_store.collection_name, limit=16384, output_fields=["file_name", "zotero", "abstract"]
        )
        files = {}
        for x in coll:
            zot = x.get("zotero")
            if not zot: raise RuntimeError
            key = zot.get("item")
            if not key: raise RuntimeError
            if key and key not in files:
                files[key] = File(
                    key=key,
                    name=x["file_name"],
                    abstract=x.get("abstract", "")
                )
        return list(files.values())

    mcp_app = mcp.http_app(path="/")
    app = FastAPI(title="PDF MCP", lifespan=mcp_app.lifespan)
    app.mount("/mcp", mcp_app)

    zot = zotero.Client(zotero_api)
    def render_zotero_row_html(item: zotero.Item) -> str:
        if not item.links.attachment: return ""
        if item.links.attachment.kind != "application/pdf": return ""
        checkbox_html = f"<input type=\"checkbox\" class=\"key-checkbox\" name=\"key\" value=\"{item.key}\" />"
        if is_indexed(item.key):
            action_html = (
                f"<button class=\"btn btn-secondary\" disabled>Indexed</button>"
            )
            title_html = (
                f"<a class=\"doc-link\" href=\"/{item.key}\">{item.data.title}</a>"
            )
        else:
            action_html = (
                f"<button class=\"btn\" hx-post=\"/index-zotero-items\" hx-vals='{{\"key\": \"{item.key}\"}}' "
                f"hx-target=\"#progress\" hx-swap=\"innerHTML\">Index</button>"
            )
            title_html = item.data.title
        return (
            f"<tr id=\"row-{item.key}\">"
            f"<td class=\"checkbox-col\">{checkbox_html}</td>"
            f"<td class=\"authors\">{item.meta.creatorSummary}</td>"
            f"<td class=\"title\">{title_html}</td>"
            f"<td>{action_html}</td>"
            f"</tr>"
        )
    def render_header_html(active: str = "") -> str:
        active_home = " nav-link-active" if active == "home" else ""
        active_explore = " nav-link-active" if active == "explore" else ""
        return (
            "<div class=\"site-header\">"
            "<div class=\"brand\"><a class=\"brand-link\" href=\"/\">Zotero MCP</a></div>"
            "<nav class=\"nav-links\">"
            f"<a class=\"nav-link{active_home}\" href=\"/\">Home</a>"
            f"<a class=\"nav-link{active_explore}\" href=\"/explore\">Explore</a>"
            "</nav>"
            "</div>"
        )

    def styles() -> str:
        return (
            "<style>"
            ":root{--bg:#ffffff;--text:#111827;--muted:#6b7280;--border:#e5e7eb;--thead-bg:#f9fafb;--primary:#2563eb;--primary-hover:#1d4ed8;--secondary:#374151;--secondary-hover:#303846;--success:#16a34a;--card-bg:#111827;--card-text:#e5e7eb;--card-border:#374151;--track:#1f2937;}"
            "*{box-sizing:border-box;}"
            "body{font-family: ui-sans-serif, system-ui, -apple-system, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, \"Noto Sans\", \"Apple Color Emoji\", \"Segoe UI Emoji\"; padding:20px; color:var(--text); background:var(--bg);}"
            "h2{margin:0 0 10px;}"
            ".muted{color:var(--muted);}"
            ".toolbar{display:flex; gap:8px; align-items:center; margin-bottom:10px;}"
            ".input{border:1px solid var(--border); border-radius:6px; padding:8px 10px; width:100%; max-width:560px;}"
            ".site-header{display:flex; align-items:center; justify-content:space-between; margin-bottom:14px; padding-bottom:10px; border-bottom:1px solid var(--border);}"
            ".brand-link{font-weight:700; color:var(--text); text-decoration:none;}"
            ".nav-links{display:flex; gap:10px;}"
            ".nav-link{color:var(--muted); text-decoration:none; padding:6px 8px; border-radius:6px;}"
            ".nav-link:hover{background:var(--thead-bg); color:var(--text);}" 
            ".nav-link-active{background:var(--thead-bg); color:var(--text);}" 
            "table{border-collapse:collapse; width:100%;}"
            "th,td{border-bottom:1px solid var(--border); padding:8px 10px;}"
            "th{text-align:left; color:#374151; background:var(--thead-bg);}" 
            ".checkbox-col{width:28px;}"
            ".title{font-weight:500;}"
            ".authors{color:var(--muted);}"
            ".doc-link{color:var(--text); text-decoration:underline;}"
            ".doc-link:hover{text-decoration:underline;}"
            ".btn{background:var(--primary); color:#fff; border:none; border-radius:6px; padding:6px 10px; cursor:pointer; text-align:center; display:inline-block;}"
            ".btn:not([disabled]):hover{background:var(--primary-hover);}" 
            ".btn-secondary{background:var(--secondary);}" 
            ".btn-secondary:not([disabled]):hover{background:var(--secondary-hover);}" 
            ".btn-success{background:var(--success); cursor:default; opacity:.9;}"
            ".btn-success[disabled]{cursor:default;}"
            ".btn[disabled]{cursor:default; opacity:.6; pointer-events:none;}"
            ".progress-card{position:fixed; right:16px; bottom:16px; width:320px; background:var(--card-bg); color:var(--card-text); border:1px solid var(--card-border); border-radius:10px; box-shadow:0 10px 30px rgba(0,0,0,0.35);}" 
            ".progress-header{padding:12px 14px; border-bottom:1px solid var(--card-border); display:flex; align-items:center; justify-content:space-between;}"
            ".progress-title{font-weight:600;}"
            ".progress-count{font-size:12px; color:#9ca3af;}"
            ".progress-body{padding:12px 14px;}"
            ".progress-bar{height:8px; background:var(--track); border-radius:9999px; overflow:hidden;}"
            ".progress-bar-fill{height:100%; background:linear-gradient(90deg,#60a5fa,#22d3ee);}" 
            ".progress-remaining{margin-top:8px; font-size:12px; color:#9ca3af;}"
            ".results-columns{column-width:22rem; column-gap:12px; padding-top:12px;}"
            ".passage{border:1px solid var(--border); border-radius:10px; padding:12px; background:#fff; box-shadow:0 2px 10px rgba(0,0,0,0.04); break-inside:avoid; display:inline-block; width:100%; margin:12px 0;}"
            ".bottom-bar{margin-top:10px; display:flex; align-items:center; justify-content:space-between;}"
            ".link-small{font-size:12px; color:var(--muted); text-decoration:none; padding:2px 6px; border-radius:6px;}"
            ".link-small:hover{background:var(--thead-bg); color:var(--text);}" 
            "</style>"
        )



    def is_indexed(key: str) -> bool:
        res = vector_store.client.query(
            vector_store.collection_name,
            limit=1,
            output_fields=["file_name"],
            filter=f'zotero["item"] == "{key}"',
        )
        return bool(res)
    
    async def index_item(key: str):
        it = zot.item(key)
        if not it.links.attachment: raise RuntimeError(f"{key} has no attached pdf")

        att = zot.item(it.links.attachment.id())
        if not att.links.file: raise RuntimeError(f"{att.key} has no file")
        if not att.links.parent: raise RuntimeError(f"{att.key} has no parent")
        file = att.links.file.path()

        if is_indexed(it.key):
            return

        from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
        from llama_index.node_parser.docling import DoclingNodeParser
        from llama_index.readers.docling import DoclingReader

        pdf_reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        dir_reader = SimpleDirectoryReader(
            input_files=[file],
            file_extractor={".pdf": pdf_reader},
            required_exts=[".pdf"],
        )

        def _do_index():
            docs = dir_reader.load_data()
            for doc in docs:
                doc.metadata["zotero"] = {'item': it.key, 'pdf': att.key}
                doc.metadata["abstract"] = it.data.abstract
                doc.metadata["title"] = it.data.title
                doc.metadata["author"] = it.meta.creatorSummary
            VectorStoreIndex.from_documents(
                documents=docs,
                transformations=[DoclingNodeParser()],
                embed_model=embed_model,
                show_progress=False,
                storage_context=storage_context,
            )
        await asyncio.to_thread(_do_index)

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
        # Render progress card and out-of-band row updates for items that finished indexing
        updates: list[str] = []
        for key in current_job.get("done_paths", []):
            if key in current_job.get("emitted_done", set()):
                continue
            try:
                row_html = render_zotero_row_html(zot.item(key))
                # Tag for OOB swap to replace the row in place
                row_html = row_html.replace("<tr ", "<tr hx-swap-oob=\"outerHTML\" ", 1)
                updates.append(row_html)
                current_job["emitted_done"].add(key)
            except Exception:
                # If Zotero fetch fails, skip OOB update; card still renders
                pass
        return render_progress_card_html() + "".join(updates)

    async def run_index_job(keys: list[str]):
        nonlocal current_job
        async with job_lock:
            current_job = {
                "total": len(keys),
                "done": 0,
                "paths": keys,
                "done_paths": [],
                "emitted_done": set(),
                "status": "running",
            }
            try:
                for key in keys:
                    await index_item(key)
                    current_job["done"] += 1
                    current_job["done_paths"].append(key)
            finally:
                current_job["status"] = "done"

    @app.get("/", response_class=HTMLResponse)
    def home():
        try:
            items = zot.items()
        except Exception as e:
            return (
                "<!doctype html>"
                "<html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>PDF Index Manager</title></head>"
                f"<body><div style=\"padding:16px; color:#b91c1c;\">Failed to connect to Zotero API. Is Zotero running?</div><div><pre>{e}</pre></div></body></html>"
            )
        rows_html = "\n".join(render_zotero_row_html(z) for z in items if z.data.kind != "attachment")
        return (
            "<!doctype html>"
            "<html><head>"
            "<meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<title>Zotero MCP</title>"
            "<script src=\"https://unpkg.com/htmx.org@2.0.6\"></script>"
            f"{styles()}"
            "</head><body>"
            f"{render_header_html('home')}"
            "<div class=\"toolbar\">"
            "<button class=\"btn btn-secondary\" hx-post=\"/index-zotero-items\" hx-include=\".key-checkbox:checked\" hx-target=\"#progress\" hx-swap=\"innerHTML\">Index selected</button>"
            "</div>"
            "<table>"
            "<thead><tr><th class=\"checkbox-col\"></th><th>Authors</th><th>Title</th><th>Action</th></tr></thead>"
            f"<tbody id=\"table-body\">{rows_html}</tbody>"
            "</table>"
            "<div id=\"progress\" hx-get=\"/index-progress\" hx-trigger=\"load, every 700ms\" hx-swap=\"innerHTML\"></div>"
            "</body></html>"
        )

    @app.post("/index-zotero-items", response_class=HTMLResponse)
    async def index_zotero_items(key: Annotated[list[str] | str, Form()]):
        keys: list[str] = key if isinstance(key, list) else [key]
        # paths = {it.links.file.path() for it in map(zot.item, keys) if it.links.file}
        if job_lock.locked():
            return render_progress_card_html()
        asyncio.create_task(run_index_job(keys))
        return render_progress_card_html()

    @app.get("/index-progress", response_class=HTMLResponse)
    def index_progress():
        if not job_lock.locked():
            # Clear card
            return ""
        return render_progress_with_oob_row_updates()

    @app.get("/i/{id}.json")
    def passage_json(id: str):
        res = vector_store.client.get(vector_store.collection_name, ids=[id])
        if not len(res) == 1:
            raise HTTPException(status_code=404)
        return {"llm": Passage.from_chunk(res[0]), "doc": res[0]}

    # Backward compatibility for older JSON path
    @app.get("/{id}.json")
    def passage_json_compat(id: str):
        return RedirectResponse(url=f"/i/{id}.json", status_code=307)

    @app.get("/i/{id}")
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

    def render_passage_html(p: Passage, *, compact: bool, anchor_id: Optional[str] = None) -> str:
        distance_badge_html = ""
        if p.distance is not None:
            # Map distance to a hue from green (good/low) to red (high)
            hue = 120 - min(max(p.distance * 120, 0), 120)
            similarity = max(0.0, 1.0 - p.distance)
            distance_badge_html = (
                f"<span title=\"similarity\" style=\"display:inline-block; padding:2px 8px; border-radius:9999px; font-size:12px; font-weight:600; color:#fff; background:hsl({hue:.0f}, 85%, 45%);\">"
                f"{similarity*100:.1f}% similar"
                "</span>"
            )
        body_style = "margin-top:8px; white-space:pre-wrap;" + (" max-height: 12rem; overflow: hidden;" if compact else "")
        right_links = ""
        more_html = f"<a class=\"link-small\" href=\"/{p.zotero_item}#p-{p.id}\" title=\"Open full passage\">More</a>" if compact and p.zotero_item else ""
        inspect_html = f"<a class=\"link-small\" href=\"/i/{p.id}\" target=\"_blank\">Inspect</a>"
        if compact and more_html:
            right_links = more_html + " · " + inspect_html
        else:
            right_links = inspect_html
        id_attr = f" id=\"{anchor_id}\"" if anchor_id else ""
        return (
            f"<div class=\"passage\"{id_attr} style=\"border:1px solid var(--border); border-radius:10px; padding:12px; background:#fff; box-shadow:0 2px 10px rgba(0,0,0,0.04); break-inside: avoid; display:inline-block; width:100%; margin:0 0 12px;\">"
            f"<div style=\"display:flex; justify-content:space-between; align-items:center;\">"
            f"<div><strong>{p.title or p.file}</strong><span class=\"muted\"> · {p.section} · p.{p.page}</span></div>"
            f"<div></div>"
            "</div>"
            f"<div class=\"passage-body\" style=\"{body_style}\">{p.text}</div>"
            f"<div class=\"bottom-bar\">"
            f"{distance_badge_html}"
            f"<div>{right_links}</div>"
            "</div>"
            "</div>"
        )

    @app.get("/explore", response_class=HTMLResponse)
    def explore(q: Optional[str] = None, top_k: int = 10):
        results_html = ""
        if q:
            embeddings = embed_model.get_text_embedding_batch([q])
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
                    "title",
                    "zotero",
                ],
            )
            passages = list(map(Passage.from_chunk, res[0]))
            passages.sort(key=lambda p: (p.distance is None, p.distance))
            cards_html = "".join(render_passage_html(p, compact=True) for p in passages)
            results_html = (
                f"<div class=\"muted\" style=\"margin: 8px 0;\">Found {len(passages)} passages</div>"
                f"<div class=\"results-columns\">{cards_html}</div>"
            )
        return (
            "<!doctype html>"
            "<html><head>"
            "<meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<title>Explore</title>"
            f"{styles()}"
            "</head><body>"
            f"{render_header_html('explore')}"
            "<form method=\"get\" action=\"/explore\" class=\"toolbar\">"
            f"<input class=\"input\" type=\"text\" name=\"q\" value=\"{q or ''}\" placeholder=\"Search\" autofocus />"
            "<button class=\"btn\" type=\"submit\">Search</button>"
            "</form>"
            f"{results_html}"
            "</body></html>"
        )

    @app.get("/{item}", response_class=HTMLResponse)
    def view_item(item: str):
        # Avoid swallowing other static routes
        if item in {"mcp", "index-progress", "index-zotero-items", "explore", "i"}:
            raise HTTPException(status_code=404)
        if not is_indexed(item):
            raise HTTPException(status_code=404, detail="Item not indexed")
        try:
            it = zot.item(item)
        except Exception:
            it = None
        res = vector_store.client.query(
            vector_store.collection_name,
            filter=f'zotero["item"] == "{item}"',
            limit=16384,
            output_fields=[
                "text",
                "file_name",
                "file_path",
                "doc_items",
                "headings",
            ],
        )
        passages = sorted(list(map(Passage.from_chunk, res)), key=lambda x: x.ref)
        header_html = (
            f"<div class=\"muted\">{it.meta.creatorSummary}</div><div style=\"font-size:18px; font-weight:600;\">{it.data.title}</div>"
            if it is not None
            else f"<div class=\"muted\">Zotero item {item}</div>"
        )
        return (
            "<!doctype html>"
            "<html><head>"
            "<meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>Item {item}</title>"
            f"{styles()}"
            "</head><body>"
            f"{render_header_html('')}"
            f"{header_html}"
            f"<div class=\"muted\" style=\"margin-top:6px;\">{len(passages)} passages</div>"
            + "".join(render_passage_html(p, compact=False, anchor_id=f"p-{p.id}") for p in passages)
            + "</body></html>"
        )

    # On macOS, run rumps on the main thread and uvicorn in the background.
    # On other platforms, run uvicorn normally.
    if sys.platform == "darwin":
        def _start_server():
            uvicorn.run(app, host="0.0.0.0", port=7777)
        threading.Thread(target=_start_server, name="uvicorn-server", daemon=True).start()

        try:
            import rumps
        except Exception:
            # If rumps is unavailable, just keep the server running in background
            # This avoids breaking non-macOS environments inadvertently.
            threading.Event().wait()
            return

        def open_home(_=None):
            webbrowser.open("http://localhost:7777/")

        def quit_app(_=None):
            sys.exit(0)

        app_tray = rumps.App("Zotero MCP", callback=open_home)
        app_tray.menu = [
            rumps.MenuItem("Open Home", callback=open_home),
            None,
            rumps.MenuItem("Quit", callback=quit_app),
        ]
        app_tray.run()
    else:
        uvicorn.run(app, host="0.0.0.0", port=7777)


if __name__ == "__main__":
    arguably.run()
