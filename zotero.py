from pydantic import AliasPath, Field, TypeAdapter
from typing import Optional, List
from pydantic import BaseModel
import httpx
import urllib.parse
from pathlib import Path

class ItemData(BaseModel):
    title: str
    kind: str = Field(alias="itemType")
    abstract: str = Field(alias="abstractNote", default="")

class Meta(BaseModel):
    creatorSummary: str = "(unknown)"
    parsedDate: str = "(unknown)"

class Link(BaseModel):
    href: str
    def id(self) -> str:
        return self.href.split("/")[-1]

class Attachment(Link):
    kind: str = Field(alias="attachmentType")

class File(BaseModel):
    href: str
    def path(self)-> Path:
        addr = urllib.parse.urlparse(self.href)
        return Path(urllib.parse.unquote(addr.path))

class Links(BaseModel):
    attachment: Optional[Attachment] = None
    file: Optional[File] = Field(alias="enclosure", default=None)
    parent: Optional[Link] = Field(alias="up", default=None)

class Item(BaseModel):
    key: str
    version: int
    meta: Meta
    data: ItemData
    links: Links

class Client(httpx.Client):
    def __init__(self, base_url: str = "http://localhost:23119/api/users/0") -> None:
        super().__init__(base_url=base_url.rstrip("/"))

    def items(self) -> List[Item]:
        res = self.get("/items?itemType=-attachment")
        res.raise_for_status()
        return TypeAdapter(List[Item]).validate_python(res.json())

    def item(self, key: str) -> Item:
        res = self.get(f"/items/{key}")
        res.raise_for_status()
        return Item.model_validate(res.json())