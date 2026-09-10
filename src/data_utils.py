from pathlib import Path
from PIL import Image,ImageOps
import hashlib
IMAGE_EXTENSIONS={".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"}
def iter_images(folder):
    for p in sorted(Path(folder).rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS: yield p
def label_from_path(path):
    name=Path(path).parent.name.lower()
    if name=="real": return 0
    if name=="fake": return 1
    raise ValueError(f"Expected real/fake parent: {path}")
def validate_image(path):
    try:
        with Image.open(path) as im: im.verify()
        return True,""
    except Exception as e: return False,str(e)
def sha256_file(path,chunk_size=1024*1024):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(chunk_size),b""): h.update(chunk)
    return h.hexdigest()
