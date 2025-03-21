import functools
import os
import sys
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .status import get_status
from .guilds import get_guilds
from .channels import get_channels
from .roles import get_roles
from .search import get_searchcategories
from .search import search
from .search import get_autocomplete
from .messages import get_messages

# fix PIPE encoding error on Windows, auto flush print
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
print = functools.partial(print, flush=True)

# Get the absolute path to the frontend build directory
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../_temp/frontend"))
EXPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../release/exports"))

app = FastAPI(
	title="DCEF backend api",
	description="This is the backend api for the DCEF viewer.",
	version="0.2.0"
)

# Mount static files from SvelteKit build output
app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets")
app.mount("/fonts", StaticFiles(directory=os.path.join(FRONTEND_DIR, "fonts")), name="fonts")
app.mount("/twemoji-svg", StaticFiles(directory=os.path.join(FRONTEND_DIR, "twemoji-svg")), name="twemoji-svg")

# Serve individual files
@app.get("/favicon.png")
async def favicon():
	return FileResponse(os.path.join(FRONTEND_DIR, "favicon.png"))

@app.get("/favicon2.png")
async def favicon2():
	return FileResponse(os.path.join(FRONTEND_DIR, "favicon2.png"))

# Mount static files from exports directory
app.mount("/input", StaticFiles(directory=EXPORTS_DIR), name="exports")

# Serve index.html for root path
@app.get("/")
async def read_root():
	return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# Mount API routes under /api
app.include_router(get_status.router, prefix="/api")
app.include_router(get_guilds.router, prefix="/api")
app.include_router(get_channels.router, prefix="/api")
app.include_router(get_roles.router, prefix="/api")
app.include_router(get_searchcategories.router, prefix="/api")
app.include_router(search.router, prefix="/api")
app.include_router(get_autocomplete.router, prefix="/api")
app.include_router(get_messages.router, prefix="/api")
