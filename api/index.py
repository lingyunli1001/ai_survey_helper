# Vercel serverless entrypoint.
# Vercel's @vercel/python runtime detects the `app` ASGI callable and serves it.
# All routes are rewritten here by vercel.json.
from app import app  # noqa: F401
