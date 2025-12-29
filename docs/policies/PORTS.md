# Ports

This project runs multiple local services and must avoid conflicts with the server-wide port inventory.

## Current ports

- Frontend (Vite dev server): `5174`
- Backend (FastAPI/Uvicorn): `7998`

## Notes

- Port `5176` is reserved for the `cryptotrader` frontend dev server (see `/home/flip/caramba/docs/PORTS.md`).
- Port `7999` is reserved for the Caramba FastAPI backend.
- External tools keep their defaults (e.g. ComfyUI `8188`).
