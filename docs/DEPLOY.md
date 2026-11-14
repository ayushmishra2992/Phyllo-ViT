# Deploy & GitHub guide

## 1. Push to GitHub (≈ 2 minutes of your time)

The two model files are ~330 MB each, above GitHub's 100 MB file limit, so they go through **Git LFS** (already configured in `.gitattributes`).

1. On github.com create a **new empty repository** (no README, no .gitignore, no licence).
2. Double-click **`PUSH_TO_GITHUB.bat`** and paste the repo URL when asked.

It runs `git init`, `git lfs install`, `git add -A`, `git commit`, `git remote add origin`, `git push -u origin main`.
Manual equivalent:
```bash
git init -b main
git lfs install --local
git add -A
git commit -m "Phyllo-ViT v3"
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```
Notes
* First push uploads ~700 MB of LFS objects. Check your account's Git LFS storage/bandwidth quota on GitHub (Settings → Billing). If it is too small, delete `models/vit_phyllotaxy/` (the 3-class model used only for the withheld AQI value) and set `PHYLLO_LOAD_AQI_MODEL=0` — the 4-class app still works.
* Anyone cloning needs Git LFS installed, otherwise the model files arrive as small text pointers.
* Never commit `.venv/`, `frontend/node_modules/` or `frontend/.env.local` (already in `.gitignore`).

## 2. Run locally (no deployment)
`RUN_ME.bat` → http://127.0.0.1:8000 (built-in UI). Optional Next.js UI: `START_FRONTEND.bat` → http://localhost:3000/lab.

## 3. Deploy online

The backend runs PyTorch + two ViT-B/16 models (~2 GB RAM; ~1.2 GB with `PHYLLO_LOAD_AQI_MODEL=0`). It does **not** fit Vercel/Netlify serverless functions or 512 MB free tiers. Recommended free combination:

### Backend → Hugging Face Spaces (Docker, free CPU, 16 GB RAM)
1. huggingface.co → New Space → SDK **Docker** → Blank. Note its name `<user>/<space-name>`.
2. Build the backend-only upload folder and upload it (from this project folder, using the project's Python):
   ```bash
   python scripts/make_hf_space.py
   hf auth login
   hf upload <user>/<space-name> build_hf_space . --repo-type space
   ```
   (`hf auth login` asks for a **write** token from huggingface.co/settings/tokens. The folder already contains the Space's `README.md` header with `sdk: docker` and `app_port: 7860`.)
   Do not `git push` the whole repo to the Space: Hugging Face rejects the dataset images unless they are in LFS.
3. In the Space's Settings → Variables add `PHYLLO_CORS_ORIGINS=https://<your-frontend>.vercel.app`.
4. When the build finishes the API is at `https://<user>-<space-name>.hf.space` (built-in UI at `/`, docs at `/docs`).

The same `Dockerfile` works on Railway, Fly.io, Google Cloud Run, Azure Container Apps or any VM (`docker build -t phyllo . && docker run -p 7860:7860 phyllo`). The port comes from `$PORT` (default 7860).

### Frontend → Vercel (free)
1. vercel.com → Add New Project → import the GitHub repo.
2. **Root Directory: `frontend`** (framework auto-detected as Next.js).
3. Environment variable: `NEXT_PUBLIC_API_URL=https://<you>-<space-name>.hf.space`
4. Deploy. Then add the Vercel URL to `PHYLLO_CORS_ORIGINS` on the backend (step 3 above) and restart the Space.

### Environment variables
| Variable | Where | Default | Meaning |
|---|---|---|---|
| `NEXT_PUBLIC_API_URL` | frontend | `http://127.0.0.1:8000` | backend base URL |
| `PHYLLO_CORS_ORIGINS` | backend | (none) | extra allowed frontend origins, comma separated; localhost:3000 is always allowed |
| `PHYLLO_LOAD_AQI_MODEL` | backend | `1` | `0` skips loading the 3-class AQI model (saves ~350 MB RAM; AQI is withheld by default anyway) |
| `PORT` | backend (Docker) | `7860` | listen port |

## What was verified (12 Nov 2026)
* `npm run build` of the frontend (Next.js 16.4.6, `npm audit`: 0 vulnerabilities): compiles and type-checks.
* Frontend ↔ backend in a browser: Alternate, Opposite, Whorled images all accepted with the correct class; CORS via `PHYLLO_CORS_ORIGINS` confirmed.
* The exact file set the `Dockerfile` copies starts the API and serves `/predict`, `/evidence`, `/`.
* `scripts/make_hf_space.py` builds the 720 MB Space folder.
* **Not verified:** an actual `docker build` (Docker was not installed on the build machine), the upload itself, and the live Hugging Face / Vercel deployments.
