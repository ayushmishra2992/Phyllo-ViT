# Phyllo-ViT — Frontend

Next.js (App Router, TypeScript, Tailwind) frontend for the Phyllo-ViT leaf
phyllotaxy classifier and preliminary AQI estimator.

## Pages

- `/` — landing page (project intro)
- `/lab` — the working tool: upload a leaf image, see the prediction

## Setup

```bash
npm install
cp .env.local.example .env.local   # adjust NEXT_PUBLIC_API_URL if needed
npm run dev
```

The app expects Ayush's FastAPI backend running at
`http://127.0.0.1:8000` (see the main repo's setup instructions). Start that
first — `uvicorn src.api:app --reload` — then run `npm run dev` here and
open `http://localhost:3000`.

## Notes

- AQI is only shown when the backend returns a non-null `aqi.value`
  (currently only for the Whorled class). The frontend never fabricates a
  value for the other classes — it shows the backend's explanatory message
  instead.
- `lib/api.ts` is the only file that talks to the backend. If the API
  response shape changes, update the `PredictResponse` type there.
