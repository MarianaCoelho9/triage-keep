# Frontend

Next.js UI for TriageKeep voice triage.

## Runtime Configuration

The frontend proxies backend routes through same-origin paths. Configure backend origin with:

```bash
BACKEND_ORIGIN=http://127.0.0.1:8000 npm run dev
```

Default backend origin (if unset): `http://127.0.0.1:8000`.

For websocket backend host override (when frontend/backend are on different origins), set:

```bash
NEXT_PUBLIC_BACKEND_ORIGIN=http://127.0.0.1:8000
```

## Proxied Routes

- `/report` -> `${BACKEND_ORIGIN}/report`
- `/report/fhir` -> `${BACKEND_ORIGIN}/report/fhir`
- `/ws/audio` -> same-origin websocket by default, or `${NEXT_PUBLIC_BACKEND_ORIGIN}/ws/audio` when override is set
- `/static/*` -> `${BACKEND_ORIGIN}/static/*`

## Production Note

In production, keep these same-origin routes available so the browser does not use hardcoded backend hosts/ports. This avoids mixed-content issues under HTTPS.

## Local Development

```bash
npm install
BACKEND_ORIGIN=http://127.0.0.1:8000 npm run dev
```
