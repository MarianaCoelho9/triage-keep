# Frontend

Next.js 16 UI for TriageKeep voice triage.

## Scripts

- `npm run dev`: start local dev server.
- `npm run build`: production build.
- `npm run start`: run the production build.
- `npm run lint`: run ESLint.

## Runtime Configuration

### Environment Variables

- `BACKEND_ORIGIN` (server-side, optional)
  - Used by Next.js rewrites for HTTP routes.
  - Default: `http://127.0.0.1:8000`.
- `NEXT_PUBLIC_BACKEND_ORIGIN` (client-side, optional)
  - If set, websocket connects to `${NEXT_PUBLIC_BACKEND_ORIGIN}/ws/audio`.
  - Useful when frontend and backend are on different public origins.
- `NEXT_PUBLIC_VOICE_INTERACTION_MODE` (client-side, optional)
  - `ptt` (default) or `auto_turn`.

### Proxied HTTP Routes

- `/report` -> `${BACKEND_ORIGIN}/report`
- `/report/fhir` -> `${BACKEND_ORIGIN}/report/fhir`
- `/static/:path*` -> `${BACKEND_ORIGIN}/static/:path*`

### WebSocket URL Resolution

The app resolves `/ws/audio` in this order:

1. `NEXT_PUBLIC_BACKEND_ORIGIN` override (converted to `ws://` or `wss://`).
2. Localhost fallback: `ws://127.0.0.1:8000/ws/audio` (or `wss://` under HTTPS).
3. Same-origin host fallback: `{window.location.host}/ws/audio`.

## Local Development

```bash
npm install
BACKEND_ORIGIN=http://127.0.0.1:8000 npm run dev
```

Open `http://localhost:3000`.

## Production Note

Keep rewritten HTTP routes (`/report`, `/report/fhir`, `/static/*`) reachable from the deployed frontend origin to avoid mixed-content and CORS surprises.
