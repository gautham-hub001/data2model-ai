# Data2Model AI

Upload a CSV dataset and a multi-agent AI pipeline analyzes it, recommends the best ML model, and streams production-ready scikit-learn code back in real time.

## Architecture

```
User
 │
 ▼
Next.js  (Vercel)              ← TypeScript, Tailwind, Clerk auth, STOMP WebSocket
 │
 ▼
Spring Boot  (Render)          ← Spring AI multi-agent orchestration, WebSocket streaming
 │
 ├──► Python Flask  (Render)   ← scikit-learn, pandas, SMOTE (ML operations)
 └──► OpenRouter API           ← LLM (Llama 3, free tier)
```

## Services

| Directory | Tech | Hosts on |
|---|---|---|
| `frontend/` | Next.js 15 + TypeScript + Tailwind | Vercel |
| `backend-java/` | Spring Boot 3.4 + Spring AI 1.0 | Render |
| `backend-python/` | Python Flask + scikit-learn | Render |

## Agent Workflow

1. **DataAnalysisAgent** — calls Python Flask tool to analyze the uploaded CSV (statistics, correlations, class imbalance detection)
2. **ModelRecommendationAgent** — LLM explains why a specific model fits the data (streamed)
3. **ClarificationAgent** — if class imbalance detected, asks user whether to apply SMOTE; re-runs analysis if yes
4. **CodeGenerationAgent** — streams complete scikit-learn training pipeline code

## Local Development

### Python Backend
```bash
cd backend-python
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add OPENROUTER_API_KEY and INTERNAL_TOKEN
python app.py           # runs on :5001
```

### Java Backend
```bash
cd backend-java
# Requires Java 21 and Maven
export OPENROUTER_API_KEY=...
export INTERNAL_TOKEN=...        # must match Python backend
export CLERK_JWKS_URI=...        # from your Clerk dashboard
export PYTHON_API_URL=http://localhost:5001
mvn spring-boot:run              # runs on :8080
```

### Frontend
```bash
cd frontend
cp .env.local.example .env.local  # fill in Clerk + backend URLs
npm install
npm run dev                        # runs on :3000
```

## Deployment

All services deploy automatically via GitHub Actions on push to `main`.

**Required GitHub Secrets:**

| Secret | Description |
|---|---|
| `RENDER_PYTHON_DEPLOY_HOOK` | Render deploy webhook URL for Python service |
| `RENDER_JAVA_DEPLOY_HOOK` | Render deploy webhook URL for Java service |
| `VERCEL_TOKEN` | Vercel personal access token |
| `VERCEL_ORG_ID` | Vercel org ID |
| `VERCEL_PROJECT_ID` | Vercel project ID |
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | Clerk publishable key |
| `CLERK_SECRET_KEY` | Clerk secret key |
| `NEXT_PUBLIC_JAVA_API_URL` | Production Java API URL |
| `NEXT_PUBLIC_JAVA_WS_URL` | Production Java WebSocket URL |
| `NEXT_PUBLIC_POSTHOG_KEY` | PostHog project API key |

## Free Hosting Stack

- **Vercel** — Next.js frontend (unlimited hobby)
- **Render** — Both backend services (512MB RAM each, spins down at idle)
- **Supabase** — PostgreSQL + file storage (see `supabase/schema.sql`)
- **Clerk** — Auth: Google/GitHub OAuth + email (10k MAU free)
- **OpenRouter** — LLM API (free tier models)
- **PostHog** — Analytics (1M events/month free)
- **Cloudflare** — DNS + CDN (free)

> **Keep-warm tip:** Add a free [UptimeRobot](https://uptimerobot.com) monitor to ping `/health` on both Render services every 5 minutes to prevent cold starts.
