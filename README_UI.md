# Claude Researcher - Web UI

Modern web interface for the hierarchical deep research agent.

## 🚀 Quick Start

```bash
# Start both API and UI servers
researcher ui

# Or use the dev script
./scripts/dev.sh

# Or start manually
python -m api.server          # API on :8080
cd ui && npm run dev          # UI on :3000
```

**Access**: http://localhost:3000

## 📋 Features

### ✅ Available Now

- **Session Dashboard** - View all research sessions from database
- **Create Sessions** - Start new research with custom time limits
- **Real-time Status** - See which sessions are active/completed
- **Beautiful UI** - Dark theme, smooth animations, responsive design

### 🚧 Coming Soon

- **Live Activity Feed** - Watch agents work in real-time
- **Agent Thinking Panel** - See Director/Manager/Intern decision-making
- **Knowledge Graph** - Interactive D3.js visualization
- **Findings Browser** - Search and filter research findings
- **Report Preview** - View generated reports

## 🛠️ Tech Stack

- **Frontend**: Next.js 16 + React 19 + TypeScript + TailwindCSS
- **Backend**: FastAPI + uvicorn + aiosqlite
- **Database**: SQLite (shared with CLI)
- **Design**: Custom design system (#2b7cee primary, #101822 dark)

## 📁 Structure

```
api/                    # FastAPI backend
├── server.py          # Main app + WebSocket
├── routes/            # API endpoints
├── models.py          # Pydantic models
└── db.py              # Database service

ui/                     # Next.js frontend
├── app/               # Pages (App Router)
├── components/        # React components
└── lib/               # Utilities

scripts/                # Dev tools
├── dev.sh             # Start both servers
└── start_api.sh       # Start API only
```

## 🎯 API Endpoints

```
GET    /                      Health check
GET    /api/sessions/         List all sessions
POST   /api/sessions/         Create new session
GET    /api/sessions/{id}     Get specific session
DELETE /api/sessions/{id}     Delete session
WS     /ws/{session_id}       Real-time events
```

**Documentation**: http://localhost:8080/docs

## 🎨 Design System

**Colors:**
- Primary: `#2b7cee` (blue)
- Background: `#101822` (dark)
- Surface: `#1a2332` (lighter dark)
- Success: `#10b981` (green)
- Error: `#ef4444` (red)

**Typography:**
- UI: Inter (Google Fonts)
- Code: JetBrains Mono

## 🔧 Development

### First Time Setup

```bash
# Install Python dependencies
pip install -e .

# Install Node dependencies
cd ui && npm install && cd ..
```

### Running

```bash
# Full stack
./scripts/dev.sh

# API only
python -m api.server

# UI only
cd ui && npm run dev
```

### Logs

```bash
# API logs
tail -f /tmp/claude_api.log

# UI logs
tail -f /tmp/claude_ui.log
```

### Stop Servers

```bash
# If using dev.sh
Ctrl+C

# If started separately
kill $(cat /tmp/claude_api.pid)
kill $(cat /tmp/claude_ui.pid)

# Force kill
lsof -ti:8080,3000 | xargs kill -9
```

## 📚 Documentation

- **Implementation Summary**: `IMPLEMENTATION_SUMMARY.md`
- **Phase 1 Progress**: `PHASE_1_PROGRESS.md`
- **API Reference**: `API_IMPLEMENTATION.md`
- **Future Roadmap**: `future_scope/MASTER_ROADMAP.md`

## 🐛 Troubleshooting

**Port already in use:**
```bash
lsof -ti:8080 | xargs kill -9
lsof -ti:3000 | xargs kill -9
```

**Database locked:**
```bash
# Close all running researcher processes
pkill -f researcher
```

**UI not showing sessions:**
- Check API is running: `curl http://localhost:8080/`
- Check browser console for errors
- Verify CORS is configured (should be automatic)

**Can't create session:**
- Check API logs: `tail /tmp/claude_api.log`
- Verify database exists: `ls -la research.db`
- Test API directly: `curl -X POST http://localhost:8080/api/sessions/ -H "Content-Type: application/json" -d '{"goal": "test", "time_limit": 30}'`

## 🎯 What's Different from CLI?

| Feature | CLI | Web UI |
|---------|-----|--------|
| Start research | `researcher "topic"` | Click "New Research" |
| View sessions | `researcher logs` | Dashboard grid |
| See progress | Terminal spinner | Real-time feed (soon) |
| View report | Saved to `output/` | In-browser preview (soon) |
| Agent thinking | Hidden | Visible panel (soon) |
| Knowledge graph | HTML file | Interactive D3.js (soon) |

## 🔐 Security

⚠️ **For local use only** - No authentication implemented

Do NOT expose to the internet without:
- Authentication (JWT/OAuth)
- HTTPS
- Rate limiting
- Input validation
- CORS restrictions

## 🚢 Deployment (Future)

Not production-ready yet. When ready:

```bash
# Build UI
cd ui && npm run build

# Run production
python -m api.server  # Add --host 0.0.0.0 --port 80
cd ui && npm start    # Runs on :3000
```

## 📊 Performance

- API startup: ~2s
- UI startup: ~5s
- Session list: <100ms
- Session create: <50ms
- Page load: <1s

## 🤝 Contributing

1. Follow existing code style
2. Test both API and UI
3. Update documentation
4. Check browser console for errors
5. Verify mobile responsiveness

## 📄 License

Same as main project.

---

**Status**: ✅ Phase 1 Complete

**Next**: WebSocket events + Session detail page

**Questions?** Check `IMPLEMENTATION_SUMMARY.md`
