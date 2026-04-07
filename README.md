# AR-Builder-Coach

## Project overview

This repository contains the implementation prototype for **AR Builder Coach**, a mobile-style interface that helps beginners assemble simple electronics projects by:

- scanning available materials,
- recommending feasible build plans based on detected inventory,
- guiding the user through step-by-step assembly with AR-style overlays and live feedback.

The current prototype is split into:

- a **FastAPI backend** in `app.py` that manages plans, sessions, step progression, live frame analysis, pause/zoom/repeat controls, and reference wiring generation
- a **single-page frontend** in `index.html` that presents the mobile UI and connects to the backend at `http://localhost:8000/api` by default 
- Python dependencies listed in `requirements.txt`

---

This project's code was developed with the help of ChatGPT and Claude.

---

## Repository files

- `app.py` — backend server and prototype computer vision / AR guidance logic
- `index.html` — frontend prototype UI
- `requirements.txt` — Python dependencies
- `env.example` — sample environment variables

---

## Important note about the current prototype

The current codebase uses **two different paths**:

### Local backend path
Screens 3 and 4 rely on the local FastAPI backend in `app.py`.

### Direct browser scanning path
The screen 2 scanning UI in `index.html` currently performs image analysis through a **direct browser-side external API call**, rather than sending scan frames to `app.py`.

Because of this, there are two ways staff may reproduce the prototype:

- **Full UI path**: run frontend + backend, allow camera access, and use the scan flow normally
- **Backend-supported path**: run frontend + backend and continue using the default inventory behavior even if screen 2 scanning is unavailable in the staff environment

This means that even if browser-side scanning is blocked, the staff can still reproduce the core implementation prototype for feasible plans, guided steps, overlays, session progression, and reference wiring.

---

## System requirements

Recommended:

- Python 3.10 or 3.11
- Chrome or another modern browser
- camera access enabled in the browser
- internet connection

Internet is needed because:

- the frontend loads React, ReactDOM, Babel, and Tailwind from CDNs
- the current screen 2 scanning implementation uses an external browser-side API

Optional:

- an OpenAI API key for backend-generated short instructions and reference images.Without an OpenAI key, the backend still runs. In that case it uses built-in step subtitles and falls back to a locally generated SVG reference image. 

---

## Setup instructions

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd AR-Builder-Coach
```

### 2. Create and activate a virtual environment

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

The required packages include FastAPI, Uvicorn, OpenCV, MediaPipe, NumPy, Pydantic, and OpenAI.

### 4. Create `.env`

Copy the example environment file:

```bash
cp env.example .env
```

Expected values:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_TEXT_MODEL=gpt-4.1-mini
OPENAI_IMAGE_MODEL=gpt-image-1
API_HOST=0.0.0.0
API_PORT=8000
```

Notes:

- `OPENAI_API_KEY` is optional
- if it is missing, the prototype still runs
- the backend health check will show whether OpenAI is configured

---

## Running the prototype

Use **two terminals**.

### Terminal 1 — run the FastAPI backend

```bash
source venv/bin/activate
python app.py
```

This starts the backend using the host and port defined in environment variables, with defaults `0.0.0.0` and `8000`.

To verify the backend is running, open:

```text
http://localhost:8000/api/health
```

Expected JSON shape:

```json
{
  "status": "ok",
  "openai_configured": true,
  "available_plans": ["plan_a", "plan_c"]
}
```

If no OpenAI key is set, `openai_configured` will be `false`.

### Terminal 2 — serve the frontend

Do not open `index.html` directly as a `file://` page. Use a local web server.

```bash
python3 -m http.server 5500
```

Then open:

```text
http://localhost:5500/index.html
```

The frontend is configured to use:

```text
http://localhost:8000/api
```

as its default backend base URL.

---

## Reproducing the implementation prototype

## 1. Home screen

1. Open `http://localhost:5500/index.html`
2. Confirm the AR Builder Coach home screen appears
3. Click **Scan Materials**

Expected result:
- the app transitions to the scanning screen

---

## 2. Scanning screen

1. Allow browser camera access
2. Point the camera at electronics components if available. 
3. Wait for labels to appear
4. Click **Confirm Parts**

Expected result:
- a live camera preview appears
- animated labels appear over detected objects
- a detected parts tray appears
- pressing **Confirm Parts** moves to the feasible plan screen

### If scanning does not work

That does **not** prevent the rest of the implementation prototype from being evaluated.

The frontend and backend both use a default electronics inventory when needed:

- breadboard
- arduino nano
- led
- 220Ω resistor
- jumper wire
- usb cable

So long as the user manually inputs or types a list of items separated by commas from this inventory, 
the remaining prototype flow can still be reproduced.

---
## 3. Feasible plans screen

---

## 4. Tracking / coaching screen
This is the main implementation prototype screen for guided assembly.
Expected visible elements:
- Click **Repeat** to replay message
- Click **Zoom** : frontend calls `/api/session/{session_id}/step/zoom` --> backend increases zoom state --> overlay rendering becomes larger
- Click **Ref Wiring** to play a reference image (either with a configured image model or a fallback SVG reference image)
- Click **Next** to proceed to the next step
- Click **Pause** to stop the instruction

