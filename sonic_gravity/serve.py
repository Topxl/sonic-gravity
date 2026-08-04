"""Sert la surface et collecte les traces d'usage.

Volontairement minimal : la démo doit se lancer d'une commande, sans build, et
tourner hors ligne. Les modules ES natifs se passent d'empaqueteur, ce qui rend
aussi le code lisible tel quel par quelqu'un qui ouvre le dépôt.

    python -m sonic_gravity.serve
    python -m sonic_gravity.serve --port 8700 --host 0.0.0.0

Les traces (`--record`) servent à entraîner le monde appris : chaque ligne est
un instant (spectres par tranche, positions des faders, action en cours). Rien
n'est envoyé nulle part : le fichier reste dans `sonic_gravity/traces/`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

HERE = pathlib.Path(__file__).resolve().parent
WEB = HERE / "web"
TRACES = HERE / "traces"


def build_app(record: bool) -> FastAPI:
    app = FastAPI(title="Gravité sonore")

    @app.middleware("http")
    async def no_cache(request, call_next):
        """Rien n'est mis en cache : c'est un serveur de développement.

        Sans ça, `StaticFiles` laisse le navigateur garder les modules ES, et on
        vérifie une correction en regardant l'ancienne version — piège payé le
        2026-08-04 sur une refonte du rendu qui semblait sans effet. L'audio
        (2 Mo) est rechargé à chaque fois, ce qui est sans importance en local.
        """
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        return response

    @app.get("/spec.json")
    def spec() -> FileResponse:
        # Servi depuis le paquet, pas recopié dans web/ : Python et le runtime
        # doivent lire le MÊME fichier, sinon les deux implémentations du champ
        # divergent sans que rien ne le signale.
        return FileResponse(HERE / "spec.json", media_type="application/json")

    @app.websocket("/trace")
    async def trace(ws: WebSocket) -> None:
        await ws.accept()
        if not record:
            await ws.close(code=1008, reason="serveur lancé sans --record")
            return
        TRACES.mkdir(exist_ok=True)
        path = TRACES / f"session-{int(time.time())}.jsonl"
        n = 0
        try:
            with path.open("w") as fh:
                while True:
                    payload = await ws.receive_text()
                    fh.write(payload + "\n")
                    n += 1
                    if n % 200 == 0:
                        fh.flush()
        except WebSocketDisconnect:
            pass
        finally:
            print(f"trace : {n} instants → {path}")

    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--record", action="store_true", help="accepter les traces d'entraînement")
    args = ap.parse_args()

    if not (WEB / "audio" / "manifest.json").exists():
        raise SystemExit(
            "aucune scène sonore : lance d'abord\n"
            "  python -m sonic_gravity.prepare_demo --list\n"
            "  python -m sonic_gravity.prepare_demo <hash>"
        )

    import uvicorn

    print(f"→ http://{args.host}:{args.port}")
    uvicorn.run(build_app(args.record), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
