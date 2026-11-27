"""Start ScamShield: predictor, EDA dashboard, and a small JSON API."""
import json
import webbrowser
import html
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import analysis
import program

HERE, PORT = Path(__file__).resolve().parent, 8000
print("Running preprocessing, EDA, and model comparison...")
RESULTS, BEST_NAME, MODEL, DASHBOARD, SIGNAL_PROFILES = analysis.build_artifacts()
print("\nTest-set comparison:\n" + RESULTS.to_string())
print(f"\nUsing {BEST_NAME} in the web app (highest F1).")


def dashboard_page(message: str = "") -> str:
    """Create the dashboard from fresh test metrics and saved plot files."""
    headers = "".join(f"<th>{name}</th>" for name in DASHBOARD["metrics"][0])
    rows = "".join("<tr>" + "".join(f"<td>{value}</td>" for value in row.values()) + "</tr>" for row in DASHBOARD["metrics"])
    plots = [("Class distribution", "class_distribution.png"), ("Message lengths", "message_length.png"), ("Scam-associated terms", "scam_terms.png"), ("Best-model errors", "confusion_matrix.png")]
    cards = "".join(f'<figure><img src="/artifacts/{file}" alt="{title}"><figcaption>{title}</figcaption></figure>' for title, file in plots)
    explanation = ""
    if message:
        detail = program.explain_prediction(MODEL, message[:1000], SIGNAL_PROFILES)
        def items(values, empty): return "<li>" + "</li><li>".join(html.escape(item) for item in values) + "</li>" if values else f"<li>{empty}</li>"
        colour = "#fee2e2" if detail["label"] == "Scam" else "#dcfce7"
        explanation = f'''<section class="decision" style="background:{colour}"><h2>How this input was assessed</h2><p><b>Message:</b> {html.escape(message[:1000])}</p><p class="result"><b>{detail["label"]}</b> — scam probability <b>{detail["probability"]}%</b>. The decision threshold is {detail["threshold"]}%.</p><div class="evidence"><div><h3>Signals associated with scam in training data</h3><ul>{items(detail["scam_signals"], "No strong known scam terms matched.")}</ul></div><div><h3>Signals associated with legitimate messages</h3><ul>{items(detail["legit_signals"], "No strong known legitimate terms matched.")}</ul></div></div><p class="note">These are the matching TF-IDF word/phrase signals learned from the training data. They help explain the prediction but do not prove a message is safe or fraudulent.</p></section>'''
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>ScamShield dashboard</title><style>body{{font-family:system-ui,sans-serif;background:#f1f5f9;margin:0;color:#172033}}main{{max-width:1100px;margin:auto;padding:28px}}a{{color:#173d68}}.hero{{background:#173d68;color:white;padding:24px;border-radius:14px}}table{{width:100%;border-collapse:collapse;background:white;margin:22px 0}}th,td{{padding:12px;border-bottom:1px solid #dbe3ee;text-align:left}}th{{background:#e8f0fb}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}}figure,.decision{{margin:22px 0;background:white;padding:18px;border-radius:12px;box-shadow:0 2px 9px #33415518}}img{{width:100%;height:auto}}figcaption{{font-weight:700;padding:8px 0 0}}.evidence{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}.note{{color:#526074;font-size:.9rem}}@media(max-width:650px){{.evidence{{grid-template-columns:1fr}}}}</style></head><body><main><p><a href="/">← Check a message</a></p><section class="hero"><h1>Model analysis dashboard</h1><p>Deployed model: <b>{BEST_NAME}</b> — selected by highest scam-class F1 score.</p></section>{explanation}<h2>Held-out test-set comparison</h2><table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table><h2>Exploratory data analysis &amp; evaluation</h2><section class="grid">{cards}</section></main></body></html>'''


class Handler(BaseHTTPRequestHandler):
    """Serve the predictor, dashboard, generated images, and prediction API."""
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/": content, kind = (HERE / "interface.html").read_bytes(), "text/html; charset=utf-8"
        elif parsed.path == "/dashboard": content, kind = dashboard_page(parse_qs(parsed.query).get("message", [""])[0]).encode(), "text/html; charset=utf-8"
        elif self.path == "/api/summary": content, kind = json.dumps(DASHBOARD).encode(), "application/json"
        elif self.path.startswith("/artifacts/"):
            file = HERE / "artifacts" / Path(self.path.removeprefix("/artifacts/")).name
            if not file.is_file(): self.send_error(404); return
            content, kind = file.read_bytes(), "image/png" if file.suffix == ".png" else "application/json"
        else: self.send_error(404); return
        self.send_response(200); self.send_header("Content-Type", kind); self.end_headers(); self.wfile.write(content)

    def do_POST(self):
        if self.path != "/predict": self.send_error(404); return
        size = min(int(self.headers.get("Content-Length", 0)), 5000)
        try: message = json.loads(self.rfile.read(size) or b"{}").get("message", "").strip()
        except (json.JSONDecodeError, UnicodeDecodeError): message = ""
        reply = {"error": "Please enter a message."} if not message else program.explain_prediction(MODEL, message[:1000], SIGNAL_PROFILES)
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(json.dumps(reply).encode())

    def log_message(self, *args): pass


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"\nOpen http://127.0.0.1:{PORT} — dashboard: /dashboard — Ctrl+C stops it.")
    webbrowser.open(f"http://127.0.0.1:{PORT}")
    try: server.serve_forever()
    except KeyboardInterrupt: print("\nStopped.")
