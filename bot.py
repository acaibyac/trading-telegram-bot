import os, requests
from fastapi import FastAPI, Request
from trading import (
    guess_ticker, load_prices_safe, sma_crossover,
    backtest_long_only, format_metrics,
    DEFAULT_PERIOD, DEFAULT_INTERVAL
)

# Setează aceste variabile în Render/Railway (Environment)
TOKEN = os.environ.get("TELEGRAM_TOKEN")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "mysecret123")

if not TOKEN:
    raise RuntimeError("Lipsește TELEGRAM_TOKEN în Environment Variables!")

BOT_API = f"https://api.telegram.org/bot{TOKEN}"
app = FastAPI()

def tg_send(chat_id: int, text: str):
    """Trimite mesaj în Telegram."""
    try:
        requests.post(f"{BOT_API}/sendMessage", json={"chat_id": chat_id, "text": text})
    except Exception as e:
        print("Eroare trimitere Telegram:", e)

@app.get("/")
def root():
    return {"ok": True, "service": "trading-telegram-bot"}

@app.post(f"/webhook/{{secret}}")
async def webhook(secret: str, request: Request):
    if secret != WEBHOOK_SECRET:
        return {"ok": False}

    update = await request.json()
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return {"ok": True}

    chat_id = msg["chat"]["id"]
    text = (msg.get("text") or "").strip()

    if not text or text.startswith("/start"):
        tg_send(chat_id, "Trimite un simbol (ex: AAPL, TSLA, BTC-USD).")
        return {"ok": True}

    ticker = guess_ticker(text, default="AAPL")
    try:
        prices, source = load_prices_safe(ticker, DEFAULT_PERIOD, DEFAULT_INTERVAL)
        df_sig = sma_crossover(prices)
        metrics = backtest_long_only(df_sig)
        reply = format_metrics(ticker, source, metrics)
    except Exception as e:
        reply = f"❌ Nu pot încărca date pentru {ticker}: {e}"

    tg_send(chat_id, reply)
    return {"ok": True}