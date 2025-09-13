from pyrogram import Client
from pyrogram.errors import FloodWait, FileReferenceEmpty, FileReferenceExpired, RPCError
import os
import argparse
import asyncio
import time
import logging
from dotenv import load_dotenv

# Load .env configuration
load_dotenv()


class ZeroSizeError(Exception):
    """Raised when a download returns an empty buffer despite a known expected size."""
    pass

def _to_int(val, default=None):
    try:
        return int(val)
    except Exception:
        return default

API_ID = os.getenv("API_ID")
API_HASH = os.getenv("API_HASH")
CHAT_ID = os.getenv("CHAT_ID")
SESSION_NAME = os.getenv("SESSION_NAME", "murderculture")
DOWNLOADS_DIR = os.getenv("DOWNLOADS_DIR", "downloads")
DEFAULT_LIMIT = _to_int(os.getenv("LIMIT", "0"), 0)
STATE_FILE = os.getenv("STATE_FILE") or os.path.join(DOWNLOADS_DIR, ".downloaded_ids.txt")
MAX_RETRIES = _to_int(os.getenv("MAX_RETRIES", "5"), 5)
LOG_DIR = os.getenv("LOG_DIR") or os.path.join(DOWNLOADS_DIR, "logs")
LOG_FILE = os.getenv("LOG_FILE") or os.path.join(LOG_DIR, "download.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
ACCS_DIR = os.getenv("ACCS_DIR", "accs")
FLOODWAIT_SWITCH_SECS = _to_int(os.getenv("FLOODWAIT_SWITCH_SECS", "300"), 300)
QUEUE_FILE = os.getenv("QUEUE_FILE") or os.path.join(DOWNLOADS_DIR, ".queue_ids.txt")
DOWNLOAD_DELAY = float(os.getenv("DOWNLOAD_DELAY", "0.5"))
CONNECT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT", "30"))
CONNECT_RETRIES = _to_int(os.getenv("CONNECT_RETRIES", "3"), 3)
CONNECT_RETRY_DELAY = float(os.getenv("CONNECT_RETRY_DELAY", "10"))

if not API_ID or not API_HASH or not CHAT_ID:
    print("Missing required env vars. Please set API_ID, API_HASH, CHAT_ID in .env")
    raise SystemExit(1)
API_ID = _to_int(API_ID)

# Configure logging
def setup_logging():
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except Exception:
        pass
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logger = logging.getLogger("downloader")
    logger.setLevel(level)
    # Avoid duplicate handlers on reload
    if not logger.handlers:
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setLevel(level)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

LOGGER = setup_logging()

def console_error(msg: str):
    try:
        print("\n" + msg, flush=True)
    except Exception:
        pass

def progress(current, total):
    # Suppress noisy per-chunk prints to keep global progress readable
    return


def get_subfolder(msg):
    # Map media types to folders
    if getattr(msg, "audio", None) or getattr(msg, "voice", None):
        return "audio"
    if getattr(msg, "video", None) or getattr(msg, "video_note", None):
        return "videos"
    if getattr(msg, "animation", None):
        return "animations"
    if getattr(msg, "photo", None):
        return "photos"
    if getattr(msg, "sticker", None):
        return "stickers"
    if getattr(msg, "document", None):
        return "documents"
    return "other"


def has_media(msg):
    return any([
        getattr(msg, "audio", None),
        getattr(msg, "voice", None),
        getattr(msg, "video", None),
        getattr(msg, "video_note", None),
        getattr(msg, "animation", None),
        getattr(msg, "photo", None),
        getattr(msg, "sticker", None),
        getattr(msg, "document", None),
    ])


def unique_path(directory: str, filename: str) -> str:
    base, ext = os.path.splitext(filename)
    candidate = filename
    i = 1
    while os.path.exists(os.path.join(directory, candidate)):
        candidate = f"{base}_{i}{ext}"
        i += 1
    return os.path.join(directory, candidate)


def guess_filename(message, fallback_base: str = "file") -> str:
    try:
        if getattr(message, "document", None) and getattr(message.document, "file_name", None):
            return message.document.file_name
        if getattr(message, "audio", None):
            a = message.audio
            if getattr(a, "file_name", None):
                return a.file_name
            parts = [p for p in [getattr(a, "performer", None), getattr(a, "title", None)] if p]
            if parts:
                return " - ".join(parts) + ".mp3"
            return f"audio_{message.id}.mp3"
        if getattr(message, "voice", None):
            return f"voice_{message.id}.ogg"
        if getattr(message, "video", None):
            v = message.video
            if getattr(v, "file_name", None):
                return v.file_name
            return f"video_{message.id}.mp4"
        if getattr(message, "video_note", None):
            return f"video_note_{message.id}.mp4"
        if getattr(message, "animation", None):
            a = message.animation
            if getattr(a, "file_name", None):
                return a.file_name
            return f"animation_{message.id}.mp4"
        if getattr(message, "photo", None):
            return f"photo_{message.id}.jpg"
        if getattr(message, "sticker", None):
            st = message.sticker
            if getattr(st, "is_animated", False):
                return f"sticker_{message.id}.tgs"
            if getattr(st, "is_video", False):
                return f"sticker_{message.id}.webm"
            return f"sticker_{message.id}.webp"
    except Exception:
        pass
    return f"{fallback_base}_{getattr(message, 'id', 'unknown')}"


def get_expected_size(message) -> int | None:
    try:
        for attr in ("document", "audio", "voice", "video", "video_note", "animation", "photo", "sticker"):
            media = getattr(message, attr, None)
            if media and hasattr(media, "file_size"):
                return getattr(media, "file_size")
    except Exception:
        return None
    return None


async def process_message(app: Client, message, base_dir: str):
    if not has_media(message):
        return None
    try:
        file = await app.download_media(message, progress=progress, in_memory=True)
        if not file:
            return None
        file_name = os.path.basename(getattr(file, "name", "")) or guess_filename(message)

        sub = get_subfolder(message)
        target_dir = os.path.join(base_dir, sub)
        os.makedirs(target_dir, exist_ok=True)

        dest_path = unique_path(target_dir, file_name)
        tmp_path = dest_path + ".part"
        buf = file.getbuffer()
        expected = get_expected_size(message)
        # Basic integrity check by size if known
        if expected is not None and len(buf) == 0:
            # Empty buffer is often a symptom of cross-DC ExportAuthorization limits.
            try:
                LOGGER.warning(
                    f"Zero-size buffer for message %s (expected=%s) file=%s",
                    message.id,
                    expected,
                    file_name,
                )
            except Exception:
                pass
            raise ZeroSizeError()
        if expected is not None and len(buf) != expected:
            # Treat as failure so caller can retry
            try:
                LOGGER.warning(f"Size mismatch for message %s: expected=%s got=%s file=%s", message.id, expected, len(buf), file_name)
            except Exception:
                pass
            return None

        with open(tmp_path, "wb") as f:
            f.write(buf)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        # Atomic-ish finalize
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except Exception:
                pass
        os.replace(tmp_path, dest_path)
        return dest_path
    except (ZeroSizeError, FloodWait, RPCError) as e:
        # Let caller handle FloodWait and other RPC errors (for session rotation/backoff)
        raise e
    except Exception as e:
        try:
            LOGGER.exception(f"Unexpected error in process_message for message {getattr(message,'id',None)}: {e}")
        except Exception:
            pass
        return None


def main():
    parser = argparse.ArgumentParser(description="Download chat media into organized folders (single-thread reliable mode) with optional session rotation.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Limit messages to scan (only with --scan; 0 = all)")
    parser.add_argument("--downloads-dir", type=str, default=DOWNLOADS_DIR, help="Base downloads directory")
    parser.add_argument("--scan", action="store_true", help="Scan chat and enqueue media IDs to queue file; do not download")
    args = parser.parse_args()

    # Prepare session rotation list from accs/*.session; fallback to SESSION_NAME
    sessions = []
    workdir = None
    try:
        if os.path.isdir(ACCS_DIR):
            for fn in os.listdir(ACCS_DIR):
                if fn.endswith('.session'):
                    sessions.append(os.path.splitext(fn)[0])
    except Exception:
        pass
    sessions = sorted(set(sessions))
    if sessions:
        workdir = ACCS_DIR
    else:
        sessions = [SESSION_NAME]
        workdir = None

    if args.scan and not sessions:
        print("No sessions available for scanning.")
        return

    # Try sessions; if all are rate-limited, wait the minimal required time and retry
    # Per-session cooldown map {session_name: unix_ts_available}
    cooldowns: dict[str, float] = {}

    while True:
        last_error = None
        any_success = False
        min_wait = None
        now_ts = time.time()
        for idx, sess in enumerate(sessions):
            # Skip sessions still in cooldown
            avail_ts = cooldowns.get(sess)
            if avail_ts and now_ts < avail_ts:
                remaining = int(avail_ts - now_ts)
                if min_wait is None or remaining < min_wait:
                    min_wait = remaining
                print(f"Skipping session {sess}: cooldown {remaining}s remaining")
                continue
            print(f"Using session: {sess} ({idx+1}/{len(sessions)})")
            try:
                result = asyncio.run(async_main(args, chosen_session=sess, workdir=workdir, alt_sessions=len(sessions) > 1))
                # If async_main requests a switch, honor it and set cooldown if provided
                if isinstance(result, tuple) and len(result) == 2 and result[0] == "SWITCH":
                    wait_val = result[1]
                    if isinstance(wait_val, (int, float)) and wait_val > 0:
                        cooldowns[sess] = time.time() + wait_val
                        min_wait = wait_val if min_wait is None else min(min_wait, wait_val)
                    print(f"Session requested switch; moving to next.")
                    continue
                any_success = True
                last_error = None
                break
            except FloodWait as e:
                last_error = e
                wait_val = getattr(e, 'value', None)
                if isinstance(wait_val, int):
                    cooldowns[sess] = time.time() + wait_val
                    min_wait = wait_val if min_wait is None else min(min_wait, wait_val)
                print(f"Switching session due to FloodWait: {wait_val}s")
                continue
            except RPCError as e:
                last_error = e
                print(f"Switching session due to Pyrogram RPCError: {e}")
                continue
            except Exception as e:
                last_error = e
                print(f"Switching session due to error: {e}")
                continue

        if any_success:
            break
        if min_wait is not None:
            # Avoid extremely long single sleep; sleep up to 30 minutes at a time
            sleep_s = max(5, min(int(min_wait), 1800))
            print(f"All sessions are rate-limited. Waiting {sleep_s}s before retrying...")
            try:
                time.sleep(sleep_s)
            except Exception:
                pass
            # After waiting, retry the rotation loop
            continue
        # No FloodWait info or persistent errors: abort with last error
        if last_error is not None:
            print("All sessions failed. Last error:", last_error)
        break


async def async_main(args, chosen_session: str, workdir: str | None = None, alt_sessions: bool = False):
    base_dir = os.path.join(args.downloads_dir)
    os.makedirs(base_dir, exist_ok=True)
    # Ensure state file directory exists
    os.makedirs(os.path.dirname(STATE_FILE) if os.path.dirname(STATE_FILE) else '.', exist_ok=True)
    # Ensure queue file directory exists
    os.makedirs(os.path.dirname(QUEUE_FILE) if os.path.dirname(QUEUE_FILE) else '.', exist_ok=True)

    # Load downloaded IDs for resume
    downloaded_ids = set()
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as sf:
                for line in sf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        downloaded_ids.add(int(line))
                    except ValueError:
                        continue
    except Exception:
        pass

    state_lock = asyncio.Lock()

    # Establish connection with retries and timeout
    app = Client(chosen_session, api_id=API_ID, api_hash=API_HASH, workdir=workdir)
    started = False
    conn_attempts = CONNECT_RETRIES
    while True:
        try:
            await asyncio.wait_for(app.start(), timeout=CONNECT_TIMEOUT)
            started = True
            break
        except Exception as e:
            try:
                console_error(f"Unable to connect due to network issues: {getattr(e, 'args', [''])[0] or str(e)}")
                LOGGER.info(f"Connection attempt failed: {e}")
            except Exception:
                pass
            if conn_attempts > 0:
                conn_attempts -= 1
                console_error("Connection failed! Trying again...")
                try:
                    await asyncio.sleep(CONNECT_RETRY_DELAY)
                except Exception:
                    pass
                continue
            # Out of attempts: switch session if possible
            if alt_sessions:
                try:
                    await app.stop()
                except Exception:
                    pass
                return ("SWITCH", FLOODWAIT_SWITCH_SECS)
            # Single session: bubble up failure
            try:
                await app.stop()
            except Exception:
                pass
            raise

    try:
        limit = args.limit if args.limit and args.limit > 0 else None

        # 1) Either scan to queue or load queue
        if args.scan:
            print("Scanning chat history for media to enqueue...")
            existing_queue: set[int] = set()
            try:
                if os.path.exists(QUEUE_FILE):
                    with open(QUEUE_FILE, "r", encoding="utf-8") as qf:
                        for line in qf:
                            line = line.strip()
                            if line:
                                try:
                                    existing_queue.add(int(line))
                                except ValueError:
                                    pass
            except Exception:
                pass

            enqueued = 0
            already = 0
            async for message in app.get_chat_history(CHAT_ID, limit=limit):
                if has_media(message):
                    mid = message.id
                    if mid in downloaded_ids:
                        already += 1
                        continue
                    if mid in existing_queue:
                        continue
                    try:
                        with open(QUEUE_FILE, "a", encoding="utf-8") as qf:
                            qf.write(f"{mid}\n")
                        existing_queue.add(mid)
                        enqueued += 1
                    except Exception:
                        pass
            print(f"Scan complete. Enqueued: {enqueued}. Already downloaded: {already}. Queue: {QUEUE_FILE}")
            return None

        # No scan: load queue
        queue_ids: list[int] = []
        try:
            if os.path.exists(QUEUE_FILE):
                with open(QUEUE_FILE, "r", encoding="utf-8") as qf:
                    for line in qf:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            queue_ids.append(int(line))
                        except ValueError:
                            continue
        except Exception:
            pass

        # Remove already downloaded from queue and rewrite file
        queue_ids = [mid for mid in queue_ids if mid not in downloaded_ids]
        try:
            with open(QUEUE_FILE, "w", encoding="utf-8") as qf:
                for mid in queue_ids:
                    qf.write(f"{mid}\n")
        except Exception:
            pass

        total = len(queue_ids)
        if total == 0:
            print("Queue is empty. Use --scan to enqueue media IDs.")
            return None
        print(f"Found {total} queued media items to download.")

        # 2) Progress bar setup
        start_ts = time.time()
        completed = 0
        progress_lock = asyncio.Lock()
        errors_count = 0
        retries_count = 0
        floodwait_seconds = 0

        def fmt_time(sec: float) -> str:
            sec = int(max(0, sec))
            h, rem = divmod(sec, 3600)
            m, s = divmod(rem, 60)
            if h:
                return f"{h:d}:{m:02d}:{s:02d}"
            return f"{m:d}:{s:02d}"

        async def print_progress():
            now = time.time()
            elapsed = now - start_ts
            done = completed
            perc = (done / total) * 100 if total else 100.0
            eta = 0
            if done > 0:
                rate = done / max(1e-6, elapsed)
                eta = (total - done) / max(1e-6, rate)
            bar_len = 28
            filled = int(bar_len * perc / 100)
            bar = "#" * filled + "-" * (bar_len - filled)
            line = f"\r[{bar}] {done}/{total} ({perc:4.1f}%) | elapsed {fmt_time(elapsed)} | eta {fmt_time(eta)}"
            print(line, end="", flush=True)

        # 3) Sequential reliable download with retries from queue
        for mid in queue_ids:
            try:
                msg = await app.get_messages(CHAT_ID, mid)
            except FloodWait as e:
                # Rotate session on FloodWait if multiple sessions; otherwise wait and continue
                wait_val = getattr(e, 'value', FLOODWAIT_SWITCH_SECS)
                if alt_sessions:
                    return ("SWITCH", wait_val)
                else:
                    try:
                        LOGGER.info(f"FloodWait in get_messages id %s: %ss", mid, wait_val)
                    except Exception:
                        pass
                    await asyncio.sleep(max(5, min(int(wait_val), 1800)))
                    completed += 1
                    errors_count += 1
                    await print_progress()
                    continue
            except RPCError as e:
                # Rotate session on any Pyrogram RPC error if possible
                if alt_sessions:
                    return ("SWITCH", FLOODWAIT_SWITCH_SECS)
                else:
                    try:
                        LOGGER.exception(f"RPCError in get_messages for id {mid}: {e}")
                    except Exception:
                        pass
                    completed += 1
                    errors_count += 1
                    await print_progress()
                    continue
            except Exception as e:
                try:
                    LOGGER.exception(f"get_messages failed for id {mid}: {e}")
                except Exception:
                    pass
                completed += 1
                errors_count += 1
                await print_progress()
                continue
            # Skip if suddenly marked done in another run
            if msg and msg.id in downloaded_ids:
                completed += 1
                await print_progress()
                continue

            attempt = 0
            while True:
                try:
                    result = await process_message(app, msg, base_dir)
                    if result:
                        # mark as done
                        try:
                            with open(STATE_FILE, "a", encoding="utf-8") as sf:
                                sf.write(f"{msg.id}\n")
                                sf.flush()
                                try:
                                    os.fsync(sf.fileno())
                                except Exception:
                                    pass
                            downloaded_ids.add(msg.id)
                        except Exception:
                            pass
                        break
                    else:
                        # integrity failed or unknown error in process
                        attempt += 1
                        retries_count += 1
                        if attempt >= MAX_RETRIES:
                            errors_count += 1
                            break
                        await asyncio.sleep(min(30, 2 ** attempt))
                        continue
                except FloodWait as e:
                    # Respect Telegram flood wait
                    wait_s = getattr(e, "value", 30)
                    # Cap wait to avoid extremely long sleeps but still respectful
                    wait_s = max(5, min(wait_s, 1800))
                    floodwait_seconds += wait_s
                    msg_txt = f"FloodWait during message {msg.id}: sleep {wait_s}s"
                    console_error(msg_txt)
                    try:
                        LOGGER.info(msg_txt)
                    except Exception:
                        pass
                    # If ExportAuthorization is the cause or wait is long and we have alt sessions, rotate
                    err_text = str(e) if e else ""
                    if alt_sessions and ("auth.ExportAuthorization" in err_text or wait_s >= FLOODWAIT_SWITCH_SECS):
                        return ("SWITCH", wait_s)
                    await asyncio.sleep(wait_s)
                    continue
                except (FileReferenceEmpty, FileReferenceExpired):
                    # File reference expired: refetch message and retry
                    try:
                        msg = await app.get_messages(CHAT_ID, msg.id)
                    except Exception:
                        pass
                    attempt += 1
                    retries_count += 1
                    if attempt >= MAX_RETRIES:
                        errors_count += 1
                        break
                    await asyncio.sleep(min(30, 2 ** attempt))
                    continue
                except RPCError as e:
                    # On any other Pyrogram RPC error, rotate if possible; otherwise retry
                    if alt_sessions:
                        return ("SWITCH", FLOODWAIT_SWITCH_SECS)
                    attempt += 1
                    retries_count += 1
                    try:
                        LOGGER.exception(f"RPCError for message {getattr(msg,'id',None)}: {e}")
                        console_error(f"RPCError: message {getattr(msg,'id',None)}; attempt {attempt}/{MAX_RETRIES}")
                    except Exception:
                        pass
                    if attempt >= MAX_RETRIES:
                        errors_count += 1
                        break
                    await asyncio.sleep(min(30, 2 ** attempt))
                    continue
                except ZeroSizeError:
                    # Postpone this message: either switch session or move on to next item
                    if alt_sessions:
                        return ("SWITCH", FLOODWAIT_SWITCH_SECS)
                    errors_count += 1
                    console_error(f"Zero-size download for message {getattr(msg,'id',None)}; postponing")
                    # Don't burn retries here; let it be retried in a later run
                    break
                except Exception as e:
                    attempt += 1
                    retries_count += 1
                    try:
                        LOGGER.exception(f"Error downloading message {getattr(msg,'id',None)}: {e}")
                        console_error(f"Error: message {getattr(msg,'id',None)}; attempt {attempt}/{MAX_RETRIES}")
                    except Exception:
                        pass
                    if attempt >= MAX_RETRIES:
                        errors_count += 1
                        break
                    await asyncio.sleep(min(30, 2 ** attempt))
                    continue
            completed += 1
            await print_progress()
            # Small throttle between items to reduce rate of ExportAuthorization calls
            if DOWNLOAD_DELAY > 0:
                try:
                    await asyncio.sleep(DOWNLOAD_DELAY)
                except Exception:
                    pass

        # finish line
        print()
        try:
            dur = time.time() - start_ts
            summary = f"Done. total={total}, errors={errors_count}, retries={retries_count}, flood_wait={floodwait_seconds}s, log={LOG_FILE}"
            print(summary)
            LOGGER.info(summary)
        except Exception:
            pass
        return None
    finally:
        if started:
            try:
                await app.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
