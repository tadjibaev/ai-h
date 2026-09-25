#!/usr/bin/env python3
"""
Мок OpenAI-совместимого сервера — только стандартная библиотека Python.

Зачем: проверить bench_swarm.py и swarm_demo.py БЕЗ видеокарты (до того, как
платить за аренду GPU). Сервер не запускает нейросеть, а имитирует поведение
vLLM/SGLang:
  * префилл — время зависит только от НЕкэшированных токенов промпта;
  * кэш префиксов — блоками, как в vLLM (одинаковое начало промпта = попадание);
  * декод — N токенов/с на поток, общий потолок токенов/с на весь сервер;
  * лимит одновременных запросов (--max-num-seqs), остальные ждут в очереди;
  * /v1/models, /tokenize, /metrics (имена метрик как у vLLM).

Запуск:
    python3 scripts/mock_server.py --port 8000
"""
import argparse
import hashlib
import json
import random
import threading
import time
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WORDS = (
    "agent context token cache model swarm layer memory prefix analysis report "
    "result value system task plan check data file code test build review note"
).split()


def estimate_tokens(text: str) -> int:
    # Грубая оценка: ~1 токен на слово. Для мока точность не важна.
    return max(1, len(text.split()))


def render_prompt(messages) -> str:
    parts = []
    for m in messages:
        content = m.get("content") or ""
        if isinstance(content, list):  # формат [{"type":"text","text":...}]
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        parts.append(f"<|{m.get('role', 'user')}|> {content}")
    return "\n".join(parts)


class PrefixCache:
    """Кэш блоков префикса: ключ блока = хэш всего текста ДО конца блока."""

    def __init__(self, block_tokens: int, capacity_tokens: int):
        self.block_tokens = block_tokens
        self.capacity_blocks = max(1, capacity_tokens // block_tokens)
        self.blocks = OrderedDict()
        self.lock = threading.Lock()

    def _block_keys(self, words):
        keys, h = [], hashlib.sha256()
        full = len(words) // self.block_tokens
        for i in range(full):
            chunk = " ".join(words[i * self.block_tokens:(i + 1) * self.block_tokens])
            h.update(chunk.encode())
            keys.append(h.copy().hexdigest())
        return keys

    def lookup_and_insert(self, text: str) -> int:
        """Возвращает число токенов, найденных в кэше, и кладёт блоки в кэш."""
        keys = self._block_keys(text.split())
        with self.lock:
            hit = 0
            for k in keys:
                if k in self.blocks:
                    self.blocks.move_to_end(k)
                    hit += 1
                else:
                    break
            for k in keys[hit:]:
                self.blocks[k] = True
                if len(self.blocks) > self.capacity_blocks:
                    self.blocks.popitem(last=False)  # вытесняем самый старый блок
        return hit * self.block_tokens


class State:
    def __init__(self, args):
        self.args = args
        self.cache = PrefixCache(args.block_tokens, args.cache_tokens)
        self.slots = threading.BoundedSemaphore(args.max_num_seqs)
        self.lock = threading.Lock()
        self.running = 0
        self.waiting = 0
        self.counters = {
            "vllm:prefix_cache_queries_total": 0,
            "vllm:prefix_cache_hits_total": 0,
            "vllm:prompt_tokens_total": 0,
            "vllm:generation_tokens_total": 0,
            "vllm:request_success_total": 0,
        }

    def add(self, name, value):
        with self.lock:
            self.counters[name] += value

    def per_stream_delay(self) -> float:
        # Скорость одного потока падает, когда потоков много (общий потолок).
        with self.lock:
            active = max(1, self.running)
        tps = min(self.args.decode_tps, self.args.max_total_tps / active)
        return 1.0 / tps


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state: State = None  # задаётся в main()

    def log_message(self, fmt, *args):  # тише в консоли
        if self.state.args.verbose:
            super().log_message(fmt, *args)

    # ---------- helpers ----------
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    # ---------- routes ----------
    def do_GET(self):
        st = self.state
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [{
                "id": st.args.model, "object": "model", "owned_by": "mock",
                "max_model_len": st.args.max_model_len}]})
        elif self.path == "/health":
            self._json(200, {"status": "ok"})
        elif self.path == "/metrics":
            with st.lock:
                lines = [f'{k}{{model_name="{st.args.model}"}} {v}' for k, v in st.counters.items()]
                lines.append(f'vllm:num_requests_running{{model_name="{st.args.model}"}} {st.running}')
                lines.append(f'vllm:num_requests_waiting{{model_name="{st.args.model}"}} {st.waiting}')
            body = ("\n".join(lines) + "\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json(404, {"error": {"message": f"unknown path {self.path}"}})

    def do_POST(self):
        path = self.path.rstrip("/")
        if path == "/tokenize":
            req = self._read_json()
            text = req.get("prompt") or render_prompt(req.get("messages", []))
            n = estimate_tokens(text)
            self._json(200, {"count": n, "tokens": list(range(n)) if n < 50_000 else [],
                             "max_model_len": self.state.args.max_model_len})
        elif path in ("/v1/chat/completions", "/chat/completions"):
            self._chat(self._read_json())
        else:
            self._json(404, {"error": {"message": f"unknown path {self.path}"}})

    def _chat(self, req):
        st, a = self.state, self.state.args
        prompt = render_prompt(req.get("messages", []))
        prompt_tokens = estimate_tokens(prompt)
        max_tokens = int(req.get("max_completion_tokens") or req.get("max_tokens") or 256)
        if prompt_tokens + max_tokens > a.max_model_len:
            self._json(400, {"error": {"message": (
                f"This model's maximum context length is {a.max_model_len} tokens. "
                f"However, you requested {prompt_tokens + max_tokens} tokens."), "type": "BadRequestError"}})
            return
        kwargs = req.get("chat_template_kwargs") or {}
        thinking = kwargs.get("enable_thinking", True) and not a.no_thinking
        stream = bool(req.get("stream"))
        include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
        rng = random.Random(prompt_tokens)
        n_out = max_tokens if req.get("ignore_eos") else rng.randint(max(1, max_tokens // 2), max_tokens)
        n_think = n_out // 3 if thinking else 0

        with st.lock:
            st.waiting += 1
        st.slots.acquire()  # очередь, как у движка при исчерпании --max-num-seqs
        with st.lock:
            st.waiting -= 1
            st.running += 1
        try:
            cached = min(st.cache.lookup_and_insert(prompt), prompt_tokens)
            st.add("vllm:prefix_cache_queries_total", prompt_tokens)
            st.add("vllm:prefix_cache_hits_total", cached)
            st.add("vllm:prompt_tokens_total", prompt_tokens)
            time.sleep((prompt_tokens - cached) / a.prefill_tps)  # «префилл»

            rid = "chatcmpl-" + uuid.uuid4().hex[:24]
            usage = {"prompt_tokens": prompt_tokens, "completion_tokens": n_out,
                     "total_tokens": prompt_tokens + n_out,
                     "prompt_tokens_details": {"cached_tokens": cached}}
            # vLLM >= 0.2x отдаёт размышления в поле "reasoning" (раньше — "reasoning_content").
            pieces = [("reasoning" if i < n_think else "content", rng.choice(WORDS) + " ")
                      for i in range(n_out)]
            if not stream:
                for _ in pieces:
                    time.sleep(st.per_stream_delay())
                content = "".join(t for k, t in pieces if k == "content")
                reasoning = "".join(t for k, t in pieces if k == "reasoning") or None
                self._json(200, {"id": rid, "object": "chat.completion", "created": int(time.time()),
                                 "model": a.model, "usage": usage, "choices": [{
                                     "index": 0, "finish_reason": "length" if n_out == max_tokens else "stop",
                                     "message": {"role": "assistant", "content": content,
                                                 "reasoning": reasoning}}]})
            else:
                self._stream(rid, pieces, usage, include_usage, n_out == max_tokens)
            st.add("vllm:generation_tokens_total", n_out)
            st.add("vllm:request_success_total", 1)
        except (BrokenPipeError, ConnectionResetError):
            pass  # клиент отключился
        finally:
            with st.lock:
                st.running -= 1
            st.slots.release()

    def _stream(self, rid, pieces, usage, include_usage, hit_limit):
        a = self.state.args
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def send(obj):
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        base = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()), "model": a.model}
        send({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]})
        for kind, text in pieces:
            time.sleep(self.state.per_stream_delay())
            send({**base, "choices": [{"index": 0, "delta": {kind: text}}]})
        send({**base, "choices": [{"index": 0, "delta": {},
                                   "finish_reason": "length" if hit_limit else "stop"}]})
        if include_usage:
            send({**base, "choices": [], "usage": usage})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256  # по умолчанию 5: при 10+ агентах лишние подключения ждали бы ~1 с


def main():
    p = argparse.ArgumentParser(description="Мок OpenAI-совместимого LLM-сервера для тестов без GPU")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    p.add_argument("--max-model-len", type=int, default=262_144)
    p.add_argument("--max-num-seqs", type=int, default=16, help="одновременных запросов (остальные ждут)")
    p.add_argument("--prefill-tps", type=float, default=20_000, help="токенов/с префилла на запрос")
    p.add_argument("--decode-tps", type=float, default=60, help="токенов/с декода на один поток")
    p.add_argument("--max-total-tps", type=float, default=600, help="общий потолок токенов/с декода")
    p.add_argument("--block-tokens", type=int, default=256, help="размер блока кэша префиксов")
    p.add_argument("--cache-tokens", type=int, default=3_000_000, help="ёмкость кэша префиксов, токенов")
    p.add_argument("--no-thinking", action="store_true", help="никогда не выдавать reasoning_content")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    Handler.state = State(args)
    srv = Server((args.host, args.port), Handler)
    print(f"mock LLM server: http://{args.host}:{args.port}/v1  model={args.model}  "
          f"(prefill {args.prefill_tps:.0f} tok/s, decode {args.decode_tps:.0f} tok/s/поток)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
