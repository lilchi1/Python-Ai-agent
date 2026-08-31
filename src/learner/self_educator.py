"""Модуль самообучения агента — рекурсивный, многопоточный, с прогресс-баром."""

import json
import os
import random
import re
import sys
import time
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from html.parser import HTMLParser
from pathlib import Path
from datetime import datetime


# ── Утилита для прогресс-бара ──────────────────────────────────────────
class ProgressBar:
    def __init__(self, total: int, prefix: str = "", width: int = 30):
        self.total = max(total, 1)
        self.prefix = prefix
        self.width = width
        self.current = 0
        self.start_time = time.time()
        self.lock = threading.Lock()

    def update(self, n: int = 1, extra: str = ""):
        with self.lock:
            self.current += n
            self._render(extra)

    def _render(self, extra: str = ""):
        pct = self.current / self.total
        filled = int(self.width * pct)
        bar = "=" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.start_time
        speed = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / speed if speed > 0 else 0
        sys.stdout.write(
            f"\r  [{bar}] {self.current}/{self.total} "
            f"({pct:.0%}) {speed:.1f}/s ETA:{eta:.0f}s {extra}"
        )
        sys.stdout.flush()

    def finish(self):
        self._render()
        sys.stdout.write("\n")


# ── Парсер HTML → текст ───────────────────────────────────────────────
class ContentExtractor(HTMLParser):
    """Извлекает текст из HTML, пропуская скрипты, стили, навигацию."""

    SKIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "form", "noscript"}

    def __init__(self):
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data):
        if self._skip_depth == 0:
            text = data.strip()
            if len(text) > 20:
                self._chunks.append(text)

    def get_text(self) -> str:
        return "\n".join(self._chunks)


# ── Динамический генератор тем ──────────────────────────────────────────
# Минимальный набор примитивов для комбинаторной генерации.
# Агент сам создаёт новые темы из этих кирпичиков + из уже изученного.

_VERBS = [
    "реализовать", "оптимизировать", "паттерн", "алгоритм", "структура",
    "модель", "архитектура", "подход", "механизм", "принцип",
    "концепция", "техника", "стратегия", "метод", "способ",
]

_NOUNS = [
    "данных", "памяти", "производительности", "безопасности", "масштабируемости",
    "параллелизма", "синхронизации", "вычислений", "ввода-вывода", "сети",
    "файлов", "процессов", "потоков", "событий", "сообщений",
    "зависимостей", "состояния", "конфигурации", "ошибок", "логирования",
    "тестирования", "деплоя", "мониторинга", "кэширования", "сериализации",
]

_ADJECTIVES = [
    "быстрый", "безопасный", "устойчивый", "масштабируемый", "эффективный",
    "параллельный", "асинхронный", "атомарный", "консистентный", "идемпотентный",
    "ленивый", "жадный", "оптимальный", "рекурсивный", "итеративный",
]

# Примитивы, из которых строятся темы ( substantially smaller than before )
_CONCEPT_PARTS = [
    "сортировка", "поиск", "граф", "дерево", "хэш",
    "стек", "очередь", "список", "массив", "кэш",
    "буфер", "очередь сообщений", "поток", "процесс", "контекст",
    "замыкание", "итератор", "генератор", "декоратор", "метaclass",
    "интерфейс", "абстракция", "наследование", "композиция", "полиморфизм",
    "рекурсия", "мемоизация", "ленивые вычисления", "尾овая рекурсия",
    "COROUTINE", "green thread", "callback", "promise", "signal",
    "лека", "сигнатура", "тип", "сущность", "контракт",
    "миграция", "транзакция", "индекс", "реплика", "шейarding",
    "рендеринг", "парсинг", "генерация", "трансформация", "валидация",
    "авторизация", "аутентификация", "шифрование", "хеширование", "подпись",
    "CLI", "TUI", "GUI", "API", "SDK",
    "CLI", "TUI", "GUI", "API", "SDK",
]


class TopicGenerator:
    """Генерирует случайные темы на основе примитивов + уже изученного контента."""

    def __init__(self):
        self._learned_keywords: list[str] = []
        self._used_hashes: set[str] = set()

    def feed_learned(self, keywords: list[str]):
        """Добавляет ключевые слова из изученного контента."""
        for kw in keywords:
            if kw not in self._learned_keywords and len(kw) > 3:
                self._learned_keywords.append(kw)

    def generate(self, lang: str = "") -> str:
        """Генерирует новую уникальную тему."""
        for _ in range(50):
            topic = self._compose(lang)
            h = hashlib.md5(topic.encode()).hexdigest()[:8]
            if h not in self._used_hashes:
                self._used_hashes.add(h)
                return topic
        # Фоллбэк — просто комбинация двух случайных слов
        return f"{random.choice(_CONCEPT_PARTS)} и {random.choice(_CONCEPT_PARTS)}"

    def _compose(self, lang: str) -> str:
        """Составляет тему из примитивов."""
        roll = random.random()

        if roll < 0.25 and self._learned_keywords:
            # Берём слово из уже изученного + примитив
            kw = random.choice(self._learned_keywords)
            noun = random.choice(_NOUNS)
            return f"{kw} и {noun}"

        if roll < 0.5:
            #Adj + Noun: "быстрый поиск данных"
            adj = random.choice(_ADJECTIVES)
            noun = random.choice(_NOUNS)
            return f"{adj} {noun}"

        if roll < 0.7:
            # Verb + Noun: "паттерн синхронизации"
            verb = random.choice(_VERBS)
            noun = random.choice(_NOUNS)
            return f"{verb} {noun}"

        if roll < 0.85:
            # Два концепта: "дерево и кэш"
            a, b = random.sample(_CONCEPT_PARTS, 2)
            return f"{a} и {b}"

        # Один конкретный концепт
        return random.choice(_CONCEPT_PARTS)

    def generate_search_url(self, topic: str, seed_lang: str = "") -> str:
        """Генерирует URL поискового запроса по теме."""
        q = topic.replace(" ", "+")
        if seed_lang:
            q += f"+{seed_lang}"
        return f"https://www.google.com/search?q={q}+programming+tutorial"


def random_topic(seed_lang: str = "", generator: TopicGenerator | None = None) -> str:
    """Генерирует случайную тему. Если есть generator — использует его."""
    if generator:
        return generator.generate(seed_lang)
    # Фоллбэк без генератора
    return f"{random.choice(_CONCEPT_PARTS)} и {random.choice(_NOUNS)}"


def random_search_queries(topic: str, count: int = 3) -> list[str]:
    """Генерирует случайные поисковые запросы по теме."""
    templates = [
        "{topic} tutorial",
        "{topic} examples",
        "{topic} best practices",
        "{topic} guide",
        "how to implement {topic}",
        "{topic} in programming",
        "{topic} explained",
        "learn {topic}",
        "{topic} for beginners",
        "{topic} advanced",
    ]
    selected = random.sample(templates, min(count, len(templates)))
    return [t.format(topic=topic) for t in selected]


# ── Сетевые утилиты ────────────────────────────────────────────────────
def fetch_page(url: str, user_agent: str = "", timeout: int = 15) -> str | None:
    """Загружает страницу, возвращает HTML или None."""
    headers = {
        "User-Agent": user_agent or "Mozilla/5.0 (compatible; Python-Ai-Agent/1.0)",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
    }
    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            for enc in ("utf-8", "cp1251", "latin-1"):
                try:
                    return raw.decode(enc)
                except UnicodeDecodeError:
                    continue
            return raw.decode("utf-8", errors="replace")
    except (URLError, HTTPError, OSError, TimeoutError) as e:
        print(f"  [ошибка] {url}: {e}")
        return None


def extract_content(html: str) -> str:
    """Извлекает чистый текст из HTML."""
    parser = ContentExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    return parser.get_text()


def extract_code_blocks(html: str) -> list[str]:
    """Извлекает блоки кода из HTML."""
    blocks = []
    print("working")

    # <code> и <pre> теги
    for match in re.finditer(r"<(?:code|pre)[^>]*>(.*?)</(?:code|pre)>", html, re.DOTALL | re.IGNORECASE):
        code = re.sub(r"<[^>]+>", "", match.group(1)).strip()
        if len(code) > 30:
            blocks.append(code)
    # Подсветка синтаксиса — <span class="...">
    for match in re.finditer(r'<div[^>]*class="[^"]*highlight[^"]*"[^>]*>(.*?)</div>', html, re.DOTALL):
        code = re.sub(r"<[^>]+>", "", match.group(1)).strip()
        if len(code) > 30:
            blocks.append(code)
    return blocks


def discover_links(base_url: str, html: str, max_links: int = 10) -> list[str]:
    """Находит внутренние ссылки для рекурсивного обхода."""
    links = []
    # Извлекаем домен из base_url
    proto_end = base_url.find("://")
    if proto_end < 0:
        return links
    base_domain = base_url[proto_end + 3:].split("/")[0]

    for match in re.finditer(r'href="([^"]+)"', html):
        href = match.group(1)
        full = href
        if href.startswith("/"):
            full = f"{base_url[:proto_end + 3]}{base_domain}{href}"
        elif not href.startswith(("http://", "https://")):
            continue
        # Только внутренние ссылки
        if base_domain in full and full not in links and len(links) < max_links:
            links.append(full)
    return links


# ── Ключевые слова для генерации знаний ───────────────────────────────
def keywordify(text: str, max_kw: int = 30) -> list[str]:
    """Извлекает ключевые слова из текста."""
    # Убираем код
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`]+`", " ", text)
    # Разбиваем на слова
    words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", text)
    # Частотный анализ
    freq: dict[str, int] = {}
    for w in words:
        wl = w.lower()
        if wl not in _STOP_WORDS:
            freq[wl] = freq.get(wl, 0) + 1
    # Сортируем по частоте
    sorted_kw = sorted(freq.items(), key=lambda x: -x[1])
    return [kw for kw, _ in sorted_kw[:max_kw]]


_STOP_WORDS = frozenset({
    "the", "and", "for", "that", "this", "with", "you", "your", "are",
    "can", "will", "not", "but", "all", "has", "had", "was", "have",
    "from", "they", "been", "said", "each", "which", "their", "will",
    "one", "would", "there", "what", "about", "into", "out", "some",
    "them", "than", "then", "other", "more", "when", "how", "its",
    "also", "just", "only", "over", "such", "after", "most", "any",
    "very", "our", "use", "are", "may", "being", "should", "could",
    "would", "does", "did", "while", "where", "those", "these",
})


# ── Состояние прогресса (для восстановления после прерывания) ─────────
PROGRESS_FILE = "data/learning_progress.json"


class LearningProgress:
    """Хранит и восстанавливает состояние обучения."""

    def __init__(self):
        self.visited_urls: set[str] = set()
        self.learned_topics: list[str] = []
        self.knowledge_count: int = 0
        self.start_depth: int = 0
        self._load()

    def _path(self) -> Path:
        return Path(PROGRESS_FILE)

    def _load(self):
        p = self._path()
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                self.visited_urls = set(data.get("visited_urls", []))
                self.learned_topics = data.get("learned_topics", [])
                self.knowledge_count = data.get("knowledge_count", 0)
                self.start_depth = data.get("start_depth", 0)
            except (json.JSONDecodeError, OSError):
                pass

    def save(self):
        p = self._path()
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "visited_urls": list(self.visited_urls),
            "learned_topics": self.learned_topics[-100:],
            "knowledge_count": self.knowledge_count,
            "start_depth": self.start_depth,
            "saved_at": datetime.now().isoformat(),
        }
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def mark_visited(self, url: str):
        self.visited_urls.add(url)

    def add_topic(self, topic: str):
        if topic not in self.learned_topics:
            self.learned_topics.append(topic)

    def reset(self):
        self.visited_urls.clear()
        self.learned_topics.clear()
        self.knowledge_count = 0
        self.start_depth = 0
        self._path().unlink(missing_ok=True)


# ── Главный класс самообучения ─────────────────────────────────────────
class SelfEducator:
    """Рекурсивный многопоточный самообучающийся модуль.

    Использование:
        edu = SelfEducator(vector_store)
        edu.start(depth=3)  # 3 уровня глубины
    """

    def __init__(self, vector_store, sites_config: str = "sites.json"):
        self.vs = vector_store
        self.config = self._load_config(sites_config)
        self.settings = self.config.get("settings", {})
        self.max_depth = self.settings.get("max_depth", 3)
        self.max_threads = self.settings.get("max_threads", 5)
        self.timeout = self.settings.get("timeout_per_page", 15)
        self.user_agent = self.settings.get("user_agent", "Python-Ai-Agent/1.0")
        self.save_interval = self.settings.get("save_interval", 10)
        self.progress = LearningProgress()
        self.topic_gen = TopicGenerator()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._knowledge_buffer: list[tuple[str, str, list[str]]] = []
        self._pages_fetched = 0
        self._pages_total = 0

    def _load_config(self, path: str) -> dict:
        p = Path(path)
        if not p.exists():
            print(f"  [!] Конфиг {path} не найден, используем дефолт")
            return {"sources": [], "settings": {"max_depth": 2, "max_threads": 3}}
        return json.loads(p.read_text(encoding="utf-8"))

    def stop(self):
        """Останавливает обучение, сохраняет прогресс."""
        self._stop_event.set()
        print("\n  [!] Получен сигнал остановки, сохраняю прогресс...")
        self._flush_buffer()
        self.progress.save()

    def _should_stop(self) -> bool:
        return self._stop_event.is_set()

    def _flush_buffer(self):
        """Сбрасывает буфер знаний в векторное хранилище."""
        with self._lock:
            if not self._knowledge_buffer:
                return
            texts = []
            metadatas = []
            for content, source, keywords in self._knowledge_buffer:
                texts.append(content)
                metadatas.append({
                    "source": source,
                    "keywords": keywords,
                    "learned_at": datetime.now().isoformat(),
                })
            self.vs.add_documents(texts, metadatas=metadatas)
            self.progress.knowledge_count += len(self._knowledge_buffer)
            self._knowledge_buffer.clear()

    def _add_knowledge(self, content: str, source: str, keywords: list[str]):
        """Добавляет знание в буфер."""
        with self._lock:
            self._knowledge_buffer.append((content, source, keywords))
            if len(self._knowledge_buffer) >= self.save_interval:
                self._flush_buffer()
                self.progress.save()

    def _process_page(self, url: str, topic: str, depth: int, progress: ProgressBar) -> list[str]:
        """Обрабатывает одну страницу: скачивает, парсит, извлекает знания."""
        if self._should_stop():
            return []
        if url in self.progress.visited_urls:
            return []

        self.progress.mark_visited(url)
        html = fetch_page(url, self.user_agent, self.timeout)
        if not html:
            return []

        text = extract_content(html)
        code_blocks = extract_code_blocks(html)

        # Сохраняем основной контент
        if len(text) > 100:
            kws = keywordify(text)
            self._add_knowledge(text[:3000], source=url, keywords=kws)
            self.topic_gen.feed_learned(kws)

        # Сохраняем блоки кода отдельно
        for block in code_blocks:
            if len(block) > 50:
                kws = keywordify(block)
                self._add_knowledge(block, source=url, keywords=kws)

        # Находим ссылки для рекурсии
        next_links = discover_links(url, html, max_links=5)
        self._pages_fetched += 1
        progress.update(1, extra=f"  тема: {topic[:40]}")
        return next_links

    def _learn_from_source(self, source: dict, depth: int, progress: ProgressBar):
        """Обучается из одного источника (рекурсивно)."""
        if self._should_stop():
            return

        seed = source.get("topic_seed", "")
        url = source["url"]
        topic = self.topic_gen.generate(seed)
        self.progress.add_topic(topic)

        print(f"\n  [тема] {topic}")
        print(f"  [url]  {url}")

        # Обрабатываем корневую страницу
        child_links = self._process_page(url, topic, depth, progress)

        # Рекурсивно обходим дочерние ссылки
        if child_links and depth < self.max_depth and not self._should_stop():
            for link in child_links:
                if self._should_stop():
                    return
                if link not in self.progress.visited_urls:
                    self._process_page(link, topic, depth + 1, progress)

    def start(self, depth: int | None = None):
        """Запускает процесс самообучения.

        Args:
            depth: максимальная глубина рекурсии (None = из конфига)
        """
        if depth is not None:
            self.max_depth = depth

        sources = self.config.get("sources", [])
        if not sources:
            print("  [!] Нет источников для обучения")
            return

        total_est = len(sources) * 8 * self.max_depth  # грубая оценка
        self._pages_total = total_est

        print(f"\n{'=' * 60}")
        print(f"  🎓 САМООБУЧЕНИЕ — {len(sources)} источников, глубина {self.max_depth}")
        print(f"  Потоков: {self.max_threads} | Таймаут: {self.timeout}с")
        print(f"  Уже изучено: {self.progress.knowledge_count} фрагментов")
        print(f"  Ctrl+C — безопасная остановка с сохранением прогресса")
        print(f"{'=' * 60}")

        progress = ProgressBar(total_est, prefix="  Прогресс:")
        interrupted = False

        try:
            # Запускаем ThreadPoolExecutor для параллельной загрузки
            with ThreadPoolExecutor(max_workers=self.max_threads-2) as pool:
                futures = []
                for src in sources:
                    if self._should_stop():
                        break
                    seed = src.get("topic_seed", "")
                    url = src["url"]
                    topic = self.topic_gen.generate(seed)
                    self.progress.add_topic(topic)

                    # Запускаем обработку в потоке
                    fut = pool.submit(self._learn_from_source, src, 0, progress)
                    futures.append((fut, topic, url))

                # Ждём завершения
                for fut, topic, url in futures:
                    if self._should_stop():
                        break
                    try:
                        fut.result(timeout=120)
                    except TimeoutError:
                        print(f"  [таймаут] {url}")
                    except Exception as e:
                        print(f"  [ошибка] {url}: {e}")

        except KeyboardInterrupt:
            interrupted = True
            self.stop()

        # Финальный сброс буфера
        self._flush_buffer()
        self.progress.save()
        progress.finish()

        print(f"\n{'=' * 60}")
        print(f"  Статистика:")
        print(f"    Страниц обработано:   {self._pages_fetched}")
        print(f"    Знаний накоплено:     {self.progress.knowledge_count}")
        print(f"    Тем изучено:         {len(self.progress.learned_topics)}")
        if interrupted:
            print(f"    Прервано! Прогресс сохранён — запустите /learn для продолжения")
        else:
            print(f"    Готово!")
        print(f"{'=' * 60}")

        return {
            "pages": self._pages_fetched,
            "knowledge": self.progress.knowledge_count,
            "topics": len(self.progress.learned_topics),
            "interrupted": interrupted,
        }

    def clear_progress(self):
        """Сбрасывает прогресс обучения."""
        self.progress.reset()
        print("  Прогресс обучения сброшен")
