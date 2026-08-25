"""
Оркестратор — управляет потоком выполнения и архивом решений.
"""
import json
import os
from typing import Dict, Any, List, Optional

from src.agent.coding_agent import CodingAgent
from src.languages.registry import LanguageRegistry


class Orchestrator:
    def __init__(self, agent: CodingAgent):
        self.agent = agent
        self.history: List[Dict] = []
        self.archive_path = os.path.join("data", "knowledge", "archive.json")
        self._load_archive()

    def _load_archive(self):
        if os.path.exists(self.archive_path):
            try:
                with open(self.archive_path, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
            except Exception:
                self.history = []

    def _save_archive(self):
        os.makedirs(os.path.dirname(self.archive_path), exist_ok=True)
        with open(self.archive_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def execute(self, task: str, language: Optional[str] = None) -> Dict[str, Any]:
        result = self.agent.solve(task, language)
        self.history.append({'task': task, 'result': result})
        self._save_archive()
        return result

    def get_stats(self) -> Dict[str, Any]:
        total = len(self.history)
        successful = sum(1 for h in self.history if h['result'].get('success'))
        langs_used = {}
        for h in self.history:
            lang = h['result'].get('language', 'unknown')
            langs_used[lang] = langs_used.get(lang, 0) + 1
        return {
            'total_tasks': total,
            'successful': successful,
            'success_rate': successful / total if total else 0,
            'languages_used': langs_used,
        }

    def get_history(self, n: Optional[int] = None) -> List[Dict]:
        if n:
            return self.history[-n:]
        return list(self.history)

    def search_archive(self, query: str, language: Optional[str] = None) -> List[Dict]:
        results = []
        query_lower = query.lower()
        for h in self.history:
            task = h.get('task', '').lower()
            lang = h['result'].get('language', '')
            if query_lower in task:
                if language is None or lang == language:
                    results.append(h)
        return results
