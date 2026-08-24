"""
Оркестратор — управляет потоком выполнения.
"""
from typing import Dict, Any
from src.agent.coding_agent import CodingAgent


class Orchestrator:
    def __init__(self, agent: CodingAgent):
        self.agent = agent
        self.history = []

    def execute(self, task: str) -> Dict[str, Any]:
        print("\n" + "=" * 60)
        print("🤖 ЗАПУСК ОРКЕСТРАТОРА")
        print("=" * 60)

        result = self.agent.solve(task)

        self.history.append({
            "task": task,
            "result": result
        })

        self._print_result(result)

        return result

    def _print_result(self, result: Dict[str, Any]):
        print("\n" + "=" * 60)
        print("📊 РЕЗУЛЬТАТ")
        print("=" * 60)

        if result["success"]:
            print("✅ УСПЕШНО!")
            print(f"\nКод ({len(result['code'])} символов):")
            print("-" * 40)
            print(result["code"])
            print("-" * 40)
        else:
            print("❌ НЕУДАЧНО")
            print(f"\nОшибка: {result['feedback']}")

        print(f"\nПопыток: {result['attempts']}")

        if len(result.get("history", [])) > 1:
            print("\n📜 История попыток:")
            for i, attempt in enumerate(result["history"], 1):
                status = "✅" if attempt["success"] else "❌"
                print(f"  {i}. {status} {attempt['feedback'][:50]}...")

    def clear_history(self):
        self.history = []

    def get_stats(self) -> Dict[str, Any]:
        total = len(self.history)
        successful = sum(1 for h in self.history if h["result"]["success"])
        return {
            "total_tasks": total,
            "successful": successful,
            "success_rate": successful / total if total > 0 else 0
        }