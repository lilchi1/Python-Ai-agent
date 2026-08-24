"""
Тесты для агента.
"""
import pytest
from src.llm.local_llm import LocalLLM
from src.memory.vector_store import LocalVectorStore
from src.agent.coding_agent import CodingAgent


@pytest.fixture
def agent():
    llm = LocalLLM("deepseek-ai/deepseek-coder-1.3b-instruct")
    vector_store = LocalVectorStore(collection_name="test_collection")
    return CodingAgent(llm, vector_store, max_attempts=2)


def test_agent_initialization(agent):
    assert agent.max_attempts == 2
    assert agent.vector_store is not None
    assert agent.llm is not None


def test_agent_solve_simple(agent):
    result = agent.solve("Напиши функцию, которая возвращает сумму двух чисел")
    assert "code" in result
    assert "success" in result


def test_agent_with_knowledge(agent):
    agent.vector_store.add_documents([
        "Задача: Напиши функцию sum_two(a, b)\nРешение: return a + b"
    ])

    result = agent.solve("Напиши функцию, которая складывает два числа")
    assert "code" in result