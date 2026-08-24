"""
Локальная LLM для генерации кода и планирования.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")


class LocalLLM:
    """Обертка над локальной моделью для инференса."""

    def __init__(
        self,
        model_name: str = "deepseek-ai/deepseek-coder-1.3b-instruct",
        device: str = None,
        load_in_8bit: bool = False
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"🚀 Загрузка модели {model_name} на {self.device}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="left"
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            load_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True
            }

            if load_in_8bit and self.device == "cuda":
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                except ImportError:
                    print("⚠️ bitsandbytes не установлен, загрузка в fp16")

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs
            )

            if self.device == "cpu":
                self.model = self.model.to("cpu")

            self.model.eval()
            print(f"✅ Модель загружена. Параметров: {self.model.num_parameters():,}")

        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            print("⚠️ Создаем заглушку для тестирования...")
            self.model = None
            self.tokenizer = None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        stop_strings: list = None,
        **kwargs
    ) -> str:
        """Генерация текста."""
        if self.model is None:
            # Заглушка для тестирования
            print("⚠️ Модель не загружена, возвращаем тестовый ответ")
            return self._mock_generate(prompt)

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if generated.startswith(prompt):
                generated = generated[len(prompt):].lstrip()

            if stop_strings:
                for stop in stop_strings:
                    if stop in generated:
                        generated = generated.split(stop)[0]

            return generated.strip()

        except Exception as e:
            print(f"⚠️ Ошибка генерации: {e}")
            return self._mock_generate(prompt)

    def _mock_generate(self, prompt: str) -> str:
        """Заглушка для тестирования без модели."""
        if "факториал" in prompt.lower():
            return """```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```"""
        elif "сумму" in prompt.lower() or "сложение" in prompt.lower():
            return """```python
def sum_two(a, b):
    return a + b
```"""
        else:
            return """```python
def solution():
    # Ваш код здесь
    pass
```"""

    def generate_code(self, task: str, context: str = "", max_tokens: int = 512) -> str:
        """Генерация кода для задачи."""
        if self.model is None:
            return self._mock_generate(task)

        prompt = f"""Ты — ассистент по программированию на Python.
Реши задачу, написав только код.

Задача: {task}

"""
        if context:
            prompt += f"""Контекст (похожие решения):
{context}

"""
        prompt += """Напиши код на Python. Используй только код, без пояснений.
```python
"""
        response = self.generate(prompt, max_new_tokens=max_tokens, temperature=0.2)

        return self._extract_code(response)

    def generate_plan(self, task: str, context: str = "") -> list:
        """Генерация плана решения задачи."""
        if self.model is None:
            return ["1. Проанализировать задачу", "2. Написать решение", "3. Проверить результат"]

        prompt = f"""Ты — ассистент по программированию на Python.
Составь план решения задачи.

Задача: {task}

"""
        if context:
            prompt += f"""Контекст (похожие решения):
{context}

"""
        prompt += """План решения (шаг за шагом):
"""
        response = self.generate(prompt, max_new_tokens=256, temperature=0.3)

        steps = []
        for line in response.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("*")):
                clean = line.lstrip("0123456789.-* ").strip()
                if clean:
                    steps.append(clean)
            elif line and not line.startswith("```"):
                steps.append(line)

        return steps if steps else [response]

    def _extract_code(self, text: str) -> str:
        """Извлечение кода из ответа."""
        if not text:
            return ""

        if "```python" in text:
            parts = text.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0].strip()
                if code:
                    return code

        if "```" in text:
            parts = text.split("```")
            if len(parts) > 1:
                code = parts[1].strip()
                if code and not code.startswith("python"):
                    return code

        if "def " in text or "class " in text:
            return text.strip()

        return ""