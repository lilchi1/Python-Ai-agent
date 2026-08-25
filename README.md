# Self-Dev Code Agent

Self-developing code generation agent. Zero external dependencies — pure Python 3 stdlib only.

## Quick Start

```bash
python run.py
```

## Interactive Commands

| Command | Description |
|---------|-------------|
| `/lang` | List 22 supported languages |
| `/init` | Seed knowledge base with examples |
| `/stats` | Show session statistics |
| `/history` | Show recent tasks |
| `/search <q>` | Search archive |
| `/help` | Show help |
| `/quit` | Exit |

## Features

- **22 languages**: Python, JavaScript, TypeScript, C, C++, Java, Go, Rust, PHP, Ruby, Swift, Kotlin, C#, Lua, Haskell, SQL, Bash, YAML, JSON, HTML, CSS, Markdown
- **Self-learning**: saves solutions to JSON knowledge base, uses TF-IDF to find similar past solutions
- **Auto language detection**: detects language from task description (supports Russian and English)
- **Code evaluation**: validates syntax, detects stubs, checks safety
- **Zero dependencies**: no pip install needed, runs on any Python 3.6+

## Architecture

```
run.py              → Interactive CLI
src/
  agent/            → Coding agent (orchestrates generate + evaluate + learn)
  evaluator/        → Multi-language code evaluator
  generator/        → Template-based code generator with task parsing
  languages/        → Language registry (syntax, keywords, templates for 22 languages)
  memory/           → TF-IDF vector store with JSON persistence
  orchestrator/     → Execution controller + archive
```

## Examples

```
task> Write factorial function in Python
task> Напиши функциюFizzBuzz на Go
task> Create a binary search in Rust
```
