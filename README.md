# EvidenceFlow AI

EvidenceFlow AI is a multi agent document intelligence assistant built with LangChain and LangGraph. It answers document questions, creates summaries, and performs source grounded calculations across financial and healthcare records.

## What it does

- **Q&A** answers a specific question after searching and reading relevant records.
- **Summarization** extracts the key points, numbers, dates, and document IDs from selected records.
- **Calculation** retrieves the necessary documents, then sends every arithmetic expression through the calculator tool.

The included in memory document collection contains invoices, a service agreement, and an insurance claim so the project runs without an external database.

## Architecture

```text
user message
    |
intent classification
    |
    +--> qa_agent
    +--> summarization_agent
    +--> calculation_agent
              |
         update_memory
              |
             end
```

`create_workflow` builds this `StateGraph` and compiles it with `InMemorySaver`. The session ID is the LangGraph thread ID, so a running assistant preserves graph state across messages in the same session.

## Structured output and safety

Pydantic models ensure every important LLM response has a known shape:

- `UserIntent` only accepts `qa`, `summarization`, `calculation`, or `unknown`, with confidence from 0 through 1.
- `AnswerResponse` includes the original question, answer, sources, confidence, and a timestamp.
- Specialist and memory nodes use their own typed responses for summaries, calculations, and retained document IDs.

The calculator parses expressions with Python's abstract syntax tree before using `eval`. It permits numbers, parentheses, and basic arithmetic only. Names, function calls, attributes, imports, and nonfinite results are rejected. It always returns text, so an integer result is `"5"`, not `5`.

## Setup

Requirements: Python 3.9 or newer and an OpenAI API key with access to the configured endpoint.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then start the assistant:

```bash
python3 main.py
```

The `/docs` command shows the available sample documents.

## Memory, sessions, and logs

Each call to `start_session` creates or restores a session ID. The assistant stores a JSON safe history of user inputs, responses, intent, sources, tools, actions, and summaries in `sessions/<session_id>.json`.

Every tool invocation is automatically written to `logs/session_<session_id>.json`. These runtime artifacts and `.env` are ignored by Git so no secrets or personal session data are published.

## Example conversations

**Q&A**

```text
Question: What is the total due on invoice INV-002?
Response: INV-002 has a total due of $69,300.
Sources: INV-002
```

**Summarization**

```text
Question: Summarize the service agreement.
Response: The agreement provides document platform access, support, analytics, and compliance monitoring for 12 months at $15,000 per month. Its total value is $180,000 and either party may terminate with 60 days written notice.
Sources: CON-001
```

**Calculation**

```text
Question: Calculate the sum of the service items in invoice INV-001.
Response: The calculator evaluated 5000 + 12500 + 2500, for a result of $20,000.
Sources: INV-001
Tools used: document_search, document_reader, calculator
```

## Verification

Run the offline test suite without an API key:

```bash
python3 -m unittest discover -s tests -v
```

The tests verify schema constraints, prompt selection, calculator safety and logging, all workflow routes, unknown intent fallback, and memory updates. For a live smoke test, run the command line application with a valid key and exercise one request from each example above.
