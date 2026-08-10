"""Offline verification for EvidenceFlow AI."""

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from src.agent import create_workflow
from src.assistant import DocumentAssistant
from src.prompts import CALCULATION_SYSTEM_PROMPT, get_chat_prompt_template, get_intent_classification_prompt
from src.schemas import AnswerResponse, CalculationResponse, SummarizationResponse, UpdateMemoryResponse, UserIntent
from src.tools import ToolLogger, create_calculator_tool


class FakeStructuredCall:
    def __init__(self, schema, intent_type):
        self.schema = schema
        self.intent_type = intent_type

    def invoke(self, _prompt):
        if self.schema is UserIntent:
            return UserIntent(intent_type=self.intent_type, confidence=0.91, reasoning="Offline test routing")
        if self.schema is UpdateMemoryResponse:
            return UpdateMemoryResponse(summary="Discussed invoice INV-002.", document_ids=["INV-002"])
        raise AssertionError(f"Unexpected structured schema: {self.schema}")


class FakeLLM:
    def __init__(self, intent_type):
        self.intent_type = intent_type

    def with_structured_output(self, schema):
        return FakeStructuredCall(schema, self.intent_type)


class FakeCalculator:
    name = "calculator"

    def invoke(self, values):
        return str(eval(values["expression"], {"__builtins__": {}}, {}))


def fake_react_agent(response_schema, _messages, _llm, _tools):
    if response_schema is AnswerResponse:
        structured = AnswerResponse(
            question="What is the invoice total?",
            answer="INV-002 totals $69,300.",
            sources=["INV-002"],
            confidence=0.95,
        )
    elif response_schema is SummarizationResponse:
        structured = SummarizationResponse(
            original_length=100,
            summary="Invoice INV-002 is due for $69,300.",
            key_points=["Total due is $69,300."],
            document_ids=["INV-002"],
        )
    else:
        structured = CalculationResponse(
            expression="5000 + 12500 + 2500",
            result=20000,
            explanation="The invoice service items total $20,000.",
        )
    return {"messages": [AIMessage(content="Offline specialist response")], "structured_response": structured}, []


class SchemaTests(unittest.TestCase):
    def test_answer_response_enforces_confidence_range(self):
        response = AnswerResponse(question="Q", answer="A", confidence=0.5)
        self.assertEqual(response.sources, [])
        for invalid_confidence in (-0.1, 1.1):
            with self.subTest(confidence=invalid_confidence), self.assertRaises(ValidationError):
                AnswerResponse(question="Q", answer="A", confidence=invalid_confidence)

    def test_user_intent_restricts_categories(self):
        self.assertEqual(UserIntent(intent_type="qa", confidence=0.7, reasoning="Question").intent_type, "qa")
        with self.assertRaises(ValidationError):
            UserIntent(intent_type="chat", confidence=0.7, reasoning="Invalid")
        with self.assertRaises(ValidationError):
            UserIntent(intent_type="qa", confidence=1.1, reasoning="Invalid confidence")


class PromptAndToolTests(unittest.TestCase):
    def test_all_chat_prompt_types_are_available(self):
        for intent_type in ("qa", "summarization", "calculation", "unknown"):
            messages = get_chat_prompt_template(intent_type).invoke({"input": "Hello", "chat_history": []}).to_messages()
            self.assertEqual(messages[-1].content, "Hello")
        self.assertIn("calculator tool", CALCULATION_SYSTEM_PROMPT.lower())

    def test_intent_prompt_includes_categories_and_examples(self):
        prompt = get_intent_classification_prompt().format(user_input="Hello", conversation_history="None")
        self.assertIn("summarization", prompt)
        self.assertIn("What is the total due on INV-002?", prompt)

    def test_calculator_returns_strings_and_logs_safe_and_unsafe_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = ToolLogger(logs_dir=directory, session_id="test")
            calculator = create_calculator_tool(logger)
            self.assertEqual(calculator.invoke({"expression": "2 + 3"}), "5")
            self.assertEqual(calculator.invoke({"expression": "5 / 2"}), "2.5")
            self.assertEqual(calculator.invoke({"expression": "$70,000 - $7,000 + $6,300"}), "69300")
            self.assertIn("Calculation error", calculator.invoke({"expression": "__import__('os').system('echo unsafe')"}))
            self.assertIn("Calculation error", calculator.invoke({"expression": "2 ** 101"}))
            self.assertIn("Calculation error", calculator.invoke({"expression": "2" * 251}))
            self.assertEqual(len(logger.get_logs()), 6)
            self.assertTrue((Path(directory) / "session_test.json").exists())


class WorkflowTests(unittest.TestCase):
    def _run_workflow(self, intent_type):
        tools = [FakeCalculator()]
        workflow = create_workflow(FakeLLM(intent_type), tools)
        initial_state = {
            "messages": [],
            "user_input": "Process INV-002",
            "intent": None,
            "next_step": "classify_intent",
            "conversation_summary": "No previous conversation.",
            "active_documents": [],
            "current_response": None,
            "tools_used": [],
            "session_id": "offline_test",
            "user_id": "tester",
            "actions_taken": [],
        }
        with patch("src.agent.invoke_react_agent", side_effect=fake_react_agent):
            return workflow.invoke(initial_state, config={"configurable": {"thread_id": intent_type, "llm": FakeLLM(intent_type), "tools": tools}})

    def test_qa_route_reaches_memory(self):
        final_state = self._run_workflow("qa")
        self.assertEqual(final_state["intent"].intent_type, "qa")
        self.assertEqual(final_state["actions_taken"], ["classify_intent", "qa_agent", "update_memory"])
        self.assertEqual(final_state["active_documents"], ["INV-002"])

    def test_calculation_route_reaches_memory(self):
        final_state = self._run_workflow("calculation")
        self.assertEqual(final_state["actions_taken"], ["classify_intent", "calculation_agent", "update_memory"])
        self.assertIn("calculator", final_state["tools_used"])

    def test_summarization_route_reaches_memory(self):
        final_state = self._run_workflow("summarization")
        self.assertEqual(final_state["actions_taken"], ["classify_intent", "summarization_agent", "update_memory"])

    def test_unknown_route_falls_back_to_qa(self):
        final_state = self._run_workflow("unknown")
        self.assertIn("qa_agent", final_state["actions_taken"])


class SessionTests(unittest.TestCase):
    def test_session_history_is_json_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            sessions = Path(directory) / "sessions"
            logs = Path(directory) / "logs"
            with patch("src.assistant.ChatOpenAI", return_value=FakeLLM("qa")), patch(
                "src.agent.invoke_react_agent", side_effect=fake_react_agent
            ):
                document_assistant = DocumentAssistant(
                    "test_key", session_storage_path=str(sessions), logs_dir=str(logs)
                )
                session_id = document_assistant.start_session("tester", session_id="saved_session")
                result = document_assistant.process_message("What is the invoice total?")
                second_result = document_assistant.process_message("Who is the client?")

            self.assertTrue(result["success"])
            self.assertTrue(second_result["success"])
            session_path = sessions / f"{session_id}.json"
            self.assertTrue(session_path.exists())
            saved_session = json.loads(session_path.read_text())
            self.assertEqual(saved_session["conversation_history"][0]["intent"]["intent_type"], "qa")
            self.assertEqual(
                saved_session["conversation_history"][1]["actions_taken"],
                ["classify_intent", "qa_agent", "update_memory"],
            )
            with patch("src.assistant.ChatOpenAI", return_value=FakeLLM("qa")):
                resumed_assistant = DocumentAssistant(
                    "test_key", session_storage_path=str(sessions), logs_dir=str(logs)
                )
                resumed_assistant.start_session("different_user", session_id=session_id)
            self.assertEqual(len(resumed_assistant.current_session.conversation_history), 2)


class RuntimeEvidenceTests(unittest.TestCase):
    def test_sample_session_covers_every_intent(self):
        project_root = Path(__file__).resolve().parents[1]
        session_path = project_root / "sessions" / "evidence_flow_demo.json"
        log_path = project_root / "logs" / "session_evidence_flow_demo.json"

        self.assertTrue(session_path.exists())
        self.assertTrue(log_path.exists())

        session_data = json.loads(session_path.read_text(encoding="utf-8"))
        tool_logs = json.loads(log_path.read_text(encoding="utf-8"))
        intents = [turn["intent"]["intent_type"] for turn in session_data["conversation_history"]]
        tool_names = [entry["tool_name"] for entry in tool_logs]

        self.assertEqual(intents, ["qa", "summarization", "calculation"])
        self.assertIn("calculator", tool_names)
        self.assertTrue(all(entry.get("timestamp") for entry in tool_logs))


if __name__ == "__main__":
    unittest.main()
