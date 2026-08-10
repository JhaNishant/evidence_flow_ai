"""Application service that connects sessions, tools, and the LangGraph workflow."""

from datetime import datetime
import json
import os
from typing import Any, Dict, List, Optional
import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from .agent import AgentState, create_workflow
from .retrieval import SimulatedRetriever
from .schemas import SessionState
from .tools import ToolLogger, get_all_tools


class DocumentAssistant:
    """Manage document assistant sessions and persistent workflow state."""

    def __init__(
        self,
        openai_api_key: str,
        model_name: str = "gpt-4o",
        temperature: float = 0.1,
        reasoning_effort: Optional[str] = None,
        session_storage_path: str = "./sessions",
        logs_dir: str = "./logs",
    ):
        model_options: Dict[str, Any] = {}
        selected_reasoning_effort = reasoning_effort or ("none" if model_name.startswith("gpt-5") else None)
        if selected_reasoning_effort:
            model_options["reasoning_effort"] = selected_reasoning_effort

        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            temperature=temperature,
            base_url="https://openai.vocareum.com/v1",
            **model_options,
        )
        self.retriever = SimulatedRetriever()
        self.session_storage_path = session_storage_path
        self.logs_dir = logs_dir
        os.makedirs(self.session_storage_path, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.current_session: Optional[SessionState] = None
        self.tool_logger = ToolLogger(logs_dir=self.logs_dir)
        self.tools = get_all_tools(self.retriever, self.tool_logger)
        self.workflow = create_workflow(self.llm, self.tools)

    def _configure_session_runtime(self, session_id: str) -> None:
        """Bind the active session to its own tool log and workflow."""
        self.tool_logger = ToolLogger(logs_dir=self.logs_dir, session_id=session_id)
        self.tools = get_all_tools(self.retriever, self.tool_logger)
        self.workflow = create_workflow(self.llm, self.tools)

    def start_session(self, user_id: str, session_id: Optional[str] = None) -> str:
        """Start a new session or load a saved one."""
        if session_id and self._session_exists(session_id):
            self.current_session = self._load_session(session_id)
            print(f"Resumed session {session_id}")
        else:
            session_id = session_id or str(uuid.uuid4())
            self.current_session = SessionState(
                session_id=session_id,
                user_id=user_id,
                conversation_history=[],
                document_context=[],
            )
            print(f"Started new session {session_id}")
        self._configure_session_runtime(self.current_session.session_id)
        return self.current_session.session_id

    def _session_path(self, session_id: str) -> str:
        return os.path.join(self.session_storage_path, f"{session_id}.json")

    def _session_exists(self, session_id: str) -> bool:
        return os.path.exists(self._session_path(session_id))

    def _load_session(self, session_id: str) -> SessionState:
        with open(self._session_path(session_id), "r", encoding="utf-8") as file:
            return SessionState.model_validate(json.load(file))

    def _save_session(self) -> None:
        if not self.current_session:
            return
        with open(self._session_path(self.current_session.session_id), "w", encoding="utf-8") as file:
            json.dump(self.current_session.model_dump(mode="json"), file, indent=2)

    def _saved_history_messages(self) -> List[BaseMessage]:
        if not self.current_session:
            return []
        messages: List[BaseMessage] = []
        for turn in self.current_session.conversation_history:
            user_input = turn.get("user_input")
            assistant_response = turn.get("assistant_response")
            if user_input:
                messages.append(HumanMessage(content=user_input))
            if assistant_response:
                messages.append(AIMessage(content=assistant_response))
        return messages

    def _get_conversation_summary(self) -> str:
        if not self.current_session or not self.current_session.conversation_history:
            return "No previous conversation."
        return str(self.current_session.conversation_history[-1].get("summary", "No previous conversation."))

    @staticmethod
    def _display_response(final_state: Dict[str, Any]) -> str:
        response = final_state.get("current_response") or {}
        if isinstance(response, dict):
            if response.get("answer"):
                return str(response["answer"])
            if response.get("summary"):
                return str(response["summary"])
            if response.get("explanation"):
                result = response.get("result")
                return f"{response['explanation']}\nResult: {result}" if result is not None else str(response["explanation"])

        for message in reversed(final_state.get("messages", [])):
            if isinstance(message, AIMessage) and message.content:
                return str(message.content)
        return "No response was produced."

    def process_message(self, user_input: str) -> Dict[str, Any]:
        """Run one user message through the persistent LangGraph workflow."""
        if not self.current_session:
            raise ValueError("No active session. Call start_session() first.")

        config = {
            "configurable": {
                "thread_id": self.current_session.session_id,
                "llm": self.llm,
                "tools": self.tools,
            }
        }
        persisted_state = self.workflow.get_state(config).values
        initial_state: AgentState = {
            "messages": [] if persisted_state else self._saved_history_messages(),
            "user_input": user_input,
            "intent": None,
            "next_step": "classify_intent",
            "conversation_summary": persisted_state.get("conversation_summary", self._get_conversation_summary()),
            "active_documents": persisted_state.get("active_documents", self.current_session.document_context),
            "current_response": None,
            "tools_used": [],
            "session_id": self.current_session.session_id,
            "user_id": self.current_session.user_id,
            "actions_taken": ["__reset_actions__"],
        }

        try:
            final_state = self.workflow.invoke(initial_state, config=config)
            response_text = self._display_response(final_state)
            intent = final_state.get("intent")
            intent_data = intent.model_dump(mode="json") if intent else None
            turn_record = {
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "assistant_response": response_text,
                "intent": intent_data,
                "sources": final_state.get("active_documents", []),
                "tools_used": final_state.get("tools_used", []),
                "actions_taken": final_state.get("actions_taken", []),
                "summary": final_state.get("conversation_summary", ""),
            }
            self.current_session.conversation_history.append(turn_record)
            self.current_session.last_updated = datetime.now()
            self.current_session.document_context = list(
                dict.fromkeys(self.current_session.document_context + final_state.get("active_documents", []))
            )
            self._save_session()
            return {
                "success": True,
                "response": response_text,
                "intent": intent_data,
                "tools_used": final_state.get("tools_used", []),
                "active_documents": final_state.get("active_documents", []),
                "actions_taken": final_state.get("actions_taken", []),
                "summary": final_state.get("conversation_summary", ""),
            }
        except Exception as error:
            return {"success": False, "error": str(error), "response": None}
