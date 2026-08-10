"""LangGraph workflow for the EvidenceFlow AI document assistant."""

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from .schemas import AnswerResponse, CalculationResponse, SummarizationResponse, UpdateMemoryResponse, UserIntent
from .prompts import MEMORY_SUMMARY_PROMPT, get_chat_prompt_template, get_intent_classification_prompt


def reduce_actions(previous: List[str], current: List[str]) -> List[str]:
    """Append workflow actions or begin a fresh turn when requested."""
    if current and current[0] == "__reset_actions__":
        return current[1:]
    return previous + current


class AgentState(TypedDict):
    """State shared by all workflow nodes."""

    user_input: Optional[str]
    messages: Annotated[List[BaseMessage], add_messages]
    intent: Optional[UserIntent]
    next_step: str
    conversation_summary: str
    active_documents: Optional[List[str]]
    current_response: Optional[Dict[str, Any]]
    tools_used: List[str]
    session_id: Optional[str]
    user_id: Optional[str]
    actions_taken: Annotated[List[str], reduce_actions]


def _configurable(config: RunnableConfig) -> Dict[str, Any]:
    configurable = config.get("configurable", {})
    if not configurable.get("llm"):
        raise ValueError("The workflow requires an LLM in config['configurable']['llm'].")
    return configurable


def _history_as_text(messages: List[BaseMessage]) -> str:
    if not messages:
        return "No previous conversation."
    return "\n".join(f"{message.type}: {message.content}" for message in messages)


def _model_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return {"response": str(value)}


def invoke_react_agent(
    response_schema: type[BaseModel], messages: List[BaseMessage], llm: Any, tools: List[Any]
) -> tuple[Dict[str, Any], List[str]]:
    """Run a structured ReAct agent and return its trace plus used tools."""
    agent = create_react_agent(model=llm, tools=tools, response_format=response_schema)
    result = agent.invoke({"messages": messages})
    existing_tool_messages = {
        (message.id, message.tool_call_id)
        for message in messages
        if isinstance(message, ToolMessage)
    }
    tools_used = [
        message.name
        for message in result.get("messages", [])
        if isinstance(message, ToolMessage) and (message.id, message.tool_call_id) not in existing_tool_messages
    ]
    return result, tools_used


def _enforce_calculator_use(result: Dict[str, Any], tools: List[Any], tools_used: List[str]) -> tuple[Dict[str, Any], List[str]]:
    """Run the trusted calculator when a model did not call it itself."""
    structured_response = result.get("structured_response")
    if not structured_response:
        raise ValueError("The calculation agent did not return a structured calculation response.")

    calculation = (
        structured_response
        if isinstance(structured_response, CalculationResponse)
        else CalculationResponse.model_validate(structured_response)
    )
    if "calculator" in tools_used:
        return result, tools_used

    calculator = next((tool for tool in tools if getattr(tool, "name", None) == "calculator"), None)
    if calculator is None:
        raise ValueError("The calculation agent requires the calculator tool.")

    calculator_result = calculator.invoke({"expression": calculation.expression})
    if str(calculator_result).startswith("Calculation error:"):
        raise ValueError(str(calculator_result))

    result["structured_response"] = calculation.model_copy(update={"result": float(calculator_result)})
    return result, tools_used + ["calculator"]


def classify_intent(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Classify a request and choose the specialist node."""
    llm = _configurable(config)["llm"]
    prompt = get_intent_classification_prompt().format(
        user_input=state.get("user_input") or "",
        conversation_history=_history_as_text(state.get("messages", [])),
    )
    response = llm.with_structured_output(UserIntent).invoke(prompt)
    intent = response if isinstance(response, UserIntent) else UserIntent.model_validate(response)
    next_step = {
        "qa": "qa_agent",
        "summarization": "summarization_agent",
        "calculation": "calculation_agent",
    }.get(intent.intent_type, "qa_agent")
    return {"actions_taken": ["classify_intent"], "intent": intent, "next_step": next_step}


def _run_specialist(
    state: AgentState,
    config: RunnableConfig,
    intent_type: str,
    response_schema: type[BaseModel],
    action_name: str,
) -> Dict[str, Any]:
    configurable = _configurable(config)
    prompt_messages = get_chat_prompt_template(intent_type).invoke(
        {"input": state.get("user_input") or "", "chat_history": state.get("messages", [])}
    ).to_messages()
    result, tools_used = invoke_react_agent(response_schema, prompt_messages, configurable["llm"], configurable.get("tools", []))
    if response_schema is CalculationResponse:
        result, tools_used = _enforce_calculator_use(result, configurable.get("tools", []), tools_used)
    return {
        "messages": result.get("messages", []),
        "actions_taken": [action_name],
        "current_response": _model_dict(result.get("structured_response", result)),
        "tools_used": tools_used,
        "next_step": "update_memory",
    }


def qa_agent(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Answer a document question with a structured response."""
    return _run_specialist(state, config, "qa", AnswerResponse, "qa_agent")


def summarization_agent(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Summarize relevant documents with a structured response."""
    return _run_specialist(state, config, "summarization", SummarizationResponse, "summarization_agent")


def calculation_agent(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Calculate values found in relevant documents with a structured response."""
    return _run_specialist(state, config, "calculation", CalculationResponse, "calculation_agent")


def update_memory(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Create a typed conversation summary and retain referenced documents."""
    llm = _configurable(config)["llm"]
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(MEMORY_SUMMARY_PROMPT),
            MessagesPlaceholder("chat_history"),
        ]
    ).invoke({"chat_history": state.get("messages", [])})
    response = llm.with_structured_output(UpdateMemoryResponse).invoke(prompt)
    memory = response if isinstance(response, UpdateMemoryResponse) else UpdateMemoryResponse.model_validate(response)
    document_ids = list(dict.fromkeys((state.get("active_documents") or []) + memory.document_ids))
    return {
        "conversation_summary": memory.summary,
        "active_documents": document_ids,
        "actions_taken": ["update_memory"],
        "next_step": "end",
    }


def should_continue(state: AgentState) -> str:
    """Return the node selected by intent classification."""
    return state.get("next_step", "qa_agent")


def create_workflow(llm: Any, tools: List[Any]):
    """Build and compile the persistent LangGraph workflow."""
    workflow = StateGraph(AgentState)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("qa_agent", qa_agent)
    workflow.add_node("summarization_agent", summarization_agent)
    workflow.add_node("calculation_agent", calculation_agent)
    workflow.add_node("update_memory", update_memory)

    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        should_continue,
        {
            "qa_agent": "qa_agent",
            "summarization_agent": "summarization_agent",
            "calculation_agent": "calculation_agent",
        },
    )
    workflow.add_edge("qa_agent", "update_memory")
    workflow.add_edge("summarization_agent", "update_memory")
    workflow.add_edge("calculation_agent", "update_memory")
    workflow.add_edge("update_memory", END)
    try:
        serializer = JsonPlusSerializer(allowed_msgpack_modules=[("src.schemas", "UserIntent")])
    except TypeError:
        serializer = JsonPlusSerializer()
    return workflow.compile(checkpointer=InMemorySaver(serde=serializer))
