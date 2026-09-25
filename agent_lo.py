import wikipedia
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, wrap_tool_call
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from pydantic import BaseModel, Field

load_dotenv()

# Wikipedia rate-limits (HTTP 429) the wikipedia package's shared default User-Agent
wikipedia.set_user_agent("research-assistant/1.0 (portfolio project)")

MEMORY = InMemorySaver()


def get_tools():
    # Short snippets make the model keep re-searching for details, so return full-ish abstracts
    arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=2500))
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1500))
    search = DuckDuckGoSearchRun(name="search")
    return [wiki, arxiv, search]


@wrap_tool_call
def tool_errors_to_model(request, handler):
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(f"Tool failed: {e}. Try a different tool.", tool_call_id=request.tool_call["id"])


class State(MessagesState):
    plan: list[str]
    draft: str
    critique: str


class Plan(BaseModel):
    steps: list[str] = Field(description="2-4 short research steps; each says what to look up and which source fits best (wikipedia, arxiv, or web search)")


class Reflection(BaseModel):
    critique: str = Field(description=(
        "Reflection report in markdown with three sections: "
        "**Strengths** (what the draft got right and which sources it used), "
        "**Limitations** (missing information, unsupported claims, missing citations, unclear or off-topic parts), "
        "**Suggestions** (the concrete fixes applied in the revised answer)"
    ))
    revised_answer: str = Field(description="Improved final answer that applies every suggestion from the report")


PLAN_PROMPT = """You are a research planner. Read the conversation and write a short plan
to answer the user's latest question. Use earlier messages to resolve references like "it" or "that paper"."""

RESEARCH_PROMPT = """You are a research assistant with web, Wikipedia and arXiv tools.
Use the conversation for context and follow this plan to answer the user's latest question:

{plan}

Base your answer on tool results and mention which source each fact came from."""

REFLECT_PROMPT = """You are a strict reviewer. Critique the draft answer to the question below,
then rewrite it to fix every issue. Keep facts that came from the draft; do not invent sources or data.

Question: {question}

Draft answer:
{draft}"""


def build_graph(llm):
    researcher_agent = create_agent(
        llm,
        get_tools(),
        # After 6 tool calls further calls are rejected, so the model answers with what it has
        middleware=[tool_errors_to_model, ToolCallLimitMiddleware(run_limit=6)],
    )

    def planner(state: State):
        plan = llm.with_structured_output(Plan, method="function_calling").invoke(
            [SystemMessage(PLAN_PROMPT), *state["messages"]]
        )
        return {"plan": plan.steps}

    def researcher(state: State):
        steps = "\n".join(f"{i}. {s}" for i, s in enumerate(state["plan"], 1))
        result = researcher_agent.invoke(
            {"messages": [SystemMessage(RESEARCH_PROMPT.format(plan=steps)), *state["messages"]]},
            {"recursion_limit": 25},
        )
        return {"draft": result["messages"][-1].content}

    def reviewer(state: State):
        question = state["messages"][-1].content
        review = llm.with_structured_output(Reflection, method="function_calling").invoke(
            REFLECT_PROMPT.format(question=question, draft=state["draft"])
        )
        return {"critique": review.critique, "messages": [AIMessage(review.revised_answer)]}

    graph = StateGraph(State)
    graph.add_sequence([planner, researcher, reviewer])
    graph.add_edge(START, "planner")
    graph.add_edge("reviewer", END)
    return graph.compile(checkpointer=MEMORY)


# Reads OPENAI_API_KEY and OPENAI_BASE_URL from .env
GRAPH = build_graph(ChatOpenAI(model="deepseek-v4-flash", temperature=0))


def run(query: str, thread_id: str) -> dict:
    state = GRAPH.invoke(
        {"messages": [HumanMessage(query)]},
        {"configurable": {"thread_id": thread_id}},
    )
    return {
        "answer": state["messages"][-1].content,
        "plan": state["plan"],
        "draft": state["draft"],
        "critique": state["critique"],
    }