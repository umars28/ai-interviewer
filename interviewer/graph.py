from functools import partial

from langgraph.graph import END, StateGraph

from interviewer.llm import LLMClient
from interviewer.nodes import (
    ask,
    assess,
    critic,
    plan_question,
    prepare,
    route_after_critic,
    wrapup,
)
from interviewer.nodes.ask import Respondent
from interviewer.routing import CONTINUE, WRAPUP, route_after_assess
from interviewer.state import InterviewState, initial_state


def build_graph(client: LLMClient, respondent: Respondent):
    graph = StateGraph(InterviewState)

    graph.add_node("prepare", partial(prepare, client=client))
    graph.add_node("plan_question", partial(plan_question, client=client))
    graph.add_node("critic", partial(critic, client=client))
    graph.add_node("ask", partial(ask, respondent=respondent))
    graph.add_node("assess", partial(assess, client=client))
    graph.add_node("wrapup", partial(wrapup, respondent=respondent))

    graph.set_entry_point("prepare")
    graph.add_edge("prepare", "plan_question")
    graph.add_edge("plan_question", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"ask": "ask", "plan_question": "plan_question"},
    )
    graph.add_edge("ask", "assess")
    graph.add_conditional_edges(
        "assess",
        route_after_assess,
        {CONTINUE: "plan_question", WRAPUP: "wrapup"},
    )
    graph.add_edge("wrapup", END)

    return graph.compile()


def run_interview(
    research_goal: str,
    client: LLMClient,
    respondent: Respondent,
    recursion_limit: int = 200,
) -> InterviewState:
    compiled = build_graph(client, respondent)
    return compiled.invoke(
        initial_state(research_goal),
        config={"recursion_limit": recursion_limit},
    )
