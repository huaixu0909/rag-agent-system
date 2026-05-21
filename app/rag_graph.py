from typing import Any, Callable, TypedDict


NO_ENOUGH_CONTEXT_ANSWER = "当前知识库中没有足够信息回答这个问题。"


class RagGraphState(TypedDict, total=False):
    question: str
    top_k: int
    score_threshold: float
    search_response: Any
    sources: list[Any]
    answer: str
    mode: str
    retrieval_mode: str
    graph_path: list[str]


def run_langgraph_rag_chat(
    *,
    question: str,
    top_k: int,
    score_threshold: float,
    search_fn: Callable[[str, int, float], Any],
    build_sources_fn: Callable[[list[Any]], list[Any]],
    langchain_answer_fn: Callable[[str, list[dict[str, Any]]], str | None],
    deepseek_answer_fn: Callable[[str, list[Any]], str | None],
    template_answer_fn: Callable[[str, list[Any]], str],
) -> dict[str, Any] | None:
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        return None

    def retrieve(state: RagGraphState) -> RagGraphState:
        search_response = search_fn(
            state["question"],
            state["top_k"],
            state["score_threshold"],
        )
        return {
            "search_response": search_response,
            "retrieval_mode": search_response.mode,
            "graph_path": ["retrieve"],
        }

    def route_after_retrieve(state: RagGraphState) -> str:
        search_response = state["search_response"]
        if search_response.results:
            return "generate_answer"
        return "reject_answer"

    def reject_answer(state: RagGraphState) -> RagGraphState:
        return {
            "answer": NO_ENOUGH_CONTEXT_ANSWER,
            "sources": [],
            "mode": "retrieval_template",
            "graph_path": [*state.get("graph_path", []), "reject_answer"],
        }

    def generate_answer(state: RagGraphState) -> RagGraphState:
        search_response = state["search_response"]
        results = search_response.results
        sources = build_sources_fn(results)
        chunks = [
            item.model_dump() if hasattr(item, "model_dump") else dict(item)
            for item in results
        ]

        answer = langchain_answer_fn(state["question"], chunks)
        if answer:
            return {
                "answer": answer,
                "sources": sources,
                "mode": "langgraph_deepseek",
                "graph_path": [*state.get("graph_path", []), "generate_answer"],
            }

        answer = deepseek_answer_fn(state["question"], results)
        if answer:
            return {
                "answer": answer,
                "sources": sources,
                "mode": "deepseek",
                "graph_path": [*state.get("graph_path", []), "generate_answer"],
            }

        return {
            "answer": template_answer_fn(state["question"], results),
            "sources": sources,
            "mode": "retrieval_template",
            "graph_path": [*state.get("graph_path", []), "generate_answer"],
        }

    try:
        graph = StateGraph(RagGraphState)
        graph.add_node("retrieve", retrieve)
        graph.add_node("reject_answer", reject_answer)
        graph.add_node("generate_answer", generate_answer)
        graph.add_edge(START, "retrieve")
        graph.add_conditional_edges(
            "retrieve",
            route_after_retrieve,
            {
                "reject_answer": "reject_answer",
                "generate_answer": "generate_answer",
            },
        )
        graph.add_edge("reject_answer", END)
        graph.add_edge("generate_answer", END)

        compiled_graph = graph.compile()
        return compiled_graph.invoke(
            {
                "question": question,
                "top_k": top_k,
                "score_threshold": score_threshold,
            }
        )
    except Exception:
        return None
