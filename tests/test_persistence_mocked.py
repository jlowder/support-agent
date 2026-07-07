
import unittest
from unittest.mock import MagicMock, patch
import json
import uuid
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from backend.app.main import build_agent_graph, AgentState
from langgraph.checkpoint.memory import MemorySaver

class TestPersistence(unittest.TestCase):

    def test_session_memory_with_mock_llm(self):
        # Create a mock LLM that doesn't need a real connection
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm

        # Turn 1 response
        mock_llm.invoke.side_effect = [
            AIMessage(content="Hello Lauren, how can I help you?"),
            AIMessage(content="I see your order ORD-000006. What's wrong with it?")
        ]

        # Re-build graph with mock LLM and its own memory
        memory = MemorySaver()

        from backend.app.main import _ensure_base_messages, _get_system_prompt, node_generate_response, tools_node
        from langgraph.graph import StateGraph, END

        def agent_node(state, config):
            messages = _ensure_base_messages(state.get("messages", []))
            system_prompt = _get_system_prompt()
            langchain_messages = [SystemMessage(content=system_prompt)] + messages
            response = mock_llm.invoke(langchain_messages, config=config)
            return {"messages": [response]}

        def route_after_agent(state):
            messages = state.get("messages", [])
            if messages and getattr(messages[-1], 'tool_calls', None):
                return "tools"
            return "generate_response"

        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tools_node)
        graph.add_node("generate_response", node_generate_response)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "generate_response": "generate_response"})
        graph.add_edge("tools", "agent")
        graph.add_edge("generate_response", END)

        compiled_graph = graph.compile(checkpointer=memory)

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # Turn 1
        input_1 = {"messages": [HumanMessage(content="I am lauren.lewis@server.com")]}
        result_1 = compiled_graph.invoke(input_1, config=config)

        # Turn 2
        input_2 = {"messages": [HumanMessage(content="I want a refund")]}
        result_2 = compiled_graph.invoke(input_2, config=config)

        # Verify state in checkpointer
        state = compiled_graph.get_state(config)
        messages = state.values["messages"]

        # We expect: Turn 1 Human, Turn 1 AI, Turn 2 Human, Turn 2 AI
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0].content, "I am lauren.lewis@server.com")
        self.assertEqual(messages[1].content, "Hello Lauren, how can I help you?")
        self.assertEqual(messages[2].content, "I want a refund")
        self.assertEqual(messages[3].content, "I see your order ORD-000006. What's wrong with it?")
        print("\nPersistence test passed!")

if __name__ == "__main__":
    unittest.main()
