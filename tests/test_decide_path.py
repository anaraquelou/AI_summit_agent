"""
Unit tests for decide_path using GenericFakeChatModel.
"""

import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.runnables import RunnableConfig

from agent.return_agent import decide_path


@pytest.fixture
def config():
    return RunnableConfig()


def make_fake_llm(output: str):
    """
    Create a GenericFakeChatModel that always returns the given output,
    regardless of the input.
    """
    return GenericFakeChatModel(
        messages=iter([AIMessage(content=output)])
    )


class TestDecidePath:

    def test_sql_branch(self, config):
        fake_llm = make_fake_llm("sql_branch")

        state = {
            "messages": [
                HumanMessage(content="Há quantos pedidos com status cancelado?")
            ]
        }

        result = decide_path(state, config, llm_override=fake_llm)
        assert result["decide_path"] == "sql_branch"

    def test_pdf_branch(self, config):
        fake_llm = make_fake_llm("pdf_branch")

        state = {
            "messages": [
                HumanMessage(content="Qual é a política de devolução?")
            ]
        }

        result = decide_path(state, config, llm_override=fake_llm)
        assert result["decide_path"] == "pdf_branch"

    def test_pdf_sql_branch(self, config):
        fake_llm = make_fake_llm("pdf_sql_branch")

        state = {
            "messages": [
                HumanMessage(content="O pedido e481f51... é elegível para devolução?")
            ]
        }

        result = decide_path(state, config, llm_override=fake_llm)
        assert result["decide_path"] == "pdf_sql_branch"

    def test_process_return(self, config):
        fake_llm = make_fake_llm("process_return")

        state = {
            "messages": [
                HumanMessage(content="Devolver o pedido e481f51cbdc54678b7cc49136f2d6af7")
            ]
        }

        result = decide_path(state, config, llm_override=fake_llm)
        assert result["decide_path"] == "process_return"

    def test_analyze_seller_reliability(self, config):
        fake_llm = make_fake_llm("analyze_seller_reliability")

        state = {
            "messages": [
                HumanMessage(content="O seller 3442f8959a84dea7ee197c632cb2df15 é confiável?")
            ]
        }

        result = decide_path(state, config, llm_override=fake_llm)
        assert result["decide_path"] == "analyze_seller_reliability"

    def test_general(self, config):
        fake_llm = make_fake_llm("general")

        state = {
            "messages": [
                HumanMessage(content="Quem é você?")
            ]
        }

        result = decide_path(state, config, llm_override=fake_llm)
        assert result["decide_path"] == "general"

    def test_invalid_falls_back_to_general(self, config):
        fake_llm = make_fake_llm("BANANA")  # invalid routing

        state = {
            "messages": [
                HumanMessage(content="Some query")
            ]
        }

        result = decide_path(state, config, llm_override=fake_llm)
        assert result["decide_path"] == "general"

    def test_lowercase_normalization(self, config):
        fake_llm = make_fake_llm("SQL_BRANCH")

        state = {
            "messages": [
                HumanMessage(content="Query about orders")
            ]
        }

        result = decide_path(state, config, llm_override=fake_llm)
        assert result["decide_path"] == "sql_branch"

    def test_strip_whitespace(self, config):
        fake_llm = make_fake_llm("   sql_branch   ")

        state = {
            "messages": [
                HumanMessage(content="Query about orders")
            ]
        }

        result = decide_path(state, config, llm_override=fake_llm)
        assert result["decide_path"] == "sql_branch"

    def test_uses_last_message(self, config):
        fake_llm = make_fake_llm("pdf_branch")

        state = {
            "messages": [
                HumanMessage(content="Mensagem inicial"),
                HumanMessage(content="Qual é a política de devolução?")
            ]
        }

        result = decide_path(state, config, llm_override=fake_llm)
        assert result["decide_path"] == "pdf_branch"
