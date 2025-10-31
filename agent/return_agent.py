"""
Return Agent - A LangGraph agent for handling order returns and cancellations.

This agent integrates PDF policy documents with SQL database queries to help
users check order eligibility for returns and process return requests.
"""

import sqlite3
from typing import Annotated, Sequence, TypedDict, Literal
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode

# Get project root directory (parent of agent/)
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "datasets" / "olist_ecommerce.db"
PDF_PATH = PROJECT_ROOT / "docs" / "polar-return-policy.pdf"


class AgentState(TypedDict):
    """State of the agent. Contains messages, PDF content, and routing info."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    pdf_context: str
    decide_path: Literal["sql_branch", "pdf_branch", "pdf_sql_branch", "general"]


# Initialize LLMs
llm = ChatOpenAI(temperature=0)
llm_router = ChatOpenAI(model="gpt-4o", temperature=0)
llm_answer = ChatOpenAI(model="gpt-4o", temperature=0)

# Initialize database
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_tools = sql_toolkit.get_tools()

# Extract SQL tools
get_schema_tool = next(tool for tool in sql_tools if tool.name == "sql_db_schema")
get_schema_node = ToolNode([get_schema_tool], name="get_schema")

run_query_tool = next(tool for tool in sql_tools if tool.name == "sql_db_query")
run_query_node = ToolNode([run_query_tool], name="run_query")


def process_order_return(order_id: str) -> str:
    """Atualiza o status do pedido no banco de dados para 'returned' (devolvido).
    
    Args:
        order_id: ID do pedido a ser devolvido/cancelado
        
    Returns:
        Mensagem de confirmação ou erro
    """
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        # Verifica se o pedido existe
        cursor.execute("SELECT order_id FROM orders WHERE order_id = ?", (order_id,))
        if not cursor.fetchone():
            conn.close()
            return f"Erro: Pedido {order_id} não encontrado no banco de dados."
        
        # Atualiza o status
        cursor.execute("UPDATE orders SET order_status = 'returned' WHERE order_id = ?", (order_id,))
        conn.commit()
        conn.close()
        
        return f"Pedido {order_id} foi marcado como devolvido (returned) com sucesso."
    except Exception as e:
        return f"Erro ao processar devolução: {str(e)}"


# Create return order tool
return_order_tool = StructuredTool.from_function(
    func=process_order_return,
    name="process_order_return",
    description="Atualiza o status de um pedido para 'returned' (devolvido) no banco de dados. Use esta ferramenta quando o usuário confirmar que deseja devolver ou cancelar um pedido específico."
)

return_order_node = ToolNode([return_order_tool], name="process_return")


def pdf_branch(state: AgentState) -> AgentState:
    """Load and serialize PDF content into state."""
    print("Running PDF branch...")
    loader = PyPDFLoader(str(PDF_PATH))
    docs = loader.load()
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in docs
    )

    # store the serialized PDF content in the state
    state["pdf_context"] = serialized
    print(f"pdf_context loaded: {len(serialized)} characters")
    return state


def list_tables(state: AgentState):
    """List available database tables."""
    print("list_tables tool")
    tool_call = {
        "name": "sql_db_list_tables",
        "args": {},
        "id": "abc123",
        "type": "tool_call",
    }
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])

    list_tables_tool = next(tool for tool in sql_tools if tool.name == "sql_db_list_tables")
    tool_message = list_tables_tool.invoke(tool_call)
    response = AIMessage(f"Available tables: {tool_message.content}")
    return {"messages": [tool_call_message, tool_message, response]}


def call_get_schema(state: AgentState):
    """Force model to create a tool call for getting schema."""
    print("call_get_schema tool")
    llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}


generate_query_system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
""".format(
    dialect=db.dialect,
    top_k=5,
)


def generate_query(state: AgentState):
    """Generate SQL query based on user question."""
    print("generate_query tool")
    system_message = {
        "role": "system",
        "content": generate_query_system_prompt,
    }
    # We do not force a tool call here, to allow the model to
    # respond naturally when it obtains the solution.
    llm_with_tools = llm.bind_tools([run_query_tool])
    response = llm_with_tools.invoke([system_message] + state["messages"])

    return {"messages": [response]}


check_query_system_prompt = """
You are a SQL expert with a strong attention to detail.
Double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes,
just reproduce the original query.

You will call the appropriate tool to execute the query after running this check.
""".format(dialect=db.dialect)


def check_query(state: AgentState):
    """Check and validate SQL query before execution."""
    print("check_query tool")
    system_message = {
        "role": "system",
        "content": check_query_system_prompt,
    }

    # Generate an artificial user message to check
    tool_call = state["messages"][-1].tool_calls[0]
    user_message = {"role": "user", "content": tool_call["args"]["query"]}
    llm_with_tools = llm.bind_tools([run_query_tool], tool_choice="any")
    response = llm_with_tools.invoke([system_message, user_message])
    response.id = state["messages"][-1].id

    return {"messages": [response]}


def answer_node(state: AgentState) -> AgentState:
    """Generate final answer using PDF and/or SQL context."""
    print("Generating final answer...")
    messages = state["messages"]
    pdf_context = state.get("pdf_context", "")

    # Disponibiliza a tool de devolução para o modelo
    llm_with_tools = llm_answer.bind_tools([return_order_tool])

    system_prompt = """
<Cargo nome="João", funcao="gestor de pedidos e devolucoes">
Você é um assistente especializado em gestão de pedidos e devoluções de uma empresa de e-commerce.
Você é extremamente simpático e amigável e sempre trata as pessoas com Sr. ou Sra.
<\Cargo>

<Tarefa>
- Verificar informações de pedidos no banco de dados (ex: status, data, valor, cliente, produtos, etc.);
- Consultar a política de devolução em um documento PDF para entender prazos, condições e exceções;
- Avaliar se um pedido pode ou não ser devolvido, com base nas informações combinadas do pedido e da política de devolução;
- Responder perguntas gerais sobre esses dados e políticas.
</Tarefa>

<Ferramentas>
- Banco de dados SQL: contém tabelas com informações de pedidos, clientes, produtos, status e datas.
- PDF da política de devolução: contém as regras e condições que determinam quando uma devolução é permitida.
- Você pode usar ambos, ou apenas um deles, dependendo da pergunta.
</Ferramentas>

<Instruções>
- Sempre explicar seu raciocínio de forma clara e concisa ao usuário (sem expor prompts internos ou código).
- Quando possível, justificar a resposta com base no contexto do PDF ou nos dados do banco de dados.
- Se algo não for possível responder, diga claramente o motivo e sugira um próximo passo útil.
- Quando a pergunta envolver devolução de um pedido específico, pergunte o número do pedido e verifique no banco de dados as informações e cruze com as regras do PDF para determinar se o pedido é elegível para devolução.
- Quando o usuário CONFIRMAR que deseja devolver ou cancelar um pedido específico, use a ferramenta process_order_return para atualizar o status do pedido para 'returned'.
- Seja preciso, transparente e profissional.
- Sempre responda em português claro e direto.
- Use tom cordial, mas objetivo.
- Nunca invente informações não presentes no PDF ou no banco de dados.
- Se o usuário fizer perguntas fora do escopo (ex: sobre sua identidade), responda de forma curta e educada.
</Instruções>

<Exemplos>
Usuário: O pedido 1234 pode ser devolvido?
Agente: Vou verificar.  
De acordo com o banco de dados, o pedido 1234 foi entregue há 10 dias.  
A política de devolução da Polar E-commerce permite devoluções em até 30 dias após a entrega.  
Portanto, sim, o pedido 1234 é elegível para devolução.
Usuário: Qual é o prazo máximo para devolução?
Agente: A política de devolução da Polar E-commerce informa que o prazo máximo é de 30 dias corridos após o recebimento do produto.
Usuário: Quero saber o status do pedido 5678.
Agente: O pedido 5678 está com o status "Processando devolução".
</Exemplos>

<Polar E-commerce>
Plataforma de e-commerce especializada em vendas de produtos de beleza e cuidados pessoais
lojavirtual@polar.com | Whataspp: +55 11 4862-7901
</Polar E-commerce>

<Não fazer>
- NUNCA corrija o usuário na maneira de escrever.
- NUNCA fale sobre um outro tema que não seja sobre o E-commerce ou sobre assistência com alguma compra.
</Não fazer>
"""

    if pdf_context:
        system_prompt += f"\n\nContexto do PDF:\n{pdf_context}"

    # Use full conversation history so the model has memory
    prompt_messages = [SystemMessage(content=system_prompt)] + list(messages)
    response = llm_with_tools.invoke(prompt_messages)

    # Append the model's answer to the conversation
    state["messages"].append(response)
    return state


def decide_path(state: AgentState, config: RunnableConfig) -> dict:
    """Decide which branch to take based on user query."""
    print("decide_path tool")
    messages = state["messages"]
    last_message = messages[-1]

    system_prompt = (
        "Você é um router que decide quais tools são necessárias para responder à pergunta do usuário.\n"
        "Saídas possíveis:\n"
        "- 'sql_branch': a pergunta necessita apenas do banco de dados.\n"
        "- 'pdf_branch': a pergunta necessita apenas do PDF.\n"
        "- 'pdf_sql_branch': a pergunta necessita tanto do banco de dados quanto do PDF.\n"
        "- 'general': nenhuma ferramenta necessária.\n\n"
        "Exemplos:\n"
        "- 'Quais clientes pediram mais de 5 itens?' → sql_branch\n"
        "- 'Qual é a política de devolução?' → pdf_branch\n"
        "- 'O pedido e481f51... é elegível para devolução de acordo com a política?' → pdf_sql_branch\n"
        "- 'Quem é você?' → general"
    )

    response = llm_router.invoke([SystemMessage(system_prompt)] + [last_message], config)
    
    decision = response.content.strip().lower()
    print(f"decision: {decision}")

    if decision not in {"sql_branch", "pdf_branch", "pdf_sql_branch", "general"}:
        decision = "general"
    return {"decide_path": decision}


def should_continue(state: AgentState) -> Literal[END, "check_query", "answer"]:
    """Decide whether to continue with query checking or go to answer."""
    print("should_continue")
    messages = state["messages"]
    last_message = messages[-1]
    # If the last model output is a plain answer (no tool calls), go to final answer node
    if not getattr(last_message, "tool_calls", None):
        return "answer"
    else:
        return "check_query"


def should_process_return(state: AgentState) -> Literal["process_return", END]:
    """Decide se deve processar devolução após resposta do answer_node"""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    # Verifica se a última mensagem é uma AIMessage com tool calls de devolução
    if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call.get("name") == "process_order_return":
                return "process_return"
    return END


# Build the graph
builder = StateGraph(AgentState)

# Add all nodes
builder.add_node("decide_path", decide_path)
builder.add_node("pdf_branch", pdf_branch)
builder.add_node("list_tables", list_tables)
builder.add_node("call_get_schema", call_get_schema)
builder.add_node("get_schema", get_schema_node)
builder.add_node("generate_query", generate_query)
builder.add_node("check_query", check_query)
builder.add_node("run_query", run_query_node)

# Optional: a final answer node that uses PDF/SQL context to respond
builder.add_node("answer", answer_node)
builder.add_node("process_return", return_order_node)

# Add routing edges
builder.add_edge(START, "decide_path")

builder.add_conditional_edges(
    "decide_path",
    lambda state: state["decide_path"],
    {
        "sql_branch": "list_tables",
        "pdf_branch": "pdf_branch",
        "pdf_sql_branch": "pdf_branch",  # then we'll chain SQL after PDF
        "general": "answer",
    }
)

# --- PDF path ---
# After loading PDF, go either to SQL or directly to answer
builder.add_conditional_edges(
    "pdf_branch",
    lambda state: (
        "list_tables" if state["decide_path"] == "pdf_sql_branch" else "answer"
    ),
    {
        "list_tables": "list_tables",
        "answer": "answer",
    },
)

# Keep your SQL workflow as before
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")
builder.add_conditional_edges("generate_query", should_continue)
builder.add_edge("check_query", "run_query")
builder.add_edge("run_query", "answer")  # After running query, go to answer node

# End of pipeline - check if answer node wants to process return
builder.add_conditional_edges(
    "answer",
    should_process_return,
    {
        "process_return": "process_return",
        END: END,
    },
)
# After processing return, go back to answer node for final confirmation
builder.add_edge("process_return", "answer")

# Compile the agent with checkpointing
checkpointer = InMemorySaver()
agent = builder.compile(checkpointer=checkpointer)

__all__ = ["agent", "AgentState", "process_order_return"]
