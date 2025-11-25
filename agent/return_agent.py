"""
Data Analyst Agent - A LangGraph agent for handling data analysis.

This agent integrates PDF policy documents with SQL database queries to help
users analyze data.
"""
import sqlite3
import re
from typing import Annotated, Sequence, TypedDict, Literal
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, ToolMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.runtime import get_runtime

# Load environment variables
load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "datasets" / "olist_ecommerce.db"


@dataclass
class AgentContext:
    user_id: str
    db_connection: sqlite3.Connection
    pdf_policy: str


class AgentState(TypedDict):
    """State of the agent. Contains messages, PDF content, and routing info."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    pdf_context: str
    decide_path: Literal["sql_branch", "pdf_branch", "pdf_sql_branch", "process_return", "analyze_seller_reliability", "general"]


# Initialize LLMs
llm = ChatOpenAI(model="gpt-5", temperature=0)
llm_answer = ChatOpenAI(model="gpt-5", temperature=0, reasoning_effort="low")

# Initialize database and SQL tools (this is fine - SQLAlchemy handles pooling)
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_tools = sql_toolkit.get_tools()

# Extract SQL tools
run_query_tool = next(tool for tool in sql_tools if tool.name == "sql_db_query")
run_query_node = ToolNode([run_query_tool], name="run_query")


def process_order_return(order_id: str) -> str:
    """Atualiza o status do pedido no banco de dados para 'return_processed' (devolvido).
    Args:
        order_id: ID do pedido a ser devolvido/cancelado
    Returns:
        Mensagem de confirmação ou erro
    """
    try:
        # Access DB without reopening connection
        runtime = get_runtime(AgentContext)
        cursor = runtime.context.db_connection.cursor()
            
        # Verifica se o pedido existe
        cursor.execute("SELECT order_id FROM orders WHERE order_id = ?", (order_id,))
        runtime.context.db_connection.commit()
        if not cursor.fetchone():
            return f"Erro: Pedido {order_id} não encontrado no banco de dados."
        
        # Atualiza o status
        cursor.execute("UPDATE orders SET order_status = 'return_processed' WHERE order_id = ?", (order_id,))
        
        return f"Pedido {order_id} foi marcado como devolvido (return_processed) com sucesso."
    except Exception as e:
        return f"Erro ao processar devolução: {str(e)}"


# Create return order tool
return_order_tool = StructuredTool.from_function(
    func=process_order_return,
    name="process_order_return",
    description="Atualiza o status de um pedido para 'return_processed' (devolvido) no banco de dados. Use esta ferramenta quando o usuário confirmar que deseja devolver ou cancelar um pedido específico."
)

return_order_node = ToolNode([return_order_tool], name="process_return")


def process_return_node(state: AgentState) -> AgentState:
    """Process return order - extract order_id and call tool directly."""
    print("process_return_node: calling tool directly")
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    if not isinstance(last_message, HumanMessage):
        # If already has tool call, use ToolNode
        result = return_order_node.invoke(state)
        # Return final answer instead of going to answer node
        tool_result = result["messages"][-1] if result.get("messages") else None
        if isinstance(tool_result, ToolMessage):
            final_response = AIMessage(content=tool_result.content)
            return {"messages": [final_response]}
        return result
    
    # Extract order_id directly from user message
    user_text = str(last_message.content)
    
    # Try to find order_id pattern (alphanumeric string, typically 32 chars)
    order_id_pattern = r'\b[a-f0-9]{20,}\b'
    matches = re.findall(order_id_pattern, user_text, re.IGNORECASE)
    
    if matches:
        order_id = matches[0]
    else:
        # Try to find any word that looks like an order ID
        words = user_text.split()
        # Look for words that are longer than 10 chars (likely order IDs)
        order_id = next((w for w in words if len(w) > 10), None)
        if not order_id:
            # Fallback: use first significant word or return error
            order_id = words[0] if words else ""
    
    # Call tool directly
    result = process_order_return(order_id)
    
    # Return final answer
    final_response = AIMessage(content=result)
    return {"messages": [final_response]}


def analyze_seller_reliability_node_custom(state: AgentState) -> AgentState:
    """Analyze seller reliability - extract parameters and call tool directly."""
    print("analyze_seller_reliability_node_custom: calling tool directly")
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    if not isinstance(last_message, HumanMessage):
        # If already has tool call, use ToolNode
        result = seller_reliability_node.invoke(state)
        # Return final answer instead of going to answer node
        tool_result = result["messages"][-1] if result.get("messages") else None
        if isinstance(tool_result, ToolMessage):
            final_response = AIMessage(content=tool_result.content)
            return {"messages": [final_response]}
        return result
    
    # Extract parameters directly from user message
    user_text = str(last_message.content)
    
    # Try to find seller_id (alphanumeric string, typically 32 chars)
    seller_id_pattern = r'\b[a-f0-9]{20,}\b'
    seller_matches = re.findall(seller_id_pattern, user_text, re.IGNORECASE)
    seller_id = seller_matches[0] if seller_matches else None
    
    # Try to find "top N" pattern (e.g., "top 3", "top 5", "top 10")
    top_n_pattern = r'\btop\s+(\d+)\b'
    top_n_match = re.search(top_n_pattern, user_text, re.IGNORECASE)
    limit = int(top_n_match.group(1)) if top_n_match else None
    
    # Try to find dates in format YYYY-MM-DD
    date_pattern = r'\b(\d{4}-\d{2}-\d{2})\b'
    dates = re.findall(date_pattern, user_text)
    
    start_date = None
    end_date = None
    
    if len(dates) >= 1:
        start_date = dates[0]
    if len(dates) >= 2:
        end_date = dates[1]
    elif len(dates) == 1:
        # If only one date found, check context to determine if it's start or end
        # For simplicity, treat as start_date
        start_date = dates[0]
    
    # Call tool directly
    result = analyze_seller_reliability(seller_id=seller_id, start_date=start_date, end_date=end_date, limit=limit)
    
    # Create a tool_call_id for the seller reliability call
    tool_call_id = f"seller_reliability_call_{len(state['messages'])}"
    
    # Build args for tool call
    tool_args = {}
    if seller_id:
        tool_args["seller_id"] = seller_id
    if start_date:
        tool_args["start_date"] = start_date
    if end_date:
        tool_args["end_date"] = end_date
    if limit:
        tool_args["limit"] = limit
    
    # Create AIMessage with tool_calls to satisfy OpenAI API requirement
    # ToolMessage must be preceded by AIMessage with tool_calls
    ai_message = AIMessage(
        content="",
        tool_calls=[{
            "id": tool_call_id,
            "name": "analyze_seller_reliability",
            "args": tool_args,
            "type": "tool_call"
        }]
    )
    
    # Return result as ToolMessage to maintain consistency with tool call pattern
    tool_message = ToolMessage(
        content=str(result),
        tool_call_id=tool_call_id,
        name="analyze_seller_reliability"
    )
    
    # Return both messages: AIMessage first, then ToolMessage
    return {"messages": [ai_message, tool_message]}


def analyze_seller_reliability(seller_id: str = None, start_date: str = None, end_date: str = None, limit: int = None) -> str:
    """Analyzes seller reliability based on late delivery rate and average review score.
    
    A seller is considered unreliable if they have more than 5% of orders delivered late
    AND an average review score below 3.5 within the specified date range.
    If dates are not provided, analyzes all available data.
    If seller_id is provided, analyzes only that specific seller.
    If limit is provided, returns only the top N sellers (ordered by worst performance).
    
    Args:
        seller_id: Optional seller ID to analyze. If provided, returns yes/no answer for that seller.
        start_date: Optional start date in format 'YYYY-MM-DD'. If None, uses all available data.
        end_date: Optional end date in format 'YYYY-MM-DD'. If None, uses all available data.
        limit: Optional limit for number of results (e.g., 3 for "top 3").
        
    Returns:
        Formatted string with analysis results. If seller_id is provided, returns yes/no answer.
        Otherwise, returns list of unreliable sellers (limited if limit is provided).
    """
    try:
        # Access DB without reopening connection
        runtime = get_runtime(AgentContext)
        cursor = runtime.context.db_connection.cursor()
        
        # Build WHERE clause based on provided parameters
        where_conditions = ["o.order_status = 'delivered'"]
        query_params = []
        
        if seller_id:
            where_conditions.append("oi.seller_id = ?")
            query_params.append(seller_id)
        
        if start_date:
            where_conditions.append("date(o.order_purchase_timestamp) >= date(?)")
            query_params.append(start_date)
        
        if end_date:
            where_conditions.append("date(o.order_purchase_timestamp) <= date(?)")
            query_params.append(end_date)
        
        where_clause = " AND ".join(where_conditions)
        
        # Query to calculate seller reliability metrics
        query = f"""
        WITH seller_metrics AS (
            SELECT 
                oi.seller_id,
                COUNT(DISTINCT o.order_id) as total_orders,
                SUM(CASE 
                    WHEN o.order_delivered_customer_date IS NOT NULL 
                    AND o.order_estimated_delivery_date IS NOT NULL
                    AND date(o.order_delivered_customer_date) > date(o.order_estimated_delivery_date)
                    THEN 1 
                    ELSE 0 
                END) as late_orders,
                AVG(or_review.review_score) as avg_review_score
            FROM order_items oi
            INNER JOIN orders o ON oi.order_id = o.order_id
            LEFT JOIN order_reviews or_review ON o.order_id = or_review.order_id
            WHERE {where_clause}
            GROUP BY oi.seller_id
        )
        SELECT 
            sm.seller_id,
            s.seller_city,
            s.seller_state,
            sm.total_orders,
            sm.late_orders,
            ROUND(CAST(sm.late_orders AS FLOAT) / sm.total_orders * 100, 2) as late_percentage,
            ROUND(COALESCE(sm.avg_review_score, 0), 2) as avg_review_score
        FROM seller_metrics sm
        INNER JOIN sellers s ON sm.seller_id = s.seller_id
        WHERE (CAST(sm.late_orders AS FLOAT) / sm.total_orders * 100) > 5.0
        AND COALESCE(sm.avg_review_score, 0) < 3.5
        ORDER BY late_percentage DESC, avg_review_score ASC
        """
        
        # Add LIMIT if specified
        if limit is not None:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, query_params)
        results = cursor.fetchall()
        
        # If seller_id was provided, return yes/no answer with metrics
        if seller_id:
            # First, check if seller exists
            cursor.execute("SELECT seller_id FROM sellers WHERE seller_id = ?", (seller_id,))
            seller_exists = cursor.fetchone()
            
            if not seller_exists:
                return f"Erro: Seller {seller_id} não encontrado no banco de dados."
            
            # Calculate seller metrics regardless of reliability status
            metrics_query = f"""
            WITH seller_metrics AS (
                SELECT 
                    oi.seller_id,
                    COUNT(DISTINCT o.order_id) as total_orders,
                    SUM(CASE 
                        WHEN o.order_delivered_customer_date IS NOT NULL 
                        AND o.order_estimated_delivery_date IS NOT NULL
                        AND date(o.order_delivered_customer_date) > date(o.order_estimated_delivery_date)
                        THEN 1 
                        ELSE 0 
                    END) as late_orders,
                    AVG(or_review.review_score) as avg_review_score
                FROM order_items oi
                INNER JOIN orders o ON oi.order_id = o.order_id
                LEFT JOIN order_reviews or_review ON o.order_id = or_review.order_id
                WHERE {where_clause}
                GROUP BY oi.seller_id
            )
            SELECT 
                sm.seller_id,
                s.seller_city,
                s.seller_state,
                sm.total_orders,
                sm.late_orders,
                ROUND(CAST(sm.late_orders AS FLOAT) / sm.total_orders * 100, 2) as late_percentage,
                ROUND(COALESCE(sm.avg_review_score, 0), 2) as avg_review_score
            FROM seller_metrics sm
            INNER JOIN sellers s ON sm.seller_id = s.seller_id
            """
            
            cursor.execute(metrics_query, query_params)
            metrics_result = cursor.fetchone()
            
            if not metrics_result:
                return f"Erro: Não foi possível calcular métricas para o seller {seller_id}."
            
            seller_id_result, city, state, total_orders, late_orders, late_pct, avg_score = metrics_result
            
            # Determine if seller is reliable
            is_unreliable = late_pct > 5.0 and avg_score < 3.5
            
            if is_unreliable:
                return (
                    f"Sim, o seller {seller_id} é considerado não confiável.\n"
                    f"Motivos:\n"
                    f"- Taxa de pedidos atrasados: {late_pct}% (acima de 5%)\n"
                    f"- Nota média de reviews: {avg_score}/5.0 (abaixo de 3.5)\n"
                    f"- Total de pedidos analisados: {total_orders}\n"
                    f"- Pedidos atrasados: {late_orders}"
                )
            else:
                return (
                    f"Não, o seller {seller_id} é considerado confiável.\n"
                    f"Motivos:\n"
                    f"- Taxa de pedidos atrasados: {late_pct}% (abaixo ou igual a 5%)\n"
                    f"- Nota média de reviews: {avg_score}/5.0 (acima ou igual a 3.5)\n"
                    f"- Total de pedidos analisados: {total_orders}\n"
                    f"- Pedidos atrasados: {late_orders}"
                )
        
        
        # Format date range text
        date_range_text = ""
        if start_date and end_date:
            date_range_text = f" no período de {start_date} a {end_date}"
        elif start_date:
            date_range_text = f" a partir de {start_date}"
        elif end_date:
            date_range_text = f" até {end_date}"
        else:
            date_range_text = " (todos os dados disponíveis)"
        
        if not results:
            return f"Nenhum vendedor não confiável encontrado{date_range_text}."
        
        # Format results
        limit_text = f" (top {limit})" if limit else ""
        result_lines = [
            f"Vendedores não confiáveis{limit_text}{date_range_text}:",
            "",
            "=" * 80
        ]
        
        for row in results:
            seller_id_result, city, state, total_orders, late_orders, late_pct, avg_score = row
            result_lines.append(
                f"Vendedor: {seller_id_result} | {city}, {state}\n"
                f"  - Total de pedidos: {total_orders}\n"
                f"  - Pedidos atrasados: {late_orders} ({late_pct}%)\n"
                f"  - Nota média de reviews: {avg_score}/5.0"
            )
            result_lines.append("-" * 80)
        
        if limit:
            result_lines.append(
                f"\nMostrando top {limit} vendedores não confiáveis."
            )
        else:
            result_lines.append(
            f"\nTotal de vendedores não confiáveis: {len(results)}"
        )
        
        return "\n".join(result_lines)
        
    except Exception as e:
        return f"Erro ao analisar confiabilidade dos vendedores: {str(e)}"


# Create seller reliability analysis tool
seller_reliability_tool = StructuredTool.from_function(
    func=analyze_seller_reliability,
    name="analyze_seller_reliability",
    description="Analisa a confiabilidade dos vendedores com base em pedidos atrasados e avaliações. Um vendedor é considerado não confiável se tiver mais de 5% dos pedidos atrasados E nota média de review abaixo de 3.5. Use esta ferramenta quando o usuário perguntar sobre vendedores com desempenho ruim, violação de regras internas, ou análise de confiabilidade de vendedores. Parâmetros: seller_id (opcional, ID do vendedor específico), start_date (opcional, formato 'YYYY-MM-DD'), end_date (opcional, formato 'YYYY-MM-DD') e limit (opcional, número para limitar resultados, ex: 3 para 'top 3'). Se seller_id for fornecido, retorna resposta sim/não para aquele vendedor. Se limit for fornecido, retorna apenas os top N vendedores não confiáveis. Se as datas não forem fornecidas, a análise será feita com todos os dados disponíveis."
)

seller_reliability_node = ToolNode([seller_reliability_tool], name="analyze_seller_reliability")


def pdf_branch(state: AgentState) -> AgentState:
    """Load and serialize PDF content into state."""
    print("Running PDF branch...")
    runtime = get_runtime(AgentContext)
    serialized = runtime.context.pdf_policy

    # store the serialized PDF content in the state
    state["pdf_context"] = serialized
    print(f"pdf_context loaded: {len(serialized)} characters")
    return state


def call_get_schema(state: AgentState):
    """Get schema directly without creating a tool call."""
    print("call_get_schema: getting schema directly")
    # Call the schema tool directly instead of creating a tool call
    table_names_str = "category_translation, customers, geolocation, order_items, order_payments, order_reviews, orders, products, sellers"
    get_schema_tool = next(tool for tool in sql_tools if tool.name == "sql_db_schema")
    schema_result = get_schema_tool.invoke({"table_names": table_names_str})
    
    # Create a tool_call_id for the schema call
    tool_call_id = f"schema_call_{len(state['messages'])}"
    
    # Create AIMessage with tool_calls to satisfy OpenAI API requirement
    # ToolMessage must be preceded by AIMessage with tool_calls
    ai_message = AIMessage(
        content="",
        tool_calls=[{
            "id": tool_call_id,
            "name": "sql_db_schema",
            "args": {"table_names": table_names_str},
            "type": "tool_call"
        }]
    )
    
    # Return schema as ToolMessage to maintain consistency with tool call pattern
    tool_message = ToolMessage(
        content=str(schema_result),
        tool_call_id=tool_call_id,
        name="sql_db_schema"
    )
    
    # Return both messages: AIMessage first, then ToolMessage
    return {"messages": [ai_message, tool_message]}

generate_query_system_prompt = f"""
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {db.dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most 5 results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

Use the database schema information provided in the conversation messages to understand the table structure.
"""


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


def reduce_messages(messages, keep_last_user=1, keep_last_ai=1):
    """
    Safe reducer that keeps the last user + AI messages
    AND preserves valid tool_call → tool_response ordering.
    """

    # Reverse iterate
    reversed_msgs = list(reversed(messages))

    kept = []
    user_count = 0
    ai_count = 0

    # Collect minimal relevant messages
    for msg in reversed_msgs:
        if isinstance(msg, HumanMessage) and user_count < keep_last_user:
            kept.append(msg)
            user_count += 1
        elif isinstance(msg, AIMessage) and ai_count < keep_last_ai:
            kept.append(msg)
            ai_count += 1

    # Determine which tool_call_ids must be preserved
    required_tool_ids = set()
    for msg in kept:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                required_tool_ids.add(tc["id"])

    # Collect matching ToolMessages
    tool_msgs = [
        msg for msg in messages
        if isinstance(msg, ToolMessage) and msg.tool_call_id in required_tool_ids
    ]

    # Build new message list in chronological order
    new_messages = []

    for msg in messages:
        # Add AI
        if msg in kept:
            new_messages.append(msg)

            # Immediately attach its tool responses
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    for tm in tool_msgs:
                        if tm.tool_call_id == tc["id"]:
                            new_messages.append(tm)

        # Add User only if in kept
        elif isinstance(msg, HumanMessage) and msg in kept:
            new_messages.append(msg)

    return new_messages


def answer_node(state: AgentState) -> AgentState:
    """Generate final answer using PDF and/or SQL context."""
    print("Generating final answer...")
    messages = reduce_messages(state["messages"])
    pdf_context = state.get("pdf_context", "")


    system_prompt = """
<Cargo nome="João" funcao="analista sênior de dados de E-commerce">
Seu tom é direto, profissional e objetivo, como um colega analista experiente.
</Cargo>

<Tarefa>
- Consultar informações de pedidos no banco de dados.
- Ler a política de devolução em PDF.
- Responder perguntas sobre os dados do banco de dados e da política de devolução.
</Tarefa>

<Instruções>
- Seja sucinto e objetivo. Responda como um analista falando com outro analista.
- Apresente apenas a lógica essencial utilizada na conclusão, sem narrar ações internas 
  (ex: "consultando", "processando", "buscando").
- Processe a devolução direto sem perguntar por mais informações.
- Evite devolver perguntas desnecessárias; tente sempre avançar com a análise.
- Nunca invente informações além do que está no banco ou no PDF.
- Quando receber resultados de análise de vendedores (especialmente listas com separadores como === ou ---), 
  reformate completamente para uma resposta natural e legível:
</Instruções>

<Exemplos>
Usuário: O pedido 1234 pode ser devolvido?
Agente: O pedido 1234 foi entregue há 10 dias. A política da BIX permite devoluções em até 30 dias após a entrega. Ele está elegível.

Usuário: Qual é o prazo máximo para devolução?
Agente: O prazo máximo é de 30 dias corridos após o recebimento.

Usuário: Status do pedido 5678.
Agente: O pedido 5678 está com status "Devolução solicitada".

Usuário: Quais os top 3 vendedores menos confiáveis?
Agente: Os top 3 vendedores menos confiáveis são:
1. 8d92f3ea807b89465643c219455e7369 (São Paulo, SP): 233% de pedidos atrasados, nota média 1.0/5.0
2. 4e42581f08e8cfc7c090f930bac4552a (Porto Ferreira, SP): 200% de pedidos atrasados, nota média 1.0/5.0
3. 2a50b7ee5aebecc6fd0ff9784a4747d6 (Brasília, DF): 200% de pedidos atrasados, nota média 1.0/5.0
</Exemplos>

<Não fazer>
- Não sugerir ações ou ferramentas que o agente não possui.
- Não inventar dados ou regras fora do banco ou do PDF.
- Não usar linguagem emocional ou frases de atendimento ao cliente.
</Não fazer>
"""

    if pdf_context:
        system_prompt += f"\n\nContexto do PDF:\n{pdf_context}"

    # Use full conversation history so the model has memory
    prompt_messages = [SystemMessage(content=system_prompt)] + list(messages)
    response = llm_answer.invoke(prompt_messages)

    # Return message to be added by LangGraph's add_messages
    return {"messages": [response]}


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
        "- 'process_return': a pergunta claramente pede para processar/devolver um pedido específico (ex: 'devolver pedido X', 'processar devolução do pedido Y').\n"
        "- 'analyze_seller_reliability': a pergunta claramente pede análise de confiabilidade de vendedor(es) (ex: 'o seller X é confiável?', 'quais sellers são não confiáveis?').\n"
        "- 'general': nenhuma ferramenta necessária.\n\n"
        "Exemplos:\n"
        "- 'Há quantos pedidos com status devolução solicitada?' → sql_branch\n"
        "- 'Quais 3 vendedores tiveram o maior número de entregas atrasadas em 2024?' → sql_branch\n"
        "- 'Qual é a política de devolução?' → pdf_branch\n"
        "- 'O que acontece se o cliente pedir devolução 31 dias depois de receber o produto danificado?' → pdf_branch\n"
        "- 'O pedido e481f51... é elegível para devolução de acordo com a política?' → pdf_sql_branch\n"
        "- 'Quantos pedidos da base de dados são elegíveis a devolução' → pdf_sql_branch\n"
        "- 'Devolver o pedido e481f51...' → process_return\n"
        "- 'Processar devolução do pedido 12345' → process_return\n"
        "- 'O seller 3442f8959a84dea7ee197c632cb2df15 é confiável?' → analyze_seller_reliability\n"
        "- 'Quais sellers são não confiáveis?' → analyze_seller_reliability\n"
        "- 'Quem é você?' → general"
    )

    response = llm.invoke([SystemMessage(system_prompt)] + [last_message], config)
    
    decision = response.content.strip().lower()
    print(f"decision: {decision}")

    valid_decisions = {"sql_branch", "pdf_branch", "pdf_sql_branch", "process_return", "analyze_seller_reliability", "general"}
    if decision not in valid_decisions:
        decision = "general"
    return {"decide_path": decision}


def should_continue(state: AgentState) -> Literal[END, "run_query", "answer"]:
    """Decide whether to run query or go to answer."""
    print("should_continue")
    messages = state["messages"]
    last_message = messages[-1]
    # If the last model output is a plain answer (no tool calls), go to final answer node
    if not getattr(last_message, "tool_calls", None):
        return "answer"
    else:
        # Go directly to run_query (query validation removed for performance)
        return "run_query"


# Build the graph
builder = StateGraph(state_schema=AgentState, context_schema=AgentContext)

# Add all nodes
builder.add_node("decide_path", decide_path)
builder.add_node("pdf_branch", pdf_branch)
builder.add_node("call_get_schema", call_get_schema)
builder.add_node("generate_query", generate_query)
builder.add_node("run_query", run_query_node)

# Optional: a final answer node that uses PDF/SQL context to respond
builder.add_node("answer", answer_node)
builder.add_node("process_return", process_return_node)
builder.add_node("analyze_seller_reliability", analyze_seller_reliability_node_custom)

# Add routing edges
builder.add_edge(START, "decide_path")

builder.add_conditional_edges(
    "decide_path",
    lambda state: state["decide_path"],
    {
        "sql_branch": "call_get_schema",
        "pdf_branch": "pdf_branch",
        "pdf_sql_branch": "pdf_branch",  # then we'll chain SQL after PDF
        "process_return": "process_return",
        "analyze_seller_reliability": "analyze_seller_reliability",
        "general": "answer",
    }
)

# --- PDF path ---
# After loading PDF, go either to SQL or directly to answer
builder.add_conditional_edges(
    "pdf_branch",
    lambda state: (
        "call_get_schema" if state["decide_path"] == "pdf_sql_branch" else "answer"
    ),
    {
        "call_get_schema": "call_get_schema",
        "answer": "answer",
    },
)

# SQL workflow - get schema then generate query
builder.add_edge("call_get_schema", "generate_query")
builder.add_conditional_edges(
    "generate_query",
    should_continue,
    {
        "run_query": "run_query",
        "answer": "answer",
    }
)
builder.add_edge("run_query", "answer")  # After running query, go to answer node
# Tools go to answer node for final formatting, then to END
builder.add_edge("process_return", "answer")
builder.add_edge("analyze_seller_reliability", "answer")
builder.add_edge("answer", END)  # Answer node goes to END

# Compile the agent with checkpointing
checkpointer = InMemorySaver()
agent = builder.compile(checkpointer=checkpointer)

__all__ = ["agent", "AgentState", "process_order_return"]
