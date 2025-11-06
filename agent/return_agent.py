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
PDF_PATH = PROJECT_ROOT / "docs" / "BIX-return-policy.pdf"


class AgentState(TypedDict):
    """State of the agent. Contains messages, PDF content, and routing info."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    pdf_context: str
    decide_path: Literal["sql_branch", "pdf_branch", "pdf_sql_branch", "general"]


# Initialize LLMs
llm = ChatOpenAI(model="gpt-5", temperature=0)
llm_router = ChatOpenAI(model="gpt-5", temperature=0)
llm_answer = ChatOpenAI(model="gpt-5", temperature=0)

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
    """Atualiza o status do pedido no banco de dados para 'return_requested' (devolvido).
    
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
        cursor.execute("UPDATE orders SET order_status = 'return_requested' WHERE order_id = ?", (order_id,))
        conn.commit()
        conn.close()
        
        return f"Pedido {order_id} foi marcado como devolvido (return_requested) com sucesso."
    except Exception as e:
        return f"Erro ao processar devolução: {str(e)}"


# Create return order tool
return_order_tool = StructuredTool.from_function(
    func=process_order_return,
    name="process_order_return",
    description="Atualiza o status de um pedido para 'return_requested' (devolvido) no banco de dados. Use esta ferramenta quando o usuário confirmar que deseja devolver ou cancelar um pedido específico."
)

return_order_node = ToolNode([return_order_tool], name="process_return")


def analyze_seller_reliability(start_date: str, end_date: str) -> str:
    """Analyzes seller reliability based on late delivery rate and average review score.
    
    A seller is considered unreliable if they have more than 5% of orders delivered late
    AND an average review score below 3.5 within the specified date range.
    
    Args:
        start_date: Start date in format 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM-DD'
        
    Returns:
        Formatted string with analysis results showing unreliable sellers
    """
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        # Query to calculate seller reliability metrics
        query = """
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
            WHERE date(o.order_purchase_timestamp) >= date(?)
            AND date(o.order_purchase_timestamp) <= date(?)
            AND o.order_status = 'delivered'
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
        
        cursor.execute(query, (start_date, end_date))
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return f"Nenhum vendedor não confiável encontrado no período de {start_date} a {end_date}."
        
        # Format results
        result_lines = [
            f"Vendedores não confiáveis no período de {start_date} a {end_date}:",
            "",
            "=" * 80
        ]
        
        for row in results:
            seller_id, city, state, total_orders, late_orders, late_pct, avg_score = row
            result_lines.append(
                f"Vendedor: {seller_id} | {city}, {state}\n"
                f"  - Total de pedidos: {total_orders}\n"
                f"  - Pedidos atrasados: {late_orders} ({late_pct}%)\n"
                f"  - Nota média de reviews: {avg_score}/5.0"
            )
            result_lines.append("-" * 80)
        
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
    description="Analisa a confiabilidade dos vendedores com base em pedidos atrasados e avaliações. Um vendedor é considerado não confiável se tiver mais de 5% dos pedidos atrasados E nota média de review abaixo de 3.5. Use esta ferramenta quando o usuário perguntar sobre vendedores com desempenho ruim, violação de regras internas, ou análise de confiabilidade de vendedores. Parâmetros: start_date (formato 'YYYY-MM-DD') e end_date (formato 'YYYY-MM-DD')."
)

seller_reliability_node = ToolNode([seller_reliability_tool], name="analyze_seller_reliability")


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
    """Create tool call for getting schema directly (no LLM needed)."""
    print("call_get_schema tool")
    # Create tool call directly - no LLM call needed
    tool_call = {
        "name": "sql_db_schema",
        "args": {"table_names": ""},  # Empty string gets all tables
        "id": f"schema_call_{len(state['messages'])}",
        "type": "tool_call",
    }
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])
    return {"messages": [tool_call_message]}

TABLE_HINTS = """
Tables:
- orders: This is the core dataset. From each order you might find all other information.
- order_items: This dataset includes data about the items purchased within each order.
Example:
The order_id = 00143d0f86d6fbd9f9b38ab440ac16f5 has 3 items (same product). Each item has the freight calculated accordingly to its measures and weight. To get the total freight value for each order you just have to sum.
The total order_item value is: 21.33 * 3 = 63.99
The total freight value is: 15.10 * 3 = 45.30
The total order value (product + freight) is: 45.30 + 63.99 = 109.29
- products: This dataset includes data about the products sold by BIX.
- customers: This dataset has information about the customer and its location. Use it to identify unique customers in the orders dataset and to find the orders delivery location.
At our system each order is assigned to a unique customer_id. This means that the same customer will get different ids for different orders. The purpose of having a customer_unique_id on the dataset is to allow you to identify customers that made repurchases at the store. Otherwise you would find that each order had a different customer associated with.
- sellers: This dataset includes data about the sellers that fulfilled orders made at BIX. Use it to find the seller location and to identify which seller fulfilled each product.
- geolocation: This dataset has information Brazilian zip codes and its lat/lng coordinates. Use it to plot maps and find distances between sellers and customers.
- category_translation: Translates the product_category_name to english.
- order_payments: This dataset includes data about the orders payment options.
- order_reviews: This dataset includes data about the reviews made by the customers.
After a customer purchases the product from BIX Store a seller gets notified to fulfill that order. Once the customer receives the product, or the estimated delivery date is due, the customer gets a satisfaction survey by email where he can give a note for the purchase experience and write down some comments.
Orders table:
- order_id: unique identifier of an order.
- customer_id: key to the customer dataset. Each order has a unique customer_id.
- order_status: reference to the order status (delivered, shipped, cancelled, return_requested, etc).
- order_purchase_timestamp: shows the purchase timestamp.
- order_approved_at: shows the payment approval timestamp.
- order_delivered_carrier_date: shows the order posting timestamp. When it was handled to the logistic partner.
- order_delivered_customer_date: shows the actual order delivery date to the customer.
- order_estimated_delivery_date: shows the estimated delivery date that was informed to customer at the purchase moment.
Order items table:
- order_item_id: sequential number identifying number of items included in the same order.
- order_id: order unique identifier
- product_id: product unique identifier
- seller_id: seller unique identifier
- shipping_limit_date: shows the seller shipping limit date for handling the order over to the logistic partner.
- price: item price
- freight_value: item freight value item (if an order has more than one item the freight value is splitted between items)
Customers table:
- customer_id: key to the orders dataset. Each order has a unique customer_id.
- customer_unique_id: unique identifier of a customer.
- customer_zip_code_prefix: first five digits of customer zip code
- customer_city: customer city name
- customer_state: customer state
Sellers table:
- seller_id: seller unique identifier
- seller_zip_code_prefix: first 5 digits of seller zip code
- seller_city: seller city
- seller_state: seller state
Geolocation table:
- geolocation_zip_code_prefix: first 5 digits of zip code
- geolocation_lat: latitude coordinate
- geolocation_lng: longitude coordinate
- geolocation_city: city name
- geolocation_state: state
Category translation table:
- product_category_name: category name in Portuguese
- product_category_name_english: category name in English
Order payments table:
- order_id: unique identifier of an order.
- payment_sequential: a customer may pay an order with more than one payment method. If he does so, a sequence will be created to accommodate all payments.
- payment_type: method of payment chosen by the customer.
- payment_installments: number of installments chosen by the customer.
- payment_value: transaction value.
Order reviews table:
- review_id: unique review identifier
- order_id: unique order identifier
- review_score: Note ranging from 1 to 5 given by the customer on a satisfaction survey.
- review_comment_title: Comment title from the review left by the customer, in Portuguese.
- review_comment_message: Comment message from the review left by the customer, in Portuguese.
- review_creation_date: Shows the date in which the satisfaction survey was sent to the customer.
- review_answer_timestamp: Shows satisfaction survey answer timestamp.
Products table:
- product_id: unique product identifier
- product_category_name: root category of product, in Portuguese.
- product_name_length: number of characters extracted from the product name.
- product_description_lengh: number of characters extracted from the product description.
- product_photos_qty: number of product published photos.
- product_weight_g: product weight measured in grams.
- product_length_cm: product length measured in centimeters.
- product_height_cm: product height measured in centimeters.
- product_width_cm: product width measured in centimeters.
"""

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
    DATABASE_DOCS=TABLE_HINTS
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

    # Disponibiliza as tools para o modelo
    llm_with_tools = llm_answer.bind_tools([return_order_tool, seller_reliability_tool])

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
- Quando o usuário CONFIRMAR que deseja devolver ou cancelar um pedido específico, use a ferramenta process_order_return para atualizar o status do pedido para 'return_requested'.
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
A política de devolução da BIX E-commerce permite devoluções em até 30 dias após a entrega.  
Portanto, sim, o pedido 1234 é elegível para devolução.
Usuário: Qual é o prazo máximo para devolução?
Agente: A política de devolução da BIX E-commerce informa que o prazo máximo é de 30 dias corridos após o recebimento do produto.
Usuário: Quero saber o status do pedido 5678.
Agente: O pedido 5678 está com o status "Processando devolução".
</Exemplos>

<BIX E-commerce>
Plataforma de e-commerce especializada em vendas de produtos de beleza e cuidados pessoais
lojavirtual@bix.com | Whataspp: +55 11 4862-7901
</BIX E-commerce>

<Não fazer>
- NUNCA corrija o usuário na maneira de escrever.
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


def should_process_tool(state: AgentState) -> Literal["process_return", "analyze_seller_reliability", END]:
    """Decide qual tool processar após resposta do answer_node"""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    # Verifica se a última mensagem é uma AIMessage com tool calls
    if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get("name")
            if tool_name == "process_order_return":
                return "process_return"
            elif tool_name == "analyze_seller_reliability":
                return "analyze_seller_reliability"
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
builder.add_node("analyze_seller_reliability", seller_reliability_node)

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

# End of pipeline - check if answer node wants to process any tool
builder.add_conditional_edges(
    "answer",
    should_process_tool,
    {
        "process_return": "process_return",
        "analyze_seller_reliability": "analyze_seller_reliability",
        END: END,
    },
)
# After processing tools, go back to answer node for final confirmation
builder.add_edge("process_return", "answer")
builder.add_edge("analyze_seller_reliability", "answer")

# Compile the agent with checkpointing
checkpointer = InMemorySaver()
agent = builder.compile(checkpointer=checkpointer)

__all__ = ["agent", "AgentState", "process_order_return"]
