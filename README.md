# AI Analyst Chat Agent

A comprehensive LLM chat agent that intelligently routes queries to retrieve information from PDFs and query databases for data analysis, return requests, and seller reliability analysis. Built with FastAPI, LangChain/LangGraph, and React.

## Features

- **Intelligent Routing**: Automatically decides which tools are needed (PDF, SQL, both, seller analysis, return processing, or general) based on user queries
- **PDF Retrieval**: Extracts and uses return policy information from PDF documents
- **Database Integration**: Queries customer orders, payment information, order eligibility, and seller metrics
- **Return Processing**: Can process return requests by updating order status to 'return_processed' in the database
- **Seller Reliability Analysis**: Analyzes seller performance based on late delivery rates and review scores
- **Conversation Memory**: Maintains full chat history using LangGraph checkpointing
- **Modern UI**: Beautiful React frontend with real-time chat interface
- **LangGraph Workflow**: Dynamic routing workflow using LangGraph state management

## Architecture

- **Backend**: FastAPI with LangChain/LangGraph agent (v2.0.0)
- **Frontend**: React with modern chat interface
- **Database**: SQLite with e-commerce order data (`datasets/olist_ecommerce.db`)
- **LLM**: OpenAI GPT-5 for routing and answer generation
- **Document Storage**: PDF-based policy retrieval (`docs/BIX-return-policy.pdf`)
- **State Management**: LangGraph with InMemorySaver for conversation checkpointing

## Agent Workflow

The agent uses an intelligent routing system that dynamically decides the best path based on user queries. Here's how it works:

### Overview

Every user query goes through a **routing decision** that determines which tools and data sources are needed. The agent then follows a specific path through the workflow, gathering necessary information, and finally generates a comprehensive answer.

### Step-by-Step Process

1. **Route Decision** (`decide_path`): 
   - The LLM router analyzes the user query and selects one of six paths:
   - `sql_branch`: Database queries only
   - `pdf_branch`: Policy information only
   - `pdf_sql_branch`: Both PDF and database needed
   - `process_return`: Direct return processing for a specific order
   - `analyze_seller_reliability`: Seller reliability analysis based on performance metrics
   - `general`: General conversation (no tools needed)

2. **PDF Processing** (`pdf_branch`): 
   - If PDF is needed, loads and serializes the return policy PDF
   - Stores content in state for use in answer generation

3. **SQL Workflow** (when database is needed):
   - `call_get_schema`: Gets schema for relevant tables directly
   - `generate_query`: LLM generates SQL query from natural language
   - `run_query`: Executes the SQL query
   - Results are stored in conversation state

4. **Answer Generation** (`answer`): 
   - Combines PDF context (if loaded) and SQL results (if available)
   - Generates comprehensive, context-aware response
   - May include tool calls for return processing if user confirms

5. **Return Processing** (`process_return`): 
   - If user requests a return, executes `process_order_return` tool directly
   - Updates order status to 'return_processed' in database
   - Returns to answer node for final confirmation

6. **Seller Reliability Analysis** (`analyze_seller_reliability`): 
   - Analyzes seller performance based on late delivery rate (>5%) and average review score (<3.5)
   - Can analyze specific sellers or return top N unreliable sellers
   - Supports date range filtering for analysis
   - Returns formatted analysis with metrics

### Workflow Examples

#### Example 1: Policy Question (PDF Branch Only)

**User Query**: "Qual é o prazo máximo para devolução?"

**Workflow Path**:
1. `decide_path` → Routes to `pdf_branch`
2. `pdf_branch` → Loads PDF policy document
3. `answer` → Generates answer using PDF context
4. **Response**: "O prazo máximo para devolução é de 30 dias corridos após o recebimento do produto, conforme nossa política..."

**Key Nodes**: `decide_path` → `pdf_branch` → `answer` → `END`

---

#### Example 2: Order Status Query (SQL Branch Only)

**User Query**: "Qual é o status do pedido 6514b8ad8028c9f2cc2374ded245783f?"

**Workflow Path**:
1. `decide_path` → Routes to `sql_branch`
2. `call_get_schema` → Gets schema for relevant tables
3. `generate_query` → Generates: `SELECT order_status FROM orders WHERE order_id = '6514b8ad8028c9f2cc2374ded245783f'`
4. `run_query` → Executes query, gets status: "delivered"
5. `answer` → Generates response: "O pedido 6514b8ad8028c9f2cc2374ded245783f está com status 'delivered' (entregue)."

**Key Nodes**: `decide_path` → `call_get_schema` → `generate_query` → `run_query` → `answer` → `END`

---

#### Example 3: Eligibility Check (PDF + SQL Branch)

**User Query**: "O pedido e481f51cbdc54678b7cc49136f2d6af7 é elegível para devolução?"

**Workflow Path**:
1. `decide_path` → Routes to `pdf_sql_branch` (needs both)
2. `pdf_branch` → Loads PDF policy document
3. `call_get_schema` → Gets schema for relevant tables
4. `generate_query` → Generates query to get order details (status, delivery date, etc.)
5. `run_query` → Executes: Gets order info (delivered 15 days ago, status: "delivered")
6. `answer` → Combines PDF context (30-day return policy) + SQL results (15 days since delivery)
   - **Response**: "Sim, o pedido e481f51cbdc54678b7cc49136f2d6af7 é elegível para devolução. Foi entregue há 15 dias e nossa política permite devoluções em até 30 dias após a entrega."

**Key Nodes**: `decide_path` → `pdf_branch` → `call_get_schema` → `generate_query` → `run_query` → `answer` → `END`

---

#### Example 4: Processing a Return

**User Query**: "Quero devolver o pedido e481f51cbdc54678b7cc49136f2d6af7"

**Workflow Path**:
1. `decide_path` → Routes to `process_return` (direct return processing)
2. `process_return` → Executes `process_order_return("e481f51cbdc54678b7cc49136f2d6af7")`
   - Updates database: `UPDATE orders SET order_status = 'return_processed' WHERE order_id = '...'`
3. `answer` → Generates confirmation: "Pedido e481f51cbdc54678b7cc49136f2d6af7 foi marcado como devolvido (return_processed) com sucesso."

**Key Nodes**: `decide_path` → `process_return` → `answer` → `END`

---

#### Example 5: Seller Reliability Analysis

**User Query**: "O seller 3442f8959a84dea7ee197c632cb2df15 é confiável?"

**Workflow Path**:
1. `decide_path` → Routes to `analyze_seller_reliability`
2. `analyze_seller_reliability` → Executes analysis query
   - Calculates late delivery rate and average review score
   - Determines if seller is unreliable (>5% late deliveries AND <3.5 average score)
3. `answer` → Generates response with metrics
   - **Response**: "Não, o seller 3442f8959a84dea7ee197c632cb2df15 é considerado não confiável. Taxa de pedidos atrasados: 33.33% (acima de 5%), Nota média de reviews: 3.0/5.0 (abaixo de 3.5)"

**Key Nodes**: `decide_path` → `analyze_seller_reliability` → `answer` → `END`

---

#### Example 6: General Conversation

**User Query**: "Quem é você?"

**Workflow Path**:
1. `decide_path` → Routes to `general` (no tools needed)
2. `answer` → Generates response using system prompt
   - **Response**: "Olá! Sou um analista sênior de dados de E-commerce, especializado em análise de pedidos, devoluções e confiabilidade de vendedores..."

**Key Nodes**: `decide_path` → `answer` → `END`

### State Management

Throughout the workflow, the agent maintains state using `AgentState`:
- **messages**: Complete conversation history (automatically maintained)
- **pdf_context**: PDF content when loaded (persists until next query)
- **decide_path**: Current routing decision

The state is checkpointed using `InMemorySaver`, allowing the agent to maintain context across multiple turns in a conversation.

## Setup Instructions

### Prerequisites

- Python 3.11+
- Node.js 16+
- OpenAI API key

### Backend Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
# Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

3. Verify data files exist:
   - `datasets/olist_ecommerce.db` - SQLite database with order data
   - `docs/BIX-return-policy.pdf` - Return policy document

4. Run the backend server:
```bash
python main.py
```

The backend will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the React development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Usage

1. Open the React app in your browser
2. Start a conversation by asking about data analysis, returns, order information, or seller reliability
3. The agent will intelligently route your query:
   - Ask about policy → Routes to PDF branch
   - Ask about specific order/data → Routes to SQL branch
   - Check order eligibility → Routes to PDF+SQL branch
   - Request return processing → Routes to process_return
   - Ask about seller reliability → Routes to analyze_seller_reliability
   - General questions → Routes to general conversation

### Example Queries

- **Policy Questions**: 
  - "Como funciona a política de devolução?"
  - "Qual é o prazo máximo para devolução?"

- **Order Information**:
  - "Qual o id do cliente para o pedido 6514b8ad8028c9f2cc2374ded245783f?"
  - "Qual é o status do pedido 123?"

- **Eligibility Checks**:
  - "Você pode checar se o pedido e481f51cbdc54678b7cc49136f2d6af7 é elegível para devolução?"
  - "O pedido X pode ser devolvido?"

- **Return Processing**:
  - "Devolver o pedido e481f51cbdc54678b7cc49136f2d6af7"
  - "Processar devolução do pedido 12345"

- **Seller Reliability Analysis**:
  - "O seller 3442f8959a84dea7ee197c632cb2df15 é confiável?"
  - "Quais sellers são não confiáveis?"
  - "Quais os top 3 vendedores menos confiáveis?"
  - "Quais sellers tiveram mais de 5% de pedidos atrasados em 2024?"

## Agent Implementation

The agent is implemented in `agent/return_agent.py` with the following key components:

- **AgentState**: TypedDict defining state with messages, PDF context, and routing decisions
- **Routing Functions**: `decide_path` uses LLM to determine optimal query path (6 possible routes)
- **PDF Branch**: Loads and serializes PDF content when needed
- **SQL Tools**: Uses SQLDatabaseToolkit for database interactions
- **Return Tool**: `process_order_return` updates order status to 'return_processed'
- **Seller Reliability Tool**: `analyze_seller_reliability` analyzes seller performance metrics
- **Answer Node**: Generates final answers combining PDF and SQL context with professional analyst tone
- **Graph Structure**: LangGraph workflow with conditional edges for dynamic routing

## Database Schema

The system uses an e-commerce database with the following key tables:

- `customers`: Customer information
- `orders`: Order details, status, and delivery dates (status can be updated to 'return_processed')
- `order_items`: Items in each order (links orders to sellers)
- `order_payments`: Payment information
- `order_reviews`: Customer reviews (used for seller reliability analysis)
- `products`: Product information
- `sellers`: Seller information (used for reliability analysis)
- `category_translation`: Product category translations
- `geolocation`: Geographic location data

## API Endpoints

- `POST /chat`: Send messages to the chat agent
  - Request body: `{"message": "user query", "thread_id": "optional_thread_id"}`
  - Response: `{"message": "agent response", "status": "success"}`
- `GET /health`: Health check endpoint
- `GET /`: API information (returns version 2.0.0)


## Key Differences from Previous Implementation

This version uses a **routing-based architecture** instead of a fixed conversation flow:

- **Dynamic Routing**: Routes queries based on intent, not predefined steps
- **Context-Aware**: Maintains full conversation history automatically
- **Tool Integration**: Seamlessly combines PDF and SQL tools based on query needs
- **Flexible Conversations**: Supports natural conversation flow without rigid stages
- **Memory**: Uses LangGraph checkpointing for persistent conversation state

## Development

The agent code is located in `agent/return_agent.py`. Key functions:

- `decide_path()`: Router function that determines query path (6 routes: sql_branch, pdf_branch, pdf_sql_branch, process_return, analyze_seller_reliability, general)
- `pdf_branch()`: Loads PDF content into state
- `call_get_schema()`: Gets database schema for relevant tables
- `generate_query()`: Generates SQL queries from natural language
- `run_query()`: Executes validated SQL queries
- `answer_node()`: Generates final responses with context (professional analyst tone)
- `process_order_return()`: Updates order status to 'return_processed' for returns
- `analyze_seller_reliability()`: Analyzes seller reliability based on late deliveries and review scores
- `process_return_node()`: Handles direct return processing workflow
- `analyze_seller_reliability_node_custom()`: Handles seller reliability analysis workflow

## License

This project is part of a LangChain AI Summit demonstration.
