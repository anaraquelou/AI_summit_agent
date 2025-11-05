# AI Analyst Chat Agent

A comprehensive LLM chat agent that intelligently routes queries to retrieve information from PDFs and query databases to help customers with return requests. Built with FastAPI, LangChain/LangGraph, and React.

## Features

- **Intelligent Routing**: Automatically decides which tools are needed (PDF, SQL, both, or general) based on user queries
- **PDF Retrieval**: Extracts and uses return policy information from PDF documents
- **Database Integration**: Queries customer orders, payment information, and order eligibility
- **Return Processing**: Can process return requests by updating order status in the database
- **Conversation Memory**: Maintains full chat history using LangGraph checkpointing
- **Modern UI**: Beautiful React frontend with real-time chat interface
- **LangGraph Workflow**: Dynamic routing workflow using LangGraph state management

## Architecture

- **Backend**: FastAPI with LangChain/LangGraph agent
- **Frontend**: React with modern chat interface
- **Database**: SQLite with e-commerce order data (`datasets/olist_ecommerce.db`)
- **LLM**: OpenAI GPT-5o for routing and answer generation
- **Document Storage**: PDF-based policy retrieval (`docs/BIX-return-policy.pdf`)
- **State Management**: LangGraph with InMemorySaver for conversation checkpointing

## Agent Workflow

The agent uses an intelligent routing system that dynamically decides the best path based on user queries. Here's how it works:

### Overview

Every user query goes through a **routing decision** that determines which tools and data sources are needed. The agent then follows a specific path through the workflow, gathering necessary information, and finally generates a comprehensive answer.

### Step-by-Step Process

1. **Route Decision** (`decide_path`): 
   - The LLM router analyzes the user query and selects one of four paths:
   - `sql_branch`: Database queries only
   - `pdf_branch`: Policy information only
   - `pdf_sql_branch`: Both PDF and database needed
   - `general`: General conversation (no tools needed)

2. **PDF Processing** (`pdf_branch`): 
   - If PDF is needed, loads and serializes the return policy PDF
   - Stores content in state for use in answer generation

3. **SQL Workflow** (when database is needed):
   - `list_tables`: Lists all available database tables
   - `call_get_schema` → `get_schema`: Gets schema for relevant tables
   - `generate_query`: LLM generates SQL query from natural language
   - `check_query`: Validates and corrects SQL query before execution
   - `run_query`: Executes the validated SQL query
   - Results are stored in conversation state

4. **Answer Generation** (`answer`): 
   - Combines PDF context (if loaded) and SQL results (if available)
   - Generates comprehensive, context-aware response
   - May include tool calls for return processing if user confirms

5. **Return Processing** (`process_return`): 
   - If user confirms a return, executes `process_order_return` tool
   - Updates order status to 'returned' in database
   - Returns to answer node for final confirmation

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
2. `list_tables` → Lists available database tables
3. `call_get_schema` → `get_schema` → Gets schema for `orders` table
4. `generate_query` → Generates: `SELECT order_status FROM orders WHERE order_id = '6514b8ad8028c9f2cc2374ded245783f'`
5. `check_query` → Validates SQL query
6. `run_query` → Executes query, gets status: "delivered"
7. `answer` → Generates response: "O pedido 6514b8ad8028c9f2cc2374ded245783f está com status 'delivered' (entregue)."

**Key Nodes**: `decide_path` → `list_tables` → `call_get_schema` → `get_schema` → `generate_query` → `check_query` → `run_query` → `answer` → `END`

---

#### Example 3: Eligibility Check (PDF + SQL Branch)

**User Query**: "O pedido e481f51cbdc54678b7cc49136f2d6af7 é elegível para devolução?"

**Workflow Path**:
1. `decide_path` → Routes to `pdf_sql_branch` (needs both)
2. `pdf_branch` → Loads PDF policy document
3. `list_tables` → Lists available tables
4. `call_get_schema` → `get_schema` → Gets schema for `orders` table
5. `generate_query` → Generates query to get order details (status, delivery date, etc.)
6. `check_query` → Validates SQL
7. `run_query` → Executes: Gets order info (delivered 15 days ago, status: "delivered")
8. `answer` → Combines PDF context (30-day return policy) + SQL results (15 days since delivery)
   - **Response**: "Sim, o pedido e481f51cbdc54678b7cc49136f2d6af7 é elegível para devolução. Foi entregue há 15 dias e nossa política permite devoluções em até 30 dias após a entrega."

**Key Nodes**: `decide_path` → `pdf_branch` → `list_tables` → `call_get_schema` → `get_schema` → `generate_query` → `check_query` → `run_query` → `answer` → `END`

---

#### Example 4: Processing a Return

**User Query**: "Quero devolver o pedido e481f51cbdc54678b7cc49136f2d6af7"

**Workflow Path**:
1. `decide_path` → Routes to `pdf_sql_branch` (checks eligibility first)
2. ... (same as Example 3, checking eligibility)
3. `answer` → Determines eligibility and offers to process return
   - User confirms: "Sim, processe a devolução"
4. `answer` → Generates tool call for `process_order_return`
5. `process_return` → Executes `process_order_return("e481f51cbdc54678b7cc49136f2d6af7")`
   - Updates database: `UPDATE orders SET order_status = 'returned' WHERE order_id = '...'`
6. `answer` → Generates confirmation: "Pedido e481f51cbdc54678b7cc49136f2d6af7 foi marcado como devolvido com sucesso."

**Key Nodes**: ... → `answer` → `process_return` → `answer` → `END`

---

#### Example 5: General Conversation

**User Query**: "Quem é você?"

**Workflow Path**:
1. `decide_path` → Routes to `general` (no tools needed)
2. `answer` → Generates response using system prompt
   - **Response**: "Olá! Sou um assistente especializado em gestão de pedidos e devoluções da BIX E-commerce..."

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
2. Start a conversation by asking about returns or order information
3. The agent will intelligently route your query:
   - Ask about policy → Routes to PDF branch
   - Ask about specific order → Routes to SQL branch
   - Check order eligibility → Routes to PDF+SQL branch
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
  - After confirming eligibility, the agent can process returns by updating order status

## Agent Implementation

The agent is implemented in `agent/return_agent.py` with the following key components:

- **AgentState**: TypedDict defining state with messages, PDF context, and routing decisions
- **Routing Functions**: `decide_path` uses LLM to determine optimal query path
- **PDF Branch**: Loads and serializes PDF content when needed
- **SQL Tools**: Uses SQLDatabaseToolkit for database interactions
- **Return Tool**: `process_order_return` updates order status to 'returned'
- **Answer Node**: Generates final answers combining PDF and SQL context
- **Graph Structure**: LangGraph workflow with conditional edges for dynamic routing

## Database Schema

The system uses an e-commerce database with the following key tables:

- `customers`: Customer information
- `orders`: Order details, status, and delivery dates
- `order_items`: Items in each order
- `order_payments`: Payment information
- `order_reviews`: Customer reviews
- `products`: Product information
- `sellers`: Seller information

## API Endpoints

- `POST /chat`: Send messages to the chat agent
- `GET /health`: Health check endpoint
- `GET /`: API information


## Key Differences from Previous Implementation

This version uses a **routing-based architecture** instead of a fixed conversation flow:

- **Dynamic Routing**: Routes queries based on intent, not predefined steps
- **Context-Aware**: Maintains full conversation history automatically
- **Tool Integration**: Seamlessly combines PDF and SQL tools based on query needs
- **Flexible Conversations**: Supports natural conversation flow without rigid stages
- **Memory**: Uses LangGraph checkpointing for persistent conversation state

## Development

The agent code is located in `agent/return_agent.py`. Key functions:

- `decide_path()`: Router function that determines query path
- `pdf_branch()`: Loads PDF content into state
- `generate_query()`: Generates SQL queries from natural language
- `answer_node()`: Generates final responses with context
- `process_order_return()`: Updates order status for returns

## License

This project is part of a LangChain AI Summit demonstration.
