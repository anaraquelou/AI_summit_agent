# Return Policy Chat Agent

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
- **Document Storage**: PDF-based policy retrieval (`docs/polar-return-policy.pdf`)
- **State Management**: LangGraph with InMemorySaver for conversation checkpointing

## Agent Workflow

The agent uses an intelligent routing system that dynamically decides the best path:

1. **Route Decision** (`decide_path`): Analyzes user query and routes to:
   - `sql_branch`: Database queries only (e.g., "What is the status of order X?")
   - `pdf_branch`: Policy information only (e.g., "What is the return policy?")
   - `pdf_sql_branch`: Both PDF and database needed (e.g., "Is order X eligible for return?")
   - `general`: General conversation (e.g., "Who are you?")

2. **PDF Processing**: Loads and serializes PDF policy document when needed

3. **SQL Workflow**: 
   - Lists available tables
   - Gets schema for relevant tables
   - Generates SQL queries
   - Validates queries
   - Executes queries
   - Returns to answer node

4. **Answer Generation**: Combines PDF context and SQL results to provide comprehensive answers

5. **Return Processing**: When user confirms a return, updates order status to 'returned'

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
   - `docs/polar-return-policy.pdf` - Return policy document

4. Run the backend server:
```bash
python main.py
```

The backend will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the React development server:
```bash
npm start
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
