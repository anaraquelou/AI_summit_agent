# AI Analyst Chat Agent

A LangGraph-based chat agent that intelligently routes queries to retrieve information from PDFs and query databases for data analysis, return requests, and seller reliability analysis.

## Features

- **Intelligent Data Source Selection**: The agent automatically determines which data sources are needed for each query:
  - **PDF only**: For policy questions that don't require database lookup
  - **Database only**: For data queries that don't need policy context
  - **Both PDF + Database**: For complex queries requiring policy rules and order data (e.g., eligibility checks)
- **Smart Routing**: Routes queries to the appropriate tools (PDF, SQL, both, seller analysis, return processing, or general conversation)
- **PDF Retrieval**: Extracts return policy information from PDF documents
- **Database Integration**: Queries customer orders, payment information, and seller metrics
- **Return Processing**: Updates order status to 'return_processed' in the database
- **Seller Reliability Analysis**: Analyzes seller performance based on late delivery rates and review scores
- **Conversation Memory**: Maintains chat history using LangGraph checkpointing

## Architecture

- **Backend**: FastAPI with LangChain/LangGraph agent
- **Frontend**: React chat interface
- **Database**: SQLite (`datasets/olist_ecommerce.db`)
- **LLM**: OpenAI GPT-5
- **Documents**: PDF policy retrieval (`docs/BIX-return-policy.pdf`)

## Agent Workflow

The agent intelligently selects which data sources are needed for each query:

![Agent Workflow](img/return_agent_workflow.png)

1. **Route Decision** (`decide_path`): LLM analyzes query and selects one of 6 paths:
   - `sql_branch`: Database queries only
   - `pdf_branch`: Policy information only
   - `pdf_sql_branch`: Both PDF and database needed
   - `process_return`: Direct return processing
   - `analyze_seller_reliability`: Seller reliability analysis
   - `general`: General conversation (no data sources needed)

2. **Execution**: Agent follows the selected path, gathering necessary information from the chosen data source(s)

3. **Answer Generation**: Combines all gathered context to generate a comprehensive response

## Setup

### Prerequisites

- Python 3.11+
- Node.js 16+
- OpenAI API key
- DBeaver Community (optional, for database inspection)

### Environment Setup

1. Copy the example environment file:
```bash
# Linux/Mac
cp .env.example .env

# Windows
copy .env.example .env
```

2. Edit `.env` and configure your API keys:
```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional: LangSmith tracing for monitoring and debugging
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=return-agent
```

**Note**: Get your LangSmith API key from [smith.langchain.com](https://smith.langchain.com/). Tracing is optional but recommended for debugging and monitoring agent behavior.

### Database Setup

1. Navigate to the datasets directory:
```bash
cd datasets
```

2. Run the database creation script:
```bash
python create_database.py
```

This will create `olist_ecommerce.db` from the CSV files in the directory.

### Accessing the Database with DBeaver Community

1. **Install DBeaver Community**: Download from [dbeaver.io](https://dbeaver.io/download/)

2. **Create a new SQLite connection**:
   - Open DBeaver
   - Click "New Database Connection" (plug icon)
   - Select "SQLite"
   - Click "Next"

3. **Configure the connection**:
   - **Path**: Browse to `datasets/olist_ecommerce.db` in your project directory
     - Example: `C:\path\to\langchain-AI-summit\datasets\olist_ecommerce.db`
   - Click "Test Connection" to verify
   - Click "Finish"

4. **Explore the database**:
   - Expand the connection to see all tables
   - Right-click any table → "View Data" to browse records
   - Use the SQL Editor to run custom queries

### Backend

```bash
pip install -r requirements.txt
python main.py
```

Backend runs at `http://localhost:8000`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:3000`

## Usage

The agent routes queries based on intent:
- Policy questions → PDF branch
- Order/data queries → SQL branch
- Eligibility checks → PDF + SQL branch
- Return requests → process_return
- Seller analysis → analyze_seller_reliability
- General questions → general conversation

### Example Queries

**Requires database only:**
- "Há quantos pedidos com status cancelado?"
- "Quais 3 vendedores tiveram o maior número de entregas atrasadas em 2024?"

**Required PDF only:**
- "No caso de devolução quem arca com custo do envio?"
- "Qual o prazo de devolução por defeito da BIX?"

**Requires database + PDF:**
- "O pedido dd787ad9c97e5504d6ea0bd294906902 está dentro do prazo de devolução por arrependimento?"
- "Quantos pedidos entregues da base de dados estão dentro do prazo de devolução por defeito? Considere que são todos itens não duráveis"

**Call seller reliability tool:**
- "O seller 3442f8959a84dea7ee197c632cb2df15 é confiável?"

**Call return order process tool:**
- "Processe devolução do pedido 2591f6277be80b0c25627c745ec900c4"

## API Endpoints

- `POST /chat`: Send messages to the chat agent
  - Request: `{"message": "user query", "thread_id": "optional"}`
  - Response: `{"message": "agent response", "status": "success"}`
- `GET /health`: Health check
- `GET /`: API information

## Testing

```bash
pytest tests/
```

## Development

Main agent code: `agent/return_agent.py`
- Uses LangGraph for workflow orchestration
- Routes queries through 6 possible paths based on LLM decision
- Maintains conversation state with checkpointing
