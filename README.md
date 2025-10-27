# Return Policy Chat Agent

A comprehensive LLM chat agent that can retrieve information from PDFs and query databases to help customers with return requests. Built with FastAPI, LangChain/LangGraph, and React.

## Features

- **PDF Retrieval**: Extracts return policy information from PDF documents
- **Database Integration**: Queries customer orders and payment information
- **Conversational Flow**: Guides users through the return process step by step
- **Modern UI**: Beautiful React frontend with real-time chat interface
- **LangGraph Workflow**: Structured conversation flow using LangGraph

## Architecture

- **Backend**: FastAPI with LangChain/LangGraph agent
- **Frontend**: React with modern chat interface
- **Database**: SQLite with e-commerce order data
- **LLM**: OpenAI GPT-3.5-turbo
- **Vector Store**: FAISS for PDF document retrieval

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

3. Run the backend server:
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
2. Start a conversation by asking about returns
3. Follow the agent's guidance through the return process:
   - Provide your customer ID
   - Select an order to return
   - Confirm the return request
   - Receive confirmation and refund information

## Conversation Flow

The agent follows this structured workflow:

1. **Greeting**: Detects return intent and welcomes the user
2. **Policy Information**: Retrieves and explains return policy from PDF
3. **Conditions Check**: Verifies customer ID and return eligibility
4. **Order Selection**: Shows recent orders and lets user select one
5. **Confirmation**: Processes the return and updates database
6. **Completion**: Provides final confirmation and refund details

## Database Schema

The system uses an e-commerce database with the following key tables:

- `customers`: Customer information
- `orders`: Order details and status
- `order_items`: Items in each order
- `order_payments`: Payment information

## API Endpoints

- `POST /chat`: Send messages to the chat agent
- `GET /health`: Health check endpoint
- `GET /`: API information

## Customization

- Modify the PDF path in `agent/return_agent.py` to use your own return policy
- Update the database schema and queries for your specific data structure
- Customize the conversation flow by modifying the LangGraph nodes
- Style the frontend by editing `frontend/frontend/src/App.css`

## Troubleshooting

- Ensure your OpenAI API key is correctly set in the `.env` file
- Check that the PDF file exists at `docs/polar-return-policy.pdf`
- Verify the database file exists at `datasets/olist_ecommerce.db`
- Make sure both backend and frontend servers are running
