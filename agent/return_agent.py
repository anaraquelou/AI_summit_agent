from __future__ import annotations
from typing import Dict, List, Any, Optional, TypedDict
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langgraph.graph import StateGraph, END
# ToolExecutor not used in this implementation
import sqlite3
import json
import os
from datetime import datetime

class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    customer_id: Optional[str]
    current_order_id: Optional[str]
    return_conditions_met: bool
    conversation_stage: str  # "greeting", "policy_info", "conditions_check", "order_selection", "confirmation", "completion"

class ReturnPolicyAgent:
    def __init__(self):
        # Get OpenAI API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Get model from environment or default to gpt-4o-mini
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.7,
            api_key=api_key
        )
        
        # Initialize PDF retriever
        self.pdf_retriever = self._setup_pdf_retriever()
        
        # Initialize database connection
        self.db_path = "datasets/olist_ecommerce.db"
        
        # Create the agent graph
        self.graph = self._create_agent_graph()
        
        # Track conversation state across messages
        self.conversation_state = {
            "stage": "greeting",
            "customer_id": None,
            "current_order_id": None,
            "return_conditions_met": False
        }
        
    def _setup_pdf_retriever(self):
        """Load and process the return policy PDF"""
        try:
            loader = PyPDFLoader("docs/polar-return-policy.pdf")
            documents = loader.load()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            # Create embeddings and vector store
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            
            embeddings = OpenAIEmbeddings(api_key=api_key)
            vectorstore = FAISS.from_documents(splits, embeddings)
            
            return vectorstore.as_retriever(search_kwargs={"k": 3})
        except Exception as e:
            print(f"Error setting up PDF retriever: {e}")
            return None
    
    def _create_agent_graph(self):
        """Create the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("greeting", self._greeting_node)
        workflow.add_node("policy_info", self._policy_info_node)
        workflow.add_node("conditions_check", self._conditions_check_node)
        workflow.add_node("order_selection", self._order_selection_node)
        workflow.add_node("confirmation", self._confirmation_node)
        workflow.add_node("completion", self._completion_node)
        
        # Add edges - simple linear flow
        workflow.set_entry_point("greeting")
        workflow.add_edge("greeting", "policy_info")
        workflow.add_edge("policy_info", "conditions_check")
        workflow.add_edge("conditions_check", "order_selection")
        workflow.add_edge("order_selection", "confirmation")
        workflow.add_edge("confirmation", "completion")
        workflow.add_edge("completion", END)
        
        return workflow.compile()
    
    def process_message(self, message: str, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process incoming message through the agent"""
        
        # Initialize state with persistent conversation stage
        state = AgentState(
            messages=conversation_history + [{"role": "user", "content": message}],
            customer_id=self.conversation_state["customer_id"],
            current_order_id=self.conversation_state["current_order_id"],
            return_conditions_met=self.conversation_state["return_conditions_met"],
            conversation_stage=self.conversation_state["stage"]
        )
        
        # Run the graph
        result = self.graph.invoke(state)
        
        # Update persistent state from result
        self.conversation_state["stage"] = result.get("conversation_stage", self.conversation_state["stage"])
        self.conversation_state["customer_id"] = result.get("customer_id", self.conversation_state["customer_id"])
        self.conversation_state["current_order_id"] = result.get("current_order_id", self.conversation_state["current_order_id"])
        self.conversation_state["return_conditions_met"] = result.get("return_conditions_met", self.conversation_state["return_conditions_met"])
        
        # Extract the final message
        final_message = result["messages"][-1]["content"] if result["messages"] else "I'm sorry, I couldn't process your request."
        
        return {
            "message": final_message,
            "conversation_history": result["messages"]
        }
    
    def _greeting_node(self, state: AgentState) -> AgentState:
        """Handle initial greeting and intent detection"""
        user_message = state["messages"][-1]["content"].lower()
        
        if any(keyword in user_message for keyword in ["return", "refund", "cancel", "exchange"]):
            response = "I'd be happy to help you with your return request! Let me first provide you with our return policy information and then we can proceed with your specific case."
            state["conversation_stage"] = "policy_info"
        else:
            response = "Hello! I'm here to help you with returns and refunds. How can I assist you today?"
            state["conversation_stage"] = "greeting"
        
        state["messages"].append({"role": "assistant", "content": response})
        # Update persistent state
        self.conversation_state["stage"] = state["conversation_stage"]
        return state
    
    def _policy_info_node(self, state: AgentState) -> AgentState:
        """Retrieve and present return policy information"""
        if self.pdf_retriever:
            try:
                # Retrieve relevant policy information
                docs = self.pdf_retriever.get_relevant_documents("return policy conditions requirements")
                policy_text = "\n".join([doc.page_content for doc in docs])
                
                # Generate response using LLM
                prompt = f"""
                Based on the following return policy information, provide a clear summary of the main return conditions:
                
                {policy_text}
                
                Please summarize the key points about return conditions and requirements in a friendly, helpful tone.
                """
                
                response = self.llm.invoke(prompt).content
                
                # Add follow-up questions
                response += "\n\nTo proceed with your return, I'll need to ask you a few questions:\n1. What is your customer ID?\n2. Can you confirm that your item meets our return conditions?"
                
            except Exception as e:
                response = "I can help you with returns! Our standard return policy includes:\n- Items must be returned within 30 days\n- Items must be in original condition\n- Original packaging preferred\n\nTo proceed, I'll need your customer ID. What is your customer ID?"
        else:
            response = "I can help you with returns! Our standard return policy includes:\n- Items must be returned within 30 days\n- Items must be in original condition\n- Original packaging preferred\n\nTo proceed, I'll need your customer ID. What is your customer ID?"
        
        state["conversation_stage"] = "conditions_check"
        state["messages"].append({"role": "assistant", "content": response})
        # Update persistent state
        self.conversation_state["stage"] = state["conversation_stage"]
        return state
    
    def _conditions_check_node(self, state: AgentState) -> AgentState:
        """Check return conditions and get customer ID"""
        user_message = state["messages"][-1]["content"]
        
        # Extract customer ID from message
        customer_id = self._extract_customer_id(user_message)
        
        if customer_id:
            state["customer_id"] = customer_id
            self.conversation_state["customer_id"] = customer_id
            
            # Check if customer exists
            if self._customer_exists(customer_id):
                response = f"Thank you! I found your account. Now let me check your recent orders to see what you'd like to return."
                state["conversation_stage"] = "order_selection"
                self.conversation_state["stage"] = "order_selection"
            else:
                response = "I couldn't find a customer with that ID. Could you please double-check your customer ID?"
                state["conversation_stage"] = "conditions_check"
                self.conversation_state["stage"] = "conditions_check"
        else:
            response = "I need your customer ID to look up your orders. Could you please provide your customer ID?"
            state["conversation_stage"] = "conditions_check"
            self.conversation_state["stage"] = "conditions_check"
        
        state["messages"].append({"role": "assistant", "content": response})
        return state
    
    def _order_selection_node(self, state: AgentState) -> AgentState:
        """Show recent orders and let user select one to return"""
        customer_id = state["customer_id"]
        
        # Get last 3 orders for the customer
        orders = self._get_customer_orders(customer_id, limit=3)
        
        if orders:
            response = "Here are your last 3 orders:\n\n"
            for i, order in enumerate(orders, 1):
                response += f"{i}. Order ID: {order['order_id']}\n"
                response += f"   Date: {order['order_purchase_timestamp']}\n"
                response += f"   Status: {order['order_status']}\n"
                response += f"   Total: ${order.get('total_amount', 'N/A')}\n\n"
            
            response += "Which order would you like to return? Please provide the order ID or number (1, 2, or 3)."
        else:
            response = "I couldn't find any orders for your account. Please check your customer ID or contact support."
            state["conversation_stage"] = "conditions_check"
            self.conversation_state["stage"] = "conditions_check"
        
        state["conversation_stage"] = "confirmation"
        self.conversation_state["stage"] = "confirmation"
        state["messages"].append({"role": "assistant", "content": response})
        return state
    
    def _confirmation_node(self, state: AgentState) -> AgentState:
        """Confirm the return and update order status"""
        user_message = state["messages"][-1]["content"]
        
        # Extract order ID from user message
        order_id = self._extract_order_id(user_message, state["customer_id"])
        
        if order_id:
            state["current_order_id"] = order_id
            self.conversation_state["current_order_id"] = order_id
            
            # Update order status to "returned"
            if self._update_order_status(order_id, "returned"):
                # Get payment information
                payment_info = self._get_order_payment_info(order_id)
                
                response = f"Perfect! I've successfully processed your return request for order {order_id}.\n\n"
                response += f"Payment Information:\n"
                response += f"- Payment Method: {payment_info.get('payment_type', 'N/A')}\n"
                response += f"- Amount: ${payment_info.get('payment_value', 'N/A')}\n"
                response += f"- Installments: {payment_info.get('payment_installments', 'N/A')}\n\n"
                response += "Your refund will be processed within 3 business days using the original payment method.\n\n"
                
                # Show order summary
                order_summary = self._get_order_summary(order_id)
                response += "Order Summary:\n"
                response += f"- Order ID: {order_summary['order_id']}\n"
                response += f"- Status: {order_summary['order_status']}\n"
                response += f"- Purchase Date: {order_summary['order_purchase_timestamp']}\n"
                response += f"- Items: {order_summary.get('item_count', 'N/A')} items\n"
                
                state["conversation_stage"] = "completion"
                self.conversation_state["stage"] = "completion"
            else:
                response = "I encountered an issue processing your return. Please try again or contact support."
                state["conversation_stage"] = "order_selection"
                self.conversation_state["stage"] = "order_selection"
        else:
            response = "I couldn't identify which order you want to return. Please provide the order ID or select a number (1, 2, or 3)."
            state["conversation_stage"] = "confirmation"
            self.conversation_state["stage"] = "confirmation"
        
        state["messages"].append({"role": "assistant", "content": response})
        return state
    
    def _completion_node(self, state: AgentState) -> AgentState:
        """Final completion message"""
        response = "Your return has been successfully processed! Is there anything else I can help you with today?"
        state["messages"].append({"role": "assistant", "content": response})
        state["conversation_stage"] = "completion"
        # Don't reset state - allow conversation to continue if user wants
        return state
    
    def _extract_customer_id(self, message: str) -> Optional[str]:
        """Extract customer ID from user message"""
        # Simple extraction - look for patterns that might be customer IDs
        words = message.split()
        for word in words:
            if len(word) > 10 and word.isalnum():  # Customer IDs are typically long alphanumeric strings
                return word
        return None
    
    def _extract_order_id(self, message: str, customer_id: str) -> Optional[str]:
        """Extract order ID from user message"""
        # Check if user provided a number (1, 2, 3)
        if message.strip() in ["1", "2", "3"]:
            orders = self._get_customer_orders(customer_id, limit=3)
            if orders and int(message.strip()) <= len(orders):
                return orders[int(message.strip()) - 1]["order_id"]
        
        # Look for order ID pattern
        words = message.split()
        for word in words:
            if len(word) > 20 and word.isalnum():  # Order IDs are typically long alphanumeric strings
                return word
        
        return None
    
    def _customer_exists(self, customer_id: str) -> bool:
        """Check if customer exists in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM customers WHERE customer_id = ?", (customer_id,))
            count = cursor.fetchone()[0]
            conn.close()
            return count > 0
        except Exception as e:
            print(f"Error checking customer: {e}")
            return False
    
    def _get_customer_orders(self, customer_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get customer's recent orders"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = """
            SELECT o.order_id, o.order_purchase_timestamp, o.order_status,
                   SUM(oi.price + oi.freight_value) as total_amount
            FROM orders o
            LEFT JOIN order_items oi ON o.order_id = oi.order_id
            WHERE o.customer_id = ?
            GROUP BY o.order_id
            ORDER BY o.order_purchase_timestamp DESC
            LIMIT ?
            """
            
            cursor.execute(query, (customer_id, limit))
            rows = cursor.fetchall()
            
            orders = []
            for row in rows:
                orders.append({
                    "order_id": row[0],
                    "order_purchase_timestamp": row[1],
                    "order_status": row[2],
                    "total_amount": round(row[3], 2) if row[3] else 0
                })
            
            conn.close()
            return orders
        except Exception as e:
            print(f"Error getting customer orders: {e}")
            return []
    
    def _update_order_status(self, order_id: str, new_status: str) -> bool:
        """Update order status in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE orders SET order_status = ? WHERE order_id = ?", (new_status, order_id))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating order status: {e}")
            return False
    
    def _get_order_payment_info(self, order_id: str) -> Dict[str, Any]:
        """Get payment information for an order"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT payment_type, payment_value, payment_installments
                FROM order_payments
                WHERE order_id = ?
                LIMIT 1
            """, (order_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    "payment_type": row[0],
                    "payment_value": row[1],
                    "payment_installments": row[2]
                }
            return {}
        except Exception as e:
            print(f"Error getting payment info: {e}")
            return {}
    
    def _get_order_summary(self, order_id: str) -> Dict[str, Any]:
        """Get order summary information"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get order info
            cursor.execute("""
                SELECT order_id, order_status, order_purchase_timestamp
                FROM orders
                WHERE order_id = ?
            """, (order_id,))
            
            order_row = cursor.fetchone()
            
            # Get item count
            cursor.execute("""
                SELECT COUNT(*) FROM order_items WHERE order_id = ?
            """, (order_id,))
            
            item_count = cursor.fetchone()[0]
            conn.close()
            
            if order_row:
                return {
                    "order_id": order_row[0],
                    "order_status": order_row[1],
                    "order_purchase_timestamp": order_row[2],
                    "item_count": item_count
                }
            return {}
        except Exception as e:
            print(f"Error getting order summary: {e}")
            return {}
