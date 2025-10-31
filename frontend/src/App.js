import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';

// Helper function to filter out intermediate step messages
const isIntermediateStep = (message) => {
  if (!message || !message.content) return false;
  const content = message.content.trim();
  
  // Patterns that indicate intermediate steps
  const intermediatePatterns = [
    /^Available tables?:/i,
    /^Thought:/i,
    /^Action:/i,
    /^Action Input:/i,
    /^Observation:/i,
    /^Tool:/i,
    /^Tables available:/i,
    /^Query:/i,
    /^Result:/i
  ];
  
  // Check if content matches any intermediate pattern
  if (intermediatePatterns.some(pattern => pattern.test(content))) {
    return true;
  }
  
  // Filter out very short messages that are just metadata (less than 10 chars, no spaces)
  if (content.length < 10 && !content.includes(' ') && !content.includes('.')) {
    return true;
  }
  
  return false;
};

function App() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Olá! Sou o João, seu assistente especializado em gestão de pedidos e devoluções da Polar E-commerce. Posso ajudá-lo(a) a verificar informações de pedidos, consultar nossa política de devolução e processar devoluções. Como posso ajudá-lo(a) hoje?',
      timestamp: new Date().toISOString()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  // Use a consistent thread_id for conversation memory
  const [threadId] = useState(() => `thread_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      role: 'user',
      content: inputMessage,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await axios.post('/chat', {
        message: inputMessage,
        conversation_history: messages,
        thread_id: threadId  // Send thread_id for conversation memory
      });

      // The backend returns the full conversation history
      if (response.data.conversation_history && response.data.conversation_history.length > 0) {
        // Add timestamps to messages that don't have them and filter out intermediate steps and empty messages
        const messagesWithTimestamps = response.data.conversation_history
          .map((msg, idx) => ({
            ...msg,
            timestamp: msg.timestamp || new Date().toISOString()
          }))
          .filter(msg => {
            // Filter out messages with no content or empty content
            if (!msg.content || typeof msg.content !== 'string' || msg.content.trim().length === 0) {
              return false;
            }
            // Filter out intermediate steps
            return !isIntermediateStep(msg);
          });
        setMessages(messagesWithTimestamps);
      } else {
        // Fallback: add assistant message if history is empty
        const assistantMessage = {
          role: 'assistant',
          content: response.data.message || 'Desculpe, não recebi uma resposta.',
          timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, assistantMessage]);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        role: 'assistant',
        content: error.response?.data?.detail 
          ? `Desculpe, ocorreu um erro: ${error.response.data.detail}` 
          : 'Desculpe, encontrei um erro. Por favor, tente novamente.',
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🛍️ Assistente de Devoluções - Polar E-commerce</h1>
        <p>Verifique pedidos, consulte políticas e processe devoluções</p>
      </header>
      
      <div className="chat-container">
        <div className="messages-container">
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.role}`}>
              <div className="message-content">
                <div className="message-text">
                  {message.content.split('\n').map((line, i) => (
                    <span key={i}>
                      {line}
                      {i < message.content.split('\n').length - 1 && <br />}
                    </span>
                  ))}
                </div>
                <div className="message-time">
                  {new Date(message.timestamp).toLocaleTimeString()}
                </div>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="message assistant">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        
        <div className="input-container">
          <div className="input-wrapper">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Digite sua mensagem aqui... (Ex: 'Como funciona a política de devolução?', 'Verificar pedido e481f51...')"
              disabled={isLoading}
              rows={3}
            />
            <button 
              onClick={sendMessage} 
              disabled={!inputMessage.trim() || isLoading}
              className="send-button"
            >
              Enviar
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
