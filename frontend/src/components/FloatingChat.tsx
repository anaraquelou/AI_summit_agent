import { useState } from 'react';
import axios from 'axios';
import { MessageCircle, X, Send, Minus, Maximize2 } from 'lucide-react';
import { ChatWindow } from './ChatWindow';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string;
}

export const FloatingChat = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [isMaximized, setIsMaximized] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: 'Olá! Sou o assistente virtual da BIX E-commerce.\n\nPosso ajudar na consulta e análise de dados relacionados a pedidos, devoluções e políticas operacionais.\n\nComo posso ajudar hoje?',
      timestamp: new Date().toISOString(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [threadId] = useState(() => `thread_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await axios.post('/chat', {
        message: input,
        conversation_history: messages.map(({ role, content, timestamp }) => ({
          role,
          content,
          timestamp,
        })),
        thread_id: threadId,
      });

      if (response.data.conversation_history && response.data.conversation_history.length > 0) {
        const messagesWithTimestamps = response.data.conversation_history
          .map((msg: any, idx: number) => ({
            id: `${msg.role}_${idx}_${Date.now()}`,
            role: msg.role as 'user' | 'assistant',
            content: msg.content,
            timestamp: msg.timestamp || new Date().toISOString(),
          }))
          .filter((msg: Message) => {
            if (!msg.content || typeof msg.content !== 'string' || msg.content.trim().length === 0) {
              return false;
            }
            if (msg.role === 'user') {
              return true;
            }
            const content = msg.content.trim();
            const intermediatePatterns = [
              /^Available tables?:/i,
              /^Thought:/i,
              /^Action:/i,
              /^Tool:/i,
            ];
            return !intermediatePatterns.some((pattern) => pattern.test(content));
          });

        setMessages(messagesWithTimestamps);
      } else {
        const assistantMessage: Message = {
          id: Date.now().toString(),
          role: 'assistant',
          content: response.data.message || 'Desculpe, não recebi uma resposta.',
          timestamp: new Date().toISOString(),
        };
        setMessages((prev) => [...prev, assistantMessage]);
      }
    } catch (error: any) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: Date.now().toString(),
        role: 'assistant',
        content:
          error.response?.data?.detail ||
          'Desculpe, encontrei um erro. Por favor, tente novamente.',
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (input.trim() && !isLoading) {
        handleSend();
      }
    }
  };

  return (
    <>
      {/* Chat Button - shown when closed */}
      {!isOpen && (
        <button
          onClick={() => {
            setIsOpen(true);
            setIsMinimized(false);
          }}
          className="fixed bottom-6 right-6 w-16 h-16 bg-blue-600 text-white rounded-full shadow-lg hover:bg-blue-700 transition-all flex items-center justify-center z-50 hover:scale-110"
          aria-label="Abrir chat"
        >
          <MessageCircle className="w-7 h-7" />
        </button>
      )}

      {/* Chat Window - Minimized */}
      {isOpen && isMinimized && (
        <div className="fixed bottom-6 right-6 w-80 bg-white rounded-2xl shadow-2xl z-50 border border-gray-200">
          <div 
            className="flex items-center justify-between p-4 bg-gradient-to-r from-blue-600 to-blue-700 rounded-2xl cursor-pointer hover:from-blue-700 hover:to-blue-800 transition-all"
            onClick={() => setIsMinimized(false)}
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-white rounded-full flex items-center justify-center">
                <MessageCircle className="w-6 h-6 text-blue-600" />
              </div>
              <div>
                <h3 className="font-semibold text-white">Chat - Assistente BIX</h3>
                <p className="text-xs text-blue-100">Clique para expandir</p>
              </div>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setIsOpen(false);
                setIsMinimized(false);
              }}
              className="text-white hover:bg-blue-800 rounded-lg p-2 transition-colors"
              aria-label="Fechar chat"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>
      )}

      {/* Chat Window - Expanded */}
      {isOpen && !isMinimized && (
        <div className={`fixed bottom-6 right-6 bg-white rounded-2xl shadow-2xl flex flex-col z-50 border border-gray-200 transition-all ${
          isMaximized ? 'w-[600px] h-[750px]' : 'w-96 h-[600px]'
        }`}>
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-gradient-to-r from-blue-600 to-blue-700 rounded-t-2xl shrink-0">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-white rounded-full flex items-center justify-center">
                <MessageCircle className="w-6 h-6 text-blue-600" />
              </div>
              <div>
                <h3 className="font-semibold text-white">Chat - Assistente BIX</h3>
              </div>
            </div>
            <div className="flex gap-1">
              <button
                onClick={() => setIsMaximized(!isMaximized)}
                className="text-white hover:bg-blue-800 rounded-lg p-2 transition-colors"
                aria-label={isMaximized ? "Restaurar tamanho" : "Maximizar chat"}
              >
                <Maximize2 className="w-5 h-5" />
              </button>
              <button
                onClick={() => setIsMinimized(true)}
                className="text-white hover:bg-blue-800 rounded-lg p-2 transition-colors"
                aria-label="Minimizar chat"
              >
                <Minus className="w-5 h-5" />
              </button>
              <button
                onClick={() => {
                  setIsOpen(false);
                  setIsMinimized(false);
                  setIsMaximized(false);
                }}
                className="text-white hover:bg-blue-800 rounded-lg p-2 transition-colors"
                aria-label="Fechar chat"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Messages with scroll */}
          <div className="flex-1 overflow-y-auto">
            <ChatWindow messages={messages} isLoading={isLoading} />
          </div>

          {/* Input */}
          <div className="border-t border-gray-200 p-4 bg-gray-50 rounded-b-2xl shrink-0">
            <div className="flex gap-2">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Digite sua mensagem..."
                className="flex-1 min-h-[50px] max-h-[100px] px-3 py-2 rounded-lg border border-gray-300 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed text-sm"
                disabled={isLoading}
                rows={2}
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || isLoading}
                className="h-[50px] w-[50px] shrink-0 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
                aria-label="Enviar mensagem"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

