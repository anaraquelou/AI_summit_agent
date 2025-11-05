import { useState } from 'react';
import axios from 'axios';
import { ChatWindow } from './components/ChatWindow';
import { ChatInput } from './components/ChatInput';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: 'Olá! Sou o João, seu assistente especializado em gestão de pedidos e devoluções da BIX E-commerce. Posso ajudá-lo(a) a verificar informações de pedidos, consultar nossa política de devolução e processar devoluções. Como posso ajudá-lo(a) hoje?',
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
            // Filter out intermediate system messages
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

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <header className="border-b border-gray-200 bg-white/80 backdrop-blur-sm px-4 py-4 shadow-sm">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-2xl font-semibold text-gray-900">
            🛍️ Analista de Pedidos - BIX E-commerce
          </h1>
          <p className="text-sm text-gray-600">
            Verifique pedidos, consulte políticas e processe devoluções
          </p>
        </div>
      </header>

      <ChatWindow messages={messages} isLoading={isLoading} />

      <ChatInput
        value={input}
        onChange={setInput}
        onSend={handleSend}
        disabled={isLoading}
      />
    </div>
  );
}

export default App;

