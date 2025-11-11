# BIX Analytics - Frontend

Dashboard React para análise de dados e-commerce com chat assistente integrado.

## 🚀 Instalação

```bash
npm install
```

## 💻 Desenvolvimento

```bash
npm run dev
```

Acesse: `http://localhost:3000`

## 📦 Build

```bash
npm run build
```

## 🎯 Funcionalidades

- Dashboard com métricas e gráficos
- Chat flutuante (minimizar/maximizar)
- Dados mockados (funcionando sem backend)

## 🔧 Tecnologias

- React 18
- TypeScript
- Tailwind CSS
- Recharts
- Vite

## 📝 Estrutura

```
src/
├── components/     # Componentes reutilizáveis
├── pages/          # Dashboard
├── hooks/          # useOlistData
└── utils/          # Mock data
```

## ⚙️ Configuração

Para usar API real ao invés de mock data, altere em:
- `src/hooks/useOlistData.ts`: `USE_MOCK_DATA = false`
- `src/pages/Dashboard.tsx`: `USE_MOCK_DATA = false`

Backend deve estar na porta 8000.
