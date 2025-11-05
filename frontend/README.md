# Return Policy Chat Frontend

A simplified React frontend for the Return Policy Chat Agent, built with Vite, TypeScript, and Tailwind CSS.

## Features

- **Simple Chat Interface**: Clean, modern chat UI
- **Real-time Messaging**: Connect to the FastAPI backend
- **Conversation Memory**: Maintains chat history via thread_id
- **TypeScript**: Type-safe React components
- **Tailwind CSS**: Minimal styling with Tailwind

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Build

```bash
npm run build
```

## Features Removed for Simplicity

- Removed shadcn/ui components (using basic Tailwind instead)
- Removed React Router (single page app)
- Removed React Query (using axios directly)
- Removed complex UI components (accordion, tabs, etc.)
- Simplified styling (basic Tailwind utility classes)

