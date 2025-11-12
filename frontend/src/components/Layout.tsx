import { ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { BarChart3 } from 'lucide-react';

interface LayoutProps {
  children: ReactNode;
}

export const Layout = ({ children }: LayoutProps) => {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Dashboard', icon: BarChart3 },
  ];

  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Sidebar */}
      <aside className="w-48 sm:w-56 bg-white border-r border-gray-200 shadow-sm flex flex-col">
        <div className="p-4 border-b border-gray-200">
          <h1 className="text-lg sm:text-xl font-bold text-gray-900 flex items-center gap-2">
            🛍️ BIX Analytics
          </h1>
          <p className="text-xs sm:text-sm text-gray-600 mt-1">
            Dashboard E-commerce
          </p>
        </div>

        <nav className="flex-1 p-3">
          <ul className="space-y-2">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              
              return (
                <li key={item.path}>
                  <Link
                    to={item.path}
                    className={`flex items-center gap-2 sm:gap-3 px-3 sm:px-4 py-2 sm:py-3 rounded-lg transition-all text-sm sm:text-base ${
                      isActive
                        ? 'bg-blue-50 text-blue-600 font-medium shadow-sm'
                        : 'text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    <Icon className="w-4 h-4 sm:w-5 sm:h-5" />
                    <span>{item.label}</span>
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>

        <div className="p-4 border-t border-gray-200">
          <div className="text-xs text-gray-500 text-center">
            Powered by React & Recharts
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-hidden">
        {children}
      </main>
    </div>
  );
};

