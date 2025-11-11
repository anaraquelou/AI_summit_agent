import { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  ShoppingCart, 
  DollarSign, 
  Users, 
  Package, 
  Star,
  TrendingUp,
  Calendar,
  MapPin
} from 'lucide-react';
import { MetricCard } from '../components/MetricCard';
import { FloatingChat } from '../components/FloatingChat';
import { useOlistData } from '../hooks/useOlistData';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { mockPaymentAnalytics, mockDeliveryMetrics } from '../utils/mockData';

// Set to true to use mock data
const USE_MOCK_DATA = true;

interface PaymentAnalytics {
  payment_type: string;
  total_orders: number;
  total_revenue: number;
  avg_installments: number;
}

interface DeliveryMetrics {
  avg_delivery_time_days: number;
  on_time_delivery_rate: number;
  delayed_orders: number;
  total_delivered: number;
}

export const Dashboard = () => {
  const {
    metrics,
    ordersByMonth,
    topCategories,
    revenueByState,
    loading,
    error,
  } = useOlistData();

  const [paymentAnalytics, setPaymentAnalytics] = useState<PaymentAnalytics[]>([]);
  const [deliveryMetrics, setDeliveryMetrics] = useState<DeliveryMetrics | null>(null);
  const [analyticsLoading, setAnalyticsLoading] = useState(true);

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        if (USE_MOCK_DATA) {
          await new Promise((resolve) => setTimeout(resolve, 400));
          setPaymentAnalytics(mockPaymentAnalytics);
          setDeliveryMetrics(mockDeliveryMetrics);
        } else {
          const [paymentsRes, deliveryRes] = await Promise.all([
            axios.get('/api/analytics/payments'),
            axios.get('/api/analytics/delivery-metrics'),
          ]);
          setPaymentAnalytics(paymentsRes.data);
          setDeliveryMetrics(deliveryRes.data);
        }
      } catch (error) {
        console.error('Error fetching analytics:', error);
      } finally {
        setAnalyticsLoading(false);
      }
    };

    fetchAnalytics();
  }, []);

  if (loading || analyticsLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Carregando dados...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center p-8 bg-red-50 rounded-lg border border-red-200">
          <p className="text-red-600 font-medium">Erro ao carregar dados</p>
          <p className="text-red-500 text-sm mt-2">{error}</p>
        </div>
      </div>
    );
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('pt-BR', {
      style: 'currency',
      currency: 'BRL',
    }).format(value);
  };

  const formatNumber = (value: number) => {
    return new Intl.NumberFormat('pt-BR').format(value);
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="p-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">BIX E-commerce</h1>
          <p className="text-gray-600">Visão geral do desempenho da plataforma</p>
        </div>

        {/* Main Metrics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          <MetricCard
            title="Total de Pedidos"
            value={formatNumber(metrics?.totalOrders || 0)}
            icon={ShoppingCart}
            iconColor="bg-blue-500"
            change="+12.5%"
            changeType="positive"
          />
          <MetricCard
            title="Receita Total"
            value={formatCurrency(metrics?.totalRevenue || 0)}
            icon={DollarSign}
            iconColor="bg-green-500"
            change="+8.2%"
            changeType="positive"
          />
          <MetricCard
            title="Ticket Médio"
            value={formatCurrency(metrics?.averageOrderValue || 0)}
            icon={TrendingUp}
            iconColor="bg-purple-500"
            change="+2.1%"
            changeType="positive"
          />
          <MetricCard
            title="Total de Clientes"
            value={formatNumber(metrics?.totalCustomers || 0)}
            icon={Users}
            iconColor="bg-orange-500"
          />
          <MetricCard
            title="Total de Produtos"
            value={formatNumber(metrics?.totalProducts || 0)}
            icon={Package}
            iconColor="bg-pink-500"
          />
          <MetricCard
            title="Avaliação Média"
            value={(metrics?.averageRating || 0).toFixed(1)}
            icon={Star}
            iconColor="bg-yellow-500"
          />
        </div>

        {/* Delivery Metrics Cards */}
        {deliveryMetrics && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm font-medium text-gray-600">Tempo Médio de Entrega</p>
                <Calendar className="w-5 h-5 text-blue-600" />
              </div>
              <p className="text-3xl font-bold text-gray-900">
                {deliveryMetrics.avg_delivery_time_days.toFixed(1)} dias
              </p>
            </div>

            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm font-medium text-gray-600">Taxa de Entrega no Prazo</p>
                <TrendingUp className="w-5 h-5 text-green-600" />
              </div>
              <p className="text-3xl font-bold text-gray-900">
                {(deliveryMetrics.on_time_delivery_rate * 100).toFixed(1)}%
              </p>
            </div>

            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm font-medium text-gray-600">Pedidos Atrasados</p>
                <Package className="w-5 h-5 text-red-600" />
              </div>
              <p className="text-3xl font-bold text-gray-900">
                {formatNumber(deliveryMetrics.delayed_orders)}
              </p>
            </div>

            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm font-medium text-gray-600">Total Entregue</p>
                <MapPin className="w-5 h-5 text-purple-600" />
              </div>
              <p className="text-3xl font-bold text-gray-900">
                {formatNumber(deliveryMetrics.total_delivered)}
              </p>
            </div>
          </div>
        )}

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Orders by Month */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Pedidos por Mês</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={ordersByMonth}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="orders" 
                  stroke="#3b82f6" 
                  strokeWidth={2}
                  name="Pedidos"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Revenue by Month */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Receita por Mês</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={ordersByMonth}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip formatter={(value: number) => formatCurrency(value)} />
                <Legend />
                <Bar 
                  dataKey="revenue" 
                  fill="#10b981"
                  name="Receita"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Top Categories */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Top 8 Categorias</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={topCategories.slice(0, 8)} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis 
                  dataKey="category" 
                  type="category" 
                  width={120}
                  tick={{ fontSize: 12 }}
                />
                <Tooltip />
                <Bar 
                  dataKey="orders" 
                  fill="#8b5cf6"
                  name="Pedidos"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Payment Methods Analysis */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Análise por Forma de Pagamento</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={paymentAnalytics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="payment_type" />
                <YAxis yAxisId="left" orientation="left" stroke="#3b82f6" />
                <YAxis yAxisId="right" orientation="right" stroke="#10b981" />
                <Tooltip 
                  formatter={(value: number, name: string) => {
                    if (name === 'Receita') return formatCurrency(value);
                    return formatNumber(value);
                  }}
                />
                <Legend />
                <Bar 
                  yAxisId="left"
                  dataKey="total_orders" 
                  fill="#3b82f6"
                  name="Pedidos"
                />
                <Bar 
                  yAxisId="right"
                  dataKey="total_revenue" 
                  fill="#10b981"
                  name="Receita"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Top States by Revenue */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Top 10 Estados por Receita</h3>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={revenueByState.slice(0, 10)}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="state" />
              <YAxis yAxisId="left" orientation="left" stroke="#3b82f6" />
              <YAxis yAxisId="right" orientation="right" stroke="#10b981" />
              <Tooltip 
                formatter={(value: number, name: string) => {
                  if (name === 'Receita') return formatCurrency(value);
                  return formatNumber(value);
                }}
              />
              <Legend />
              <Bar 
                yAxisId="left"
                dataKey="orders" 
                fill="#3b82f6"
                name="Pedidos"
              />
              <Bar 
                yAxisId="right"
                dataKey="revenue" 
                fill="#10b981"
                name="Receita"
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Floating Chat */}
      <FloatingChat />
    </div>
  );
};
