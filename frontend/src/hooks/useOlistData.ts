import { useState, useEffect } from 'react';
import axios from 'axios';
import {
  mockMetrics,
  mockOrdersByMonth,
  mockTopCategories,
  mockRevenueByState,
} from '../utils/mockData';

export interface DashboardMetrics {
  totalOrders: number;
  totalRevenue: number;
  averageOrderValue: number;
  totalCustomers: number;
  totalProducts: number;
  averageRating: number;
}

export interface OrdersByMonth {
  month: string;
  orders: number;
  revenue: number;
}

export interface TopCategory {
  category: string;
  orders: number;
  revenue: number;
}

export interface RevenueByState {
  state: string;
  revenue: number;
  orders: number;
}

const USE_MOCK_DATA = true;

export const useOlistData = () => {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [ordersByMonth, setOrdersByMonth] = useState<OrdersByMonth[]>([]);
  const [topCategories, setTopCategories] = useState<TopCategory[]>([]);
  const [revenueByState, setRevenueByState] = useState<RevenueByState[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        if (USE_MOCK_DATA) {
          await new Promise((resolve) => setTimeout(resolve, 300));
          
          setMetrics(mockMetrics);
          setOrdersByMonth(mockOrdersByMonth);
          setTopCategories(mockTopCategories);
          setRevenueByState(mockRevenueByState);
          setError(null);
        } else {
          const [metricsRes, ordersRes, categoriesRes, statesRes] = await Promise.all([
            axios.get('/api/dashboard/metrics'),
            axios.get('/api/dashboard/orders-by-month'),
            axios.get('/api/dashboard/top-categories'),
            axios.get('/api/dashboard/revenue-by-state'),
          ]);

          setMetrics(metricsRes.data);
          setOrdersByMonth(ordersRes.data);
          setTopCategories(categoriesRes.data);
          setRevenueByState(statesRes.data);
          setError(null);
        }
      } catch (err: any) {
        console.error('Error fetching dashboard data:', err);
        setError(err.message || 'Failed to load dashboard data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  return {
    metrics,
    ordersByMonth,
    topCategories,
    revenueByState,
    loading,
    error,
  };
};

