// Mock data simplificado

export const mockMetrics = {
  totalOrders: 99441,
  totalRevenue: 15422461.77,
  averageOrderValue: 159.86,
  totalCustomers: 96096,
  totalProducts: 32951,
  averageRating: 4.09,
};

export const mockOrdersByMonth = [
  { month: '2025-01', orders: 802, revenue: 123852.42 },
  { month: '2025-02', orders: 1754, revenue: 281284.89 },
  { month: '2025-03', orders: 2371, revenue: 389471.21 },
  { month: '2025-04', orders: 2638, revenue: 419341.87 },
  { month: '2025-05', orders: 3692, revenue: 589012.34 },
  { month: '2025-06', orders: 3727, revenue: 593847.23 },
  { month: '2025-07', orders: 4023, revenue: 642789.45 },
  { month: '2025-08', orders: 4339, revenue: 696234.21 },
  { month: '2025-09', orders: 4564, revenue: 729456.78 },
  { month: '2025-10', orders: 4809, revenue: 767893.34 },
];

export const mockTopCategories = [
  { category: 'cama_mesa_banho', orders: 11115, revenue: 1569234.56 },
  { category: 'beleza_saude', orders: 9670, revenue: 1456789.23 },
  { category: 'esporte_lazer', orders: 8640, revenue: 1398234.78 },
  { category: 'moveis_decoracao', orders: 8334, revenue: 2145678.90 },
  { category: 'informatica_acessorios', orders: 7827, revenue: 1876543.21 },
  { category: 'utilidades_domesticas', orders: 6964, revenue: 987654.32 },
  { category: 'relogios_presentes', orders: 5991, revenue: 1234567.89 },
  { category: 'telefonia', orders: 4545, revenue: 1567890.12 },
];

export const mockRevenueByState = [
  { state: 'SP', orders: 41746, revenue: 6643234.56 },
  { state: 'RJ', orders: 12852, revenue: 2045678.90 },
  { state: 'MG', orders: 11635, revenue: 1856789.23 },
  { state: 'RS', orders: 5466, revenue: 872345.67 },
  { state: 'PR', orders: 5045, revenue: 805432.12 },
  { state: 'SC', orders: 3637, revenue: 581234.56 },
  { state: 'BA', orders: 3380, revenue: 540123.45 },
  { state: 'DF', orders: 2140, revenue: 341234.56 },
  { state: 'ES', orders: 2033, revenue: 325678.90 },
  { state: 'GO', orders: 2020, revenue: 323456.78 },
];

export const mockPaymentAnalytics = [
  { payment_type: 'credit_card', total_orders: 76795, total_revenue: 12405234.56, avg_installments: 2.8 },
  { payment_type: 'boleto', total_orders: 19784, total_revenue: 2984567.89, avg_installments: 1.0 },
  { payment_type: 'voucher', total_orders: 5775, total_revenue: 390234.56, avg_installments: 1.0 },
  { payment_type: 'debit_card', total_orders: 1529, total_revenue: 47872.95, avg_installments: 1.0 },
];

export const mockDeliveryMetrics = {
  avg_delivery_time_days: 12.6,
  on_time_delivery_rate: 0.919,
  delayed_orders: 7826,
  total_delivered: 96478,
};
