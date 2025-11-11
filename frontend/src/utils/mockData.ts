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
  { month: '2017-01', orders: 802, revenue: 123852.42 },
  { month: '2017-02', orders: 1754, revenue: 281284.89 },
  { month: '2017-03', orders: 2371, revenue: 389471.21 },
  { month: '2017-04', orders: 2638, revenue: 419341.87 },
  { month: '2017-05', orders: 3692, revenue: 589012.34 },
  { month: '2017-06', orders: 3727, revenue: 593847.23 },
  { month: '2017-07', orders: 4023, revenue: 642789.45 },
  { month: '2017-08', orders: 4339, revenue: 696234.21 },
  { month: '2017-09', orders: 4564, revenue: 729456.78 },
  { month: '2017-10', orders: 4809, revenue: 767893.34 },
  { month: '2017-11', orders: 7543, revenue: 1203478.56 },
  { month: '2017-12', orders: 5989, revenue: 955234.89 },
  { month: '2018-01', orders: 7271, revenue: 1159823.45 },
  { month: '2018-02', orders: 6739, revenue: 1074567.23 },
  { month: '2018-03', orders: 7551, revenue: 1204982.34 },
  { month: '2018-04', orders: 6923, revenue: 1105234.56 },
  { month: '2018-05', orders: 7287, revenue: 1163478.90 },
  { month: '2018-06', orders: 6449, revenue: 1029234.67 },
  { month: '2018-07', orders: 6729, revenue: 1073234.89 },
  { month: '2018-08', orders: 6554, revenue: 1045678.34 },
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
