###############################
# UNIT TESTS
###############################
import unittest
import pandas as pd


class TestNewMetrics(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.order_data = pd.DataFrame(
            {
                "Date": pd.date_range("2023-01-01", periods=5, freq="D"),
                "Total": [100, 200, 300, 400, 500],
                "Client": ["A", "B", "A", "C", "A"],
                "Référence": ["R1", "R2", "R3", "R4", "R5"],
            }
        )

        self.cart_data = pd.DataFrame(
            {
                "ID commande": ["R1", "R2", "X1", "X2"],
                "Client": ["A", "B", "D", "E"],
                "Total": [50, 60, 70, 80],
                "Date": pd.date_range("2023-01-01", periods=4, freq="D"),
            }
        )

    def test_basic_kpis(self):
        from __main__ import basic_kpis

        kpis = basic_kpis(self.order_data)
        self.assertEqual(kpis["total_orders"], 5)
        self.assertEqual(kpis["total_revenue"], 1500)
        self.assertEqual(kpis["unique_customers"], 3)
        self.assertAlmostEqual(kpis["avg_orders_per_customer"], 5 / 3, places=5)

    def test_cart_abandonment_rate(self):
        from __main__ import compute_cart_abandonment_rate

        rate = compute_cart_abandonment_rate(self.cart_data, self.order_data)
        # total_carts = 4, completed_orders = 2 (R1, R2 match)
        # abandonment = 1 - (2/4) = 0.5 * 100 = 50%
        self.assertEqual(rate, 50.0)

    def test_repeat_purchase_interval(self):
        from __main__ import repeat_purchase_interval

        # Client A: Orders on day 1, day 3, day 5 -> intervals: (2 days, 2 days) avg = 2.0 days
        interval = repeat_purchase_interval(self.order_data)
        self.assertEqual(interval, 2.0)

    def test_churn_rate(self):
        from __main__ import churn_rate

        # last purchase date = day5 for A, day2 for B, day4 for C
        # analysis_date = day6
        # inactivity = 90 days means none have churned
        churn = churn_rate(self.order_data, inactivity_period=90)
        self.assertEqual(churn, 0.0)

    def test_refined_clv(self):
        from __main__ import refined_clv

        # avg_order_value = 300
        # freq: total orders=5, unique_cust=3, avg per cust=5/3≈1.666...
        # CLV = 300 * 1.6667 * 12 ≈ 6000
        clv = refined_clv(self.order_data)
        self.assertAlmostEqual(clv, 6000, delta=1)


if __name__ == "__main__":
    unittest.main()
