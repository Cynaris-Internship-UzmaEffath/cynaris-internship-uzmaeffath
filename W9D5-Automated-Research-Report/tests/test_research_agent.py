import unittest
from research_agent import generate_report


class TestResearchAgent(unittest.TestCase):

    def test_report_generation(self):
        report, scores = generate_report(
            "Artificial Intelligence in Healthcare"
        )

        self.assertTrue(report)
        self.assertIn("Conclusion", report)
        self.assertEqual(sum(scores.values()), 4)


if __name__ == "__main__":
    unittest.main()
