import analyze
from absl.testing import absltest

class Test1Card(absltest.TestCase):
    def generic(self, label):
        node = analyze.node(label)
        self.assertTrue(analyze.validate_truth(node))

    def test_1nt_fails(self):
        self.generic("1nt fails")

    def test_1x_outcomes(self):
        self.generic("1x outcomes")
    
    def test_1S_optimal(self):
        self.generic("1S optimal")

    def test_algorithm_exists(self):
        self.generic("algorithm constructed")


if __name__ == "__main__":
    absltest.main()
