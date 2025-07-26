import unittest

from absl.testing import absltest

import kg

@kg.rule(name="Do nothing.", inputs=[], outputs=[])
def foo(x):
    pass

@kg.rule(name="Do it.", inputs=[], outputs=[])
def foo1(x):
    pass

@kg.rule(name="add 1", inputs=["x"], outputs=["x"])
def add1(n):
    n.outputs["x"] = n.inputs["x"] + 1

@kg.rule(name="x equals 2", inputs=[], outputs=["x"])
def xequals2(n):
    n.outputs["x"] = 2


class BuilderTest(absltest.TestCase):
    def test_add_relations(self):
        b = kg.Builder()
        r1 = ('s', 'p', 'o')
        r2 = ('s1', 'p', 'o')
        r3 = ('s', 'p1', 'o')
        r4 = ('s', 'p', 'o1')
        b.add_relations([r1, r1])
        self.assertEqual(b.relations, [r1])
        b.add_relations([r2, r4, r3, r1])
        self.assertEqual(b.relations, [r1, r2, r4, r3])


class CompilerTest(absltest.TestCase):
    def test_missing_rule(self):
        b = kg.Builder()
        b.add_rules("ncard.kg_test", ["missing rule"])
        with self.assertRaises(KeyError):
            kg.link(b)

    def test_found_rules(self):
        b = kg.Builder()
        b.add_rules("ncard.kg_test", ["Do nothing.", "add 1"])
        lkg = kg.link(b)
        lkg.nodes["add 1"]
        lkg.nodes["Do nothing."]


class ExecutiveTest(absltest.TestCase):
    def test_single_node(self):
        b = kg.Builder()
        b.add_rules("ncard.kg_test", ["add 1"])
        lkg = kg.link(b)
        v = kg.execute(lkg, {"x": 1}, {"y": ("add 1", "x")})
        self.assertDictEqual(v, {"y": 2})

    def test_dependencies(self):
        b = kg.Builder()
        b.add_rules("ncard.kg_test", ["add 1", "x equals 2"])
        lkg = kg.link(b)
        v = kg.execute(lkg, {}, {"y": ("add 1", "x")})
        self.assertDictEqual(v, {"y": 3})


if __name__ == "__main__":
    absltest.main()
