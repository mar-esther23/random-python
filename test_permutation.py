
import unittest
from permutation import *
from itertools import permutations



class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.seq = range(4)
        self.index = 15

    def test_elements_none(self):
        # make sure if no elements return empty list
        self.assertEqual( index_to_permutation(self.index, ''), [] )
        self.assertEqual( index_to_permutation(self.index, []), [] )

    def test_index(self):
        #make sure if index is out of range it raises an error
        with self.assertRaises(IndexError):
            index_to_permutation(24, self.seq)

    def test_presence(self):
        # make sure the permuted sequence does not lose any elements
        seq_permuted = index_to_permutation(self.index, self.seq)
        seq_permuted.sort()
        self.assertEqual(self.seq, range(4))

    def test_iter_permutations(self):
        # make sure if iterated from 0 to (n!-1) must return the same than itertools.permutetions()
        c = 0
        for i in permutations(self.seq):
            self.assertEqual( list(i), index_to_permutation(c, self.seq) )
            c += 1

if __name__ == '__main__':
    unittest.main()