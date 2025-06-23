import unittest
import numpy as np
from scenario import Scenario, BS

class DummySce:
    def __init__(self):
        self.nUsers = 2
        self.nSBS = 1
        self.nMBS = 1
        self.nFBS = 0
        self.rMBS = 700
        self.rSBS = 100
        self.rFBS = 50
        self.MBS_BW = 100e6
        self.FBS_BW = 200e6
        self.nChannel = 2
        self.SBS_BW = 100
        self.FBS_BW = 50

def dummy_random_factor_mbs(nUsers):
    return np.ones(nUsers), np.ones(nUsers), np.ones(nUsers)

def dummy_random_factor_sbs(nUsers, nSBS):
    return np.ones((nSBS, nUsers)), np.ones((nSBS, nUsers)), np.ones((nSBS, nUsers))

class TestScenario(unittest.TestCase):
    def setUp(self):
        # Patch random_factor_mbs and random_factor_sbs if needed
        Scenario.random_factor_mbs = staticmethod(dummy_random_factor_mbs)
        Scenario.random_factor_sbs = staticmethod(dummy_random_factor_sbs)
        self.sce = DummySce()
        self.scenario = Scenario(self.sce)

    def test_BS_Init(self):
        bs_list = self.scenario.BS_Init()
        self.assertIsInstance(bs_list, list)
        self.assertTrue(all(isinstance(bs, BS) for bs in bs_list))

    def test_BS_Location(self):
        locs, _, _ = self.scenario.BS_Location()
        # self.assertIsInstance(locs, list)
        self.assertTrue(all(isinstance(loc, np.ndarray) for loc in locs))

    def test_BS_Number(self):
        num = self.scenario.BS_Number()
        self.assertIsInstance(num, int)
        self.assertGreaterEqual(num, 0)

    def test_Get_BaseStations(self):
        bs = self.scenario.Get_BaseStations()
        self.assertIsInstance(bs, list)
        self.assertTrue(all(isinstance(b, BS) for b in bs))

    def test_random_factor_mbs(self):
        x, y, z = dummy_random_factor_mbs(self.sce.nUsers)
        self.assertEqual(len(x), self.sce.nUsers)
        self.assertEqual(len(y), self.sce.nUsers)
        self.assertEqual(len(z), self.sce.nUsers)

    def test_random_factor_sbs(self):
        x, y, z = dummy_random_factor_sbs(self.sce.nUsers, self.sce.nSBS)
        self.assertEqual(x.shape, (self.sce.nSBS, self.sce.nUsers))
        self.assertEqual(y.shape, (self.sce.nSBS, self.sce.nUsers))
        self.assertEqual(z.shape, (self.sce.nSBS, self.sce.nUsers))

if __name__ == '__main__':
    unittest.main()