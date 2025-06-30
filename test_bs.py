import unittest
import numpy as np
from scenario import BS

class DummySce:
    def __init__(self):
        self.nChannel = 2
        self.nMBS = 1
        self.nSBS = 1
        self.nFBS = 0
        self.SBS_BW = 200e6
        self.FBS_BW = 50
        self.MBS_BW = 100e6
        self.rMBS = 700
        self.rSBS = 100
        self.rFBS = 50

class TestBS(unittest.TestCase):
    def setUp(self):
        self.sce = DummySce()
        self.bs_mbs = BS(self.sce, 0, "MBS", np.array([0,0]), 700, self.sce.MBS_BW)
        self.bs_sbs = BS(self.sce, 1, "SBS", np.array([1,1]), 100, self.sce.SBS_BW)
        self.bs_fbs = BS(self.sce, 2, "FBS", np.array([2,2]), 50, self.sce.FBS_BW)

    def test_reset(self):
        self.bs_mbs.reset()
        self.assertTrue(np.array_equal(self.bs_mbs.Ch_State, np.zeros(self.sce.nChannel)))

    def test_Get_Location(self):
        loc = self.bs_mbs.Get_Location()
        self.assertTrue(np.array_equal(loc, np.array([0,0])))

    def test_Transmit_Power_dBm(self):
        self.assertEqual(self.bs_mbs.Transmit_Power_dBm(), 46)
        self.assertEqual(self.bs_sbs.Transmit_Power_dBm(), 30)
        self.assertEqual(self.bs_fbs.Transmit_Power_dBm(), 20)
    
    def test_transmit_power_per_channel_dBm(self):
        # Test the transmit power per channel in dB
        tx_power_mbs = self.bs_mbs.tx_power_per_channel_dB()
        tx_power_sbs = self.bs_sbs.tx_power_per_channel_dB()
        print("Transmit Power MBS per channel dB:", tx_power_mbs)
        print("Transmit Power SBS per channel dB:", tx_power_sbs)
        self.assertIsInstance(tx_power_mbs, float)
        self.assertIsInstance(tx_power_sbs, float)

    def test_loss_sbs(self):
        # Patch global x_list_sbs for this test
        import scenario
        scenario.x_list_sbs = np.ones((1,2))
        loss = self.bs_sbs.loss_sbs_db(0, 10)
        print("Loss SBS dB:", loss)
        self.assertIsInstance(loss, float)

    def test_loss_mbs(self):
        # Patch global x_LOS_list, x_NLOS_list, z_list for this test
        import scenario
        scenario.x_LOS_list = np.ones(2)
        scenario.x_NLOS_list = np.ones(2)
        scenario.z_list = np.ones(2)
        loss = self.bs_mbs.loss_mbs_db(0, 10)
        print("Loss MBS dB:", loss)
        self.assertIsInstance(loss, float)

    def test_tx_gain_sbs(self):
        import scenario
        scenario.R_list = np.ones((1,2))
        gain = self.bs_sbs.tx_gain_sbs_db(0)
        print("Transmit Gain SBS dB:", gain)
        self.assertIsInstance(gain, float)

    def test_tx_gain_mbs(self):
        gain = self.bs_mbs.tx_gain_mbs_db(0, 10)
        print("Transmit Gain MBS dB:", gain)
        # self.assertEqual(gain, 20)

    def test_noise_mbs_dB(self):
        noise = self.bs_mbs.Noise_dB
        print("Noise MBS dB:", noise)

    def test_noise_sbs_dB(self):
        noise = self.bs_sbs.Noise_dB
        print("Noise SBS dB:", noise)

    def test_Receive_Power(self):
        # Patch required globals for loss and gain
        import scenario
        scenario.x_LOS_list = np.ones(2)
        scenario.x_NLOS_list = np.ones(2)
        scenario.z_list = np.ones(2)
        scenario.x_list_sbs = np.ones((1,2))
        scenario.R_list = np.ones((1,2))
        rx_power_mbs = self.bs_mbs.Receive_Power(0, 10)
        rx_power_sbs = self.bs_sbs.Receive_Power(0, 10)
        print("Received Power MBS:", rx_power_mbs)
        print("Received Power SBS:", rx_power_sbs)
        self.assertIsInstance(rx_power_mbs, float)
        self.assertIsInstance(rx_power_sbs, float)

if __name__ == '__main__':
    unittest.main()