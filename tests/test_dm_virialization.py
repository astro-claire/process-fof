"""
This test file is currently broken: FIX ME!! 
"""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))

from scripts.dm_virialization import chunked_calc_dm_boundedness
import config.configuration as config

#Set units and parameters
constants = config.load_constants()
UnitMass_in_g = constants['UnitMass_in_g']     
UnitVelocity_in_cm_per_s = constants['UnitVelocity_in_cm_per_s']
UnitLength_in_cm = constants['UnitLength_in_cm']
GRAVITY_cgs = constants['GRAVITY_cgs']

class TestChunkedCalcDMBoundedness(unittest.TestCase):

    def setUp(self):
        # Set up any necessary data for the tests
        self.energyStars = 1.0
        self.starPos_inGroup = np.array([[1, 2, 3], [4, 5, 6]])
        self.starMass_inGroup = np.array([1.0, 2.0])/UnitMass_in_g
        self.groupVelocity = np.array([0.1, 0.2, 0.3])/UnitVelocity_in_cm_per_s
        self.boxSize = 10000.0
        self.boxSizeVel = 0.0
        self.pDM = np.array([[1, 2, 3], [4, 5, 6]])
        self.vDM = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])/UnitVelocity_in_cm_per_s
        self.atime = 1.0
        self.massDMParticle = 1.0/UnitMass_in_g

    @patch('dm_virialization.chunked_potential_energy_same_mass')
    @patch('dm_virialization.chunked_potential_energy_between_groups')
    @patch('dm_virialization.dx_wrap')
    def test_chunked_calc_dm_boundedness(self, mock_dx_wrap, mock_chunked_potential_energy_between_groups, mock_chunked_potential_energy_same_mass):
        # Mock the return values of the dependencies
        mock_dx_wrap.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_chunked_potential_energy_same_mass.return_value = -10.0
        mock_chunked_potential_energy_between_groups.return_value = -5.0 

        # Call the function with the test data
        boundedness, totEnergy, kineticEnergyDM, potentialEnergyStarsDM, massDM = chunked_calc_dm_boundedness(
            self.energyStars, self.starPos_inGroup, self.starMass_inGroup, self.groupVelocity,
            self.boxSize, self.boxSizeVel, self.pDM, self.vDM, self.atime, self.massDMParticle
        )

        # Assertions to verify the expected outcomes
        print(totEnergy)
        self.assertEqual(boundedness, 1)
        self.assertAlmostEqual(totEnergy, -14.0)
        self.assertAlmostEqual(kineticEnergyDM, 0.035)
        self.assertAlmostEqual(potentialEnergyStarsDM, -15.0)
        self.assertEqual(massDM, 2.0)

if __name__ == '__main__':
    unittest.main()