# Third-Party Licenses

ThermoProp depends on or incorporates data from the following third-party projects.

Users are responsible for complying with the licenses of these projects when using, modifying, or redistributing ThermoProp.

## CoolProp

CoolProp is released under the MIT License.

Copyright (c) 2012-2018 Ian H. Bell and other CoolProp developers

Project:
https://coolprop.org/

License:
https://github.com/CoolProp/CoolProp/blob/master/LICENSE

---

## PYroMat

PYroMat is released under the GNU General Public License version 3 (GPL-3.0).

Copyright (c) 2015-2024 Christopher R. Martin

Project:
https://chmarti1.github.io/PYroMat/

License:
https://github.com/chmarti1/PYroMat/blob/master/LICENSE.txt

---

## RocketProps

RocketProps is released under the GNU General Public License version 3 (GPL-3.0).

Copyright (c) Charlie Taylor

Project:
https://rocketprops.readthedocs.io/

License:
https://github.com/sonofeft/RocketProps/blob/master/LICENSE.txt

---

## NumPy

NumPy is released under the BSD 3-Clause License.

Copyright (c) 2005-2025, NumPy Developers

Project:
https://numpy.org/

License:
https://github.com/numpy/numpy/blob/main/LICENSE.txt

---

## SciPy

SciPy is released under the BSD 3-Clause License.

Copyright (c) 2001-2025, SciPy Developers

Project:
https://scipy.org/

License:
https://github.com/scipy/scipy/blob/main/LICENSE.txt

---

## MatProtLib

MatProtLib is released under the MIT License.

Copyright (c) Tyson Tran

Project:
https://github.com/tysontran/MatProtLib

License:
https://github.com/tysontran/MatProtLib/blob/main/LICENSE

### Attribution

ThermoProp's isotropic material property database was adapted from material property data compiled and distributed through the MatProtLib project.

ThermoProp does not depend on MatProtLib at runtime. Material property data included with ThermoProp were adapted from MatProtLib and integrated directly into ThermoProp's material property database.

The author gratefully acknowledges Tyson Tran and the MatProtLib project for making these material property datasets publicly available.

---

## NASA CEA / CEAM Data

ThermoProp includes thermodynamic and transport-property data adapted from the NASA Chemical Equilibrium with Applications (CEA) program and associated CEAM thermodynamic and transport databases.

NASA CEA was developed by the National Aeronautics and Space Administration (NASA) to perform chemical-equilibrium, thermodynamic, and transport-property calculations for propulsion, combustion, and aerospace applications.

Primary reference:

McBride, B. J., Gordon, S., and Reno, M. A.,
*NASA Glenn Coefficients for Calculating Thermodynamic Properties of Individual Species*,
NASA/TP-2002-211556,
National Aeronautics and Space Administration,
2002.

NASA CEA:

McBride, B. J. and Gordon, S.,
*Computer Program for Calculation of Complex Chemical Equilibrium Compositions and Applications (CEA)*,
NASA Reference Publication 1311,
1994.

NASA publications are generally public-domain works of the United States Government.

Project:
https://www.nasa.gov/glenn/

CEA Reference:
https://ntrs.nasa.gov/citations/19950013764

ThermoProp is not affiliated with NASA and does not distribute the original NASA CEA software.

### Attribution

ThermoProp's `CEADatabase` and equilibrium-combustion functionality utilize thermodynamic and transport datasets derived from NASA CEA / CEAM data products.

The author gratefully acknowledges the NASA Glenn Research Center and the CEA development team for making these thermodynamic and transport datasets publicly available.

## Disclaimer

ThermoProp is an independent project and is not affiliated with, endorsed by, or sponsored by the CoolProp, PYroMat, RocketProps, NumPy, SciPy, or MatProtLib projects.

All trademarks, copyrights, and licenses remain the property of their respective owners.
