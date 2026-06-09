from src.thermoprop import Propellant, CombustionGas
from CEADatabase import CEA

p = Propellant("rp-1", temperature=100)

print(p.specific_heat_cp)

gas = CombustionGas(
    {
        "H2O": 0.35,
        "CO2": 0.25,
        "CO": 0.10,
        "H2": 0.05,
        "O2": 0.02,
        "N2": 0.23,
    },
    #basis="mole",
    temperature=3200.0,
    pressure=2.0e6,
)

print(gas.molar_mass)
print(gas.gas_constant)
print(gas.density)
print(gas.specific_heat_cp)
print(gas.specific_heat_cv)
print(gas.specific_heat_ratio)
print(gas.enthalpy)
print(gas.entropy)
print(gas.dynamic_viscosity)
print(gas.thermal_conductivity)
print(gas.prandtl)
print(gas)

print(CEA.describe("RP-1"))