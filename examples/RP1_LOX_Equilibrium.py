"""Small RP-1/LOX HP-equilibrium smoke example."""

from thermoprop import Equilibrium, Propellant, Reactants


pressure = 300.0 * 6894.757293168361
fuel = Propellant("rp-1", temperature=298.15, pressure=pressure)
oxidizer = Propellant("lox", temperature=90.17, pressure=pressure)

reactants = Reactants(
    fuels=fuel,
    oxidizers=oxidizer,
    mixture_ratio=2.2,
)

eq = Equilibrium(reactants=reactants, mode="hp", pressure=pressure)
print(eq)
