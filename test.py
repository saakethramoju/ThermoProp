from src.thermoprop import IdealGas, FluidRegistry, Propellant

ig = IdealGas("1-Butene", temperature=1000)
p = Propellant("rp-1", temperature=100)

print(ig.dynamic_viscosity)