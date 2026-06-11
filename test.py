from src.thermoprop import Material, material_aliases

grcop = Material("grcop42", temperature=300)

print(grcop.yield_strength)