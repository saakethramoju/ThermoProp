"""Lightweight API smoke test that does not require optional thermo backends."""

import thermoprop as tp


print("species", len(tp.list_species()))
print("fluid species", len(tp.supported_species("Fluid")))
print("materials", len(tp.list_materials()))
print("first species", tp.list_species()[:5])
print("first materials", tp.list_materials()[:5])
