"""
Reference-state normalization utilities shared by ThermoProp wrappers.

ThermoProp wrappers may use different backend reference states for enthalpy,
internal energy, entropy, Gibbs energy, and Helmholtz/free energy. This module
normalizes user-provided set_reference strings into canonical wrapper names.

Accepted targets
----------------

The public reference targets are:

    None
    "Fluid"
    "IdealGas"
    "Propellant"
    "CombustionGas"

Common aliases such as "coolprop", "pyromat", "rocketprops", and "cea" are
mapped to their corresponding ThermoProp wrapper names.

Behavior
--------

If the requested reference target is the same as the current wrapper,
normalize_reference_target returns None, meaning no offset is needed.

The actual reference offset calculation is performed inside each wrapper. This
module only validates and normalizes the target name.
"""
def normalize_reference_target(
    value: str | None,
    self_name: str,
) -> str | None:

    if value is None:
        return None

    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")

    if key in {"", "none", "raw", "default"}:
        return None

    aliases = {
        "fluid": "Fluid",
        "coolprop": "Fluid",
        "realfluid": "Fluid",
        "real_fluid": "Fluid",
        "idealgas": "IdealGas",
        "ideal_gas": "IdealGas",
        "ideal": "IdealGas",
        "pyromat": "IdealGas",
        "propellant": "Propellant",
        "rocketprops": "Propellant",
        "combustiongas": "CombustionGas",
        "combustion_gas": "CombustionGas",
        "cea": "CombustionGas",
    }

    if key not in aliases:
        raise ValueError(
            "set_reference must be one of None, 'Fluid', "
            "'IdealGas', 'Propellant', or 'CombustionGas'."
        )

    target = aliases[key]

    if target == self_name:
        return None

    return target