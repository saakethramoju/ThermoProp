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