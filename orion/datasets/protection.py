"""Map published crop-protection labels onto ProtectionEvent.kind."""

from __future__ import annotations

_KIND_ALIASES = {
    "fungicide": "fungicide",
    "fungicides": "fungicide",
    "herbicide": "herbicide",
    "herbicides": "herbicide",
    "insecticide": "insecticide",
    "insecticides": "insecticide",
    "pgr": "pgr",
    "growth regulator": "pgr",
    "growth_regulator": "pgr",
    "growthregulator": "pgr",
    "standard": "standard",
    "none": "none",
    "untreated": "none",
}

_PRODUCT_KIND = {
    "capalo": "fungicide",
    "opus": "fungicide",
    "prosaro": "fungicide",
    "atlantis": "herbicide",
}

_MEASURE_KIND = {
    "pesticide": "fungicide",
    "protection": "fungicide",
    "plant protection": "fungicide",
    "sowing": None,
    "fertilisation": None,
    "fertilization": None,
    "fertiliser": None,
    "irrigation": None,
}


def protection_kind(*, type_name: str = "", product: str = "", measure: str = "") -> str:
    """Infer protection kind from a type column, product, or management measure."""
    for raw in (type_name, measure):
        alias = _KIND_ALIASES.get(raw.strip().lower())
        if alias:
            return alias
    product_kind = _PRODUCT_KIND.get(product.strip().lower())
    if product_kind:
        return product_kind
    measure_kind = _MEASURE_KIND.get(measure.strip().lower())
    if measure_kind:
        return measure_kind
    if type_name or product or measure:
        return "standard"
    return "none"
