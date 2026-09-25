from typing import Any, Callable


def validate_protocol(protocol: Callable[..., Any], input_types: type | tuple[type, ...], output_types: type | tuple[type, ...]) -> None:
    """Validate a protocol is using the specified input and output types."""
    from inspect import signature

    sig = signature(protocol)
    parameters = list(sig.parameters.values())[1:]  # skip self
    for param in parameters:
        if not isinstance(param.annotation, type) or not issubclass(param.annotation, input_types):
            raise TypeError(f"Protocol parameter {param.name} must be an Input or State subclass, got {param.annotation}")

    if sig.return_annotation is sig.empty or not issubclass(sig.return_annotation, output_types):
        raise TypeError(f"Protocol return type must be a State subclass or a tuple of State, got {sig.return_annotation}")


def invoke_protocol(protocol: Callable[..., Any], args: dict[type, Any]) -> Any:
    """Call a protocol with the given arguments, validating the signature."""
    from inspect import signature

    sig = signature(protocol)
    parameters = list(sig.parameters.values())[1:]  # skip self
    call_args = {}
    for param in parameters:
        if param.annotation not in args:
            raise ValueError(f"Missing argument for protocol parameter {param.name} of type {param.annotation}")
        call_args[param.name] = args[param.annotation]

    return protocol(**call_args)
