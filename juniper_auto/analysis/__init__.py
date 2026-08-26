"""Analysis-only infrastructure that is not part of the model's forward
path: context-sensitivity probing today, extended by later phases. Nothing
in this package is imported by `juniper_auto.model` or `juniper_auto.training`
-- it consumes their public interfaces (diagnostics, MoELayer) from the
outside."""
