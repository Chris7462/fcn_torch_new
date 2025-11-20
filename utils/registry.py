"""
Registry System. A minimal registry for mapping string names to classes.
"""


class Registry:
    """
    A registry to map string names to classes.

    Example:
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module
        >>> class ResNet:
        >>>     pass
        >>> model = MODELS.get('ResNet')
    """

    def __init__(self, name):
        """
        Initialize a registry.

        Args:
            name: Name of the registry (e.g., 'models', 'datasets')
        """
        self._name = name
        self._registry = {}

    @property
    def name(self):
        """Return registry name"""
        return self._name

    @property
    def module_dict(self):
        """Return the registry dictionary"""
        return self._registry

    def get(self, key):
        """
        Get a class by name.

        Args:
            key: Class name (string)

        Returns:
            Class object or None if not found
        """
        return self._registry.get(key, None)

    def register_module(self, cls):
        """
        Register a class (used as decorator).

        Args:
            cls: Class to register

        Returns:
            The class itself (for decorator pattern)

        Example:
            >>> @MODELS.register_module
            >>> class ResNet:
            >>>     pass
        """
        if not isinstance(cls, type):
            raise TypeError(f"register_module expects a class, got {type(cls)}")

        class_name = cls.__name__

        if class_name in self._registry:
            raise KeyError(f"'{class_name}' is already registered in {self._name}")

        self._registry[class_name] = cls
        return cls

    def __repr__(self):
        """String representation"""
        return f"Registry(name='{self._name}', items={list(self._registry.keys())})"


def build_from_cfg(cfg, registry, **default_kwargs):
    """
    Build an object from a config dict.

    Args:
        cfg: Config dict with 'type' key specifying the class name
        registry: Registry instance containing the class
        **default_kwargs: Default keyword arguments (config values override these)

    Returns:
        Instantiated object

    Example:
        >>> cfg = {'type': 'ResNet', 'depth': 50}
        >>> model = build_from_cfg(cfg, MODELS, pretrained=True)
        >>> # Calls: ResNet(depth=50, pretrained=True)
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, got {type(cfg)}")

    if 'type' not in cfg:
        raise KeyError("Config dict must contain 'type' key")

    # Copy config to avoid modifying original
    cfg = cfg.copy()
    class_name = cfg.pop('type')

    if not isinstance(class_name, str):
        raise TypeError(f"'type' must be a string, got {type(class_name)}")

    # Get class from registry
    cls = registry.get(class_name)
    if cls is None:
        raise KeyError(
            f"'{class_name}' not found in '{registry.name}' registry. "
            f"Available: {list(registry.module_dict.keys())}"
        )

    # Merge kwargs: default_kwargs are overridden by cfg
    kwargs = {**default_kwargs, **cfg}

    # Instantiate and return
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise TypeError(
            f"Error instantiating {class_name}: {e}\n"
            f"Config: {cfg}\n"
            f"Default kwargs: {default_kwargs}"
        )
