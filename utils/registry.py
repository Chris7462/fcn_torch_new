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
        self._module_dict = {}

    @property
    def name(self):
        """Return registry name"""
        return self._name

    @property
    def module_dict(self):
        """Return the registry dictionary"""
        return self._module_dict

    def get(self, key):
        """
        Get a class by name.

        Args:
            key: Class name (string)

        Returns:
            Class object or None if not found
        """
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        """
        Register a module.

        Args:
            module_class: Class to be registered

        Raises:
            TypeError: If module_class is not a class
            KeyError: If class name is already registered
        """
        if not isinstance(module_class, type):
            raise TypeError(
                f'module must be a class, but got {type(module_class)}'
            )
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError(
                f'{module_name} is already registered in {self.name}'
            )
        self._module_dict[module_name] = module_class

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
        self._register_module(cls)
        return cls

    def __repr__(self):
        """String representation"""
        return f"Registry(name='{self._name}', items={list(self._module_dict.keys())})"


def build_from_cfg(cfg, registry):
    """
    Build a module from config dict.

    Args:
        cfg (dict): Config dict with 'type' key specifying the class name.
        registry (Registry): Registry instance containing the class.

    Returns:
        object: Instantiated object.

    Example:
        >>> cfg = {'type': 'ResNet', 'depth': 50}
        >>> model = build_from_cfg(cfg, MODELS)
        >>> # Calls: ResNet(depth=50, pretrained=True)
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')

    if 'type' not in cfg:
        raise KeyError("cfg must contain the key 'type'")

    # Copy config to avoid modifying original
    args = cfg.copy()
    obj_type = args.pop('type')

    # Get class from registry by name
    if not isinstance(obj_type, str):
        raise TypeError(f"type must be a str, but got {type(obj_type)}")

    obj_cls = registry.get(obj_type)
    if obj_cls is None:
        raise KeyError(
            f"'{obj_type}' is not in the '{registry.name}' registry. "
            f"Available: {list(registry.module_dict.keys())}"
        )

    return obj_cls(**args)
