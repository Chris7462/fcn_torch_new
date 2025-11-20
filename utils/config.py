"""
A minimal config system. Pythonic configuration loader for Python config files.
"""

import os


class ConfigDict(dict):
    """
    A dictionary that supports attribute-style access.

    Example:
        >>> cfg = ConfigDict({'a': 1, 'b': {'c': 2}})
        >>> cfg.a
        1
        >>> cfg.b.c
        2
    """

    def __init__(self, data=None):
        super().__init__()
        if data is not None:
            for key, value in data.items():
                self[key] = value

    def __setitem__(self, key, value):
        """Auto-convert nested dicts to ConfigDict"""
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
        super().__setitem__(key, value)

    def __getattr__(self, key):
        """Allow attribute-style access: cfg.key"""
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        """Allow attribute-style setting: cfg.key = value"""
        self[key] = value

    def __delattr__(self, key):
        """Allow attribute-style deletion: del cfg.key"""
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")


class Config:
    """
    Configuration loader for Python config files.

    Example:
        >>> cfg = Config.from_file('configs/camvid.py')
        >>> print(cfg.batch_size)
        16
        >>> print(cfg.optimizer.lr)
        0.002
    """

    def __init__(self, cfg_dict, filename=None):
        """
        Initialize Config with a dictionary.

        Args:
            cfg_dict: Dictionary of configuration values
            filename: Path to the config file (optional)
        """
        if not isinstance(cfg_dict, dict):
            raise TypeError(f"cfg_dict must be a dict, got {type(cfg_dict)}")

        self._cfg_dict = ConfigDict(cfg_dict)
        self._filename = filename

    @classmethod
    def from_file(cls, filename):
        """
        Load configuration from a Python file.

        Args:
            filename: Path to .py config file

        Returns:
            Config object

        Example:
            >>> cfg = Config.from_file('configs/camvid.py')
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Config file not found: {filename}")

        if not filename.endswith('.py'):
            raise ValueError("Only .py config files are supported")

        # Read the config file
        with open(filename, 'r') as f:
            config_text = f.read()

        # Execute the config file in a namespace
        namespace = {}
        try:
            exec(config_text, namespace)
        except Exception as e:
            raise RuntimeError(f"Error executing config file {filename}: {e}")

        # Extract config variables (exclude builtins and imports)
        cfg_dict = {
            key: value
            for key, value in namespace.items()
            if not key.startswith('_')
        }

        return cls(cfg_dict, filename=filename)

    @property
    def filename(self):
        """Return the config filename"""
        return self._filename

    def __getattr__(self, name):
        """Allow attribute-style access: cfg.batch_size"""
        try:
            return getattr(self._cfg_dict, name)
        except AttributeError:
            raise AttributeError(f"'Config' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """Allow attribute-style setting: cfg.batch_size = 16"""
        if name in ['_cfg_dict', '_filename']:
            # Internal attributes
            super().__setattr__(name, value)
        else:
            # User attributes - set on the config dict
            setattr(self._cfg_dict, name, value)

    def __getitem__(self, name):
        """Allow dict-style access: cfg['batch_size']"""
        return self._cfg_dict[name]

    def __setitem__(self, name, value):
        """Allow dict-style setting: cfg['batch_size'] = 16"""
        self._cfg_dict[name] = value

    def __contains__(self, name):
        """Allow 'in' operator: 'batch_size' in cfg"""
        return name in self._cfg_dict

    def __repr__(self):
        """String representation of config"""
        if self._filename:
            return f"Config(file='{self._filename}', keys={list(self._cfg_dict.keys())})"
        return f"Config(keys={list(self._cfg_dict.keys())})"

    def __str__(self):
        """Pretty print config"""
        def format_dict(d, indent=0):
            lines = []
            for key, value in d.items():
                if isinstance(value, (ConfigDict, dict)):
                    lines.append('  ' * indent + f"{key}:")
                    lines.append(format_dict(value, indent + 1))
                else:
                    lines.append('  ' * indent + f"{key}: {repr(value)}")
            return '\n'.join(lines)

        header = f"Config from: {self._filename}\n" if self._filename else "Config\n"
        return header + "=" * 50 + "\n" + format_dict(self._cfg_dict)
