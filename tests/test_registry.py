"""
Unit tests for utils/registry.py
"""

# pylint: disable=redefined-outer-name,too-few-public-methods

import pytest
from utils.registry import Registry, build_from_cfg


@pytest.fixture
def empty_registry():
    """Provides an empty registry"""
    return Registry('models')


@pytest.fixture
def sample_registry():
    """Provides a registry with pre-registered classes"""
    registry = Registry('models')

    @registry.register_module
    class ModelA:  # pylint: disable=unused-variable
        """Test model with optional param"""
        def __init__(self, param1=10):
            self.param1 = param1

    @registry.register_module
    class ModelB:  # pylint: disable=unused-variable
        """Test model with required and optional params"""
        def __init__(self, param1, param2=20):
            self.param1 = param1
            self.param2 = param2

    return registry


class TestRegistry:
    """Tests for Registry class"""

    def test_initialization(self, empty_registry):
        """Test registry creation with a name"""
        assert empty_registry.name == 'models'
        assert empty_registry.module_dict == {}

    def test_name_property(self, empty_registry):
        """Test name property returns correct name"""
        assert empty_registry.name == 'models'

    def test_module_dict_property(self, empty_registry):
        """Test module_dict property returns registry dictionary"""
        assert isinstance(empty_registry.module_dict, dict)
        assert len(empty_registry.module_dict) == 0

    def test_register_single_class(self, empty_registry):
        """Test successful class registration using decorator"""
        @empty_registry.register_module
        class TestModel:  # pylint: disable=unused-variable
            """Test model for registration"""

        assert 'TestModel' in empty_registry.module_dict

    def test_register_multiple_classes(self, empty_registry):
        """Test multiple classes can be registered"""
        @empty_registry.register_module
        class Model1:  # pylint: disable=unused-variable
            """First test model"""

        @empty_registry.register_module
        class Model2:  # pylint: disable=unused-variable
            """Second test model"""

        assert len(empty_registry.module_dict) == 2
        assert 'Model1' in empty_registry.module_dict
        assert 'Model2' in empty_registry.module_dict

    def test_register_duplicate_raises_error(self, empty_registry):
        """Test duplicate registration raises KeyError"""
        @empty_registry.register_module
        class DuplicateModel:  # pylint: disable=unused-variable
            """Model to be duplicated"""

        with pytest.raises(KeyError, match="DuplicateModel is already registered"):
            @empty_registry.register_module
            class DuplicateModel:  # pylint: disable=unused-variable,function-redefined
                """Duplicate model"""

    def test_register_non_class_raises_error(self, empty_registry):
        """Test registering non-class objects raises TypeError"""
        with pytest.raises(TypeError, match="module must be a class"):
            empty_registry.register_module("not a class")

        with pytest.raises(TypeError, match="module must be a class"):
            empty_registry.register_module(123)

    def test_get_existing_class(self, sample_registry):
        """Test get() returns correct class"""
        cls = sample_registry.get('ModelA')
        assert cls is not None
        assert cls.__name__ == 'ModelA'

    def test_get_non_existent_class(self, sample_registry):
        """Test get() returns None for non-existent key"""
        cls = sample_registry.get('NonExistentModel')
        assert cls is None

    def test_repr(self, sample_registry):
        """Test __repr__() output"""
        repr_str = repr(sample_registry)
        assert "Registry" in repr_str
        assert "models" in repr_str
        assert "ModelA" in repr_str
        assert "ModelB" in repr_str


class TestBuildFromCfg:
    """Tests for build_from_cfg function"""

    def test_basic_instantiation(self, sample_registry):
        """Test successful object instantiation from config dict"""
        cfg = {'type': 'ModelA', 'param1': 100}
        obj = build_from_cfg(cfg, sample_registry)

        assert obj is not None
        assert obj.param1 == 100

    def test_config_params_passed(self, sample_registry):
        """Test config params are passed to constructor"""
        cfg = {'type': 'ModelB', 'param1': 50, 'param2': 60}
        obj = build_from_cfg(cfg, sample_registry)

        assert obj.param1 == 50
        assert obj.param2 == 60

    def test_default_args(self, sample_registry):
        """Test default_args work correctly"""
        cfg = {'type': 'ModelA'}
        obj = build_from_cfg(cfg, sample_registry, default_args={'param1': 200})

        assert obj.param1 == 200

    def test_config_overrides_default_args(self, sample_registry):
        """Test config overrides default_args"""
        cfg = {'type': 'ModelA', 'param1': 300}
        obj = build_from_cfg(cfg, sample_registry, default_args={'param1': 200})

        # Config value should override default_args
        assert obj.param1 == 300

    def test_default_args_none(self, sample_registry):
        """Test default_args=None works"""
        cfg = {'type': 'ModelA'}
        obj = build_from_cfg(cfg, sample_registry, default_args=None)

        # Should use ModelA's default param1=10
        assert obj.param1 == 10

    def test_original_config_not_modified(self, sample_registry):
        """Test that original config dict is not modified"""
        cfg = {'type': 'ModelA', 'param1': 100}
        original_cfg = cfg.copy()

        build_from_cfg(cfg, sample_registry)

        assert cfg == original_cfg
        assert 'type' in cfg

    def test_non_dict_config_raises_error(self, sample_registry):
        """Test TypeError when cfg is not a dict"""
        with pytest.raises(TypeError, match="cfg must be a dict"):
            build_from_cfg("not a dict", sample_registry)

        with pytest.raises(TypeError, match="cfg must be a dict"):
            build_from_cfg([1, 2, 3], sample_registry)

    def test_missing_type_key_raises_error(self, sample_registry):
        """Test KeyError when 'type' key is missing"""
        cfg = {'param1': 100}

        with pytest.raises(KeyError, match="cfg must contain the key 'type'"):
            build_from_cfg(cfg, sample_registry)

    def test_non_string_type_raises_error(self, sample_registry):
        """Test TypeError when 'type' is not a string"""
        cfg = {'type': 123, 'param1': 100}

        with pytest.raises(TypeError, match="type must be a str"):
            build_from_cfg(cfg, sample_registry)

    def test_class_not_found_raises_error(self, sample_registry):
        """Test KeyError when class not found in registry"""
        cfg = {'type': 'NonExistentModel', 'param1': 100}

        with pytest.raises(KeyError, match="'NonExistentModel' is not in the 'models' registry"):
            build_from_cfg(cfg, sample_registry)

    def test_default_args_invalid_type_raises_error(self, sample_registry):
        """Test TypeError when default_args is not a dict or None"""
        cfg = {'type': 'ModelA'}

        with pytest.raises(TypeError, match="default_args must be a dict or None"):
            build_from_cfg(cfg, sample_registry, default_args="invalid")

    def test_multiple_default_args(self, sample_registry):
        """Test multiple default_args are passed correctly"""
        cfg = {'type': 'ModelB'}
        obj = build_from_cfg(cfg, sample_registry, default_args={'param1': 100, 'param2': 200})

        assert obj.param1 == 100
        assert obj.param2 == 200

    def test_partial_override(self, sample_registry):
        """Test config partially overrides default_args"""
        cfg = {'type': 'ModelB', 'param1': 50}
        obj = build_from_cfg(cfg, sample_registry, default_args={'param1': 100, 'param2': 200})

        assert obj.param1 == 50   # From config
        assert obj.param2 == 200  # From default_args
