"""
Unit tests for utils/registry.py
"""

import pytest
from utils.registry import Registry, build_from_cfg


# Fixtures
@pytest.fixture
def empty_registry():
    """Provides an empty registry"""
    return Registry('models')


@pytest.fixture
def sample_registry():
    """Provides a registry with pre-registered classes"""
    registry = Registry('models')

    @registry.register_module
    class ModelA:
        def __init__(self, param1=10):
            self.param1 = param1

    @registry.register_module
    class ModelB:
        def __init__(self, param1, param2=20):
            self.param1 = param1
            self.param2 = param2

    return registry


# Registry class tests
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
        class TestModel:
            pass

        assert 'TestModel' in empty_registry.module_dict
        assert empty_registry.module_dict['TestModel'] is TestModel

    def test_register_multiple_classes(self, empty_registry):
        """Test multiple classes can be registered"""
        @empty_registry.register_module
        class Model1:
            pass

        @empty_registry.register_module
        class Model2:
            pass

        assert len(empty_registry.module_dict) == 2
        assert 'Model1' in empty_registry.module_dict
        assert 'Model2' in empty_registry.module_dict

    def test_register_duplicate_raises_error(self, empty_registry):
        """Test duplicate registration raises KeyError"""
        @empty_registry.register_module
        class DuplicateModel:
            pass

        with pytest.raises(KeyError, match="'DuplicateModel' is already registered"):
            @empty_registry.register_module
            class DuplicateModel:
                pass

    def test_register_non_class_raises_error(self, empty_registry):
        """Test registering non-class objects raises TypeError"""
        with pytest.raises(TypeError, match="register_module expects a class"):
            empty_registry.register_module("not a class")

        with pytest.raises(TypeError, match="register_module expects a class"):
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
        assert "name='models'" in repr_str
        assert "ModelA" in repr_str
        assert "ModelB" in repr_str


# build_from_cfg function tests
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

    def test_default_kwargs(self, sample_registry):
        """Test default_kwargs work correctly"""
        cfg = {'type': 'ModelA'}
        obj = build_from_cfg(cfg, sample_registry, param1=200)

        assert obj.param1 == 200

    def test_config_overrides_defaults(self, sample_registry):
        """Test config overrides default_kwargs"""
        cfg = {'type': 'ModelA', 'param1': 300}
        obj = build_from_cfg(cfg, sample_registry, param1=200)

        # Config value should override default
        assert obj.param1 == 300

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

        with pytest.raises(KeyError, match="Config dict must contain 'type' key"):
            build_from_cfg(cfg, sample_registry)

    def test_non_string_type_raises_error(self, sample_registry):
        """Test TypeError when 'type' is not a string"""
        cfg = {'type': 123, 'param1': 100}

        with pytest.raises(TypeError, match="'type' must be a string"):
            build_from_cfg(cfg, sample_registry)

    def test_class_not_found_raises_error(self, sample_registry):
        """Test KeyError when class not found in registry"""
        cfg = {'type': 'NonExistentModel', 'param1': 100}

        with pytest.raises(KeyError, match="'NonExistentModel' not found in 'models' registry"):
            build_from_cfg(cfg, sample_registry)

    def test_instantiation_error(self, sample_registry):
        """Test TypeError when instantiation fails with wrong params"""
        # ModelB requires param1, but we don't provide it
        cfg = {'type': 'ModelB'}

        with pytest.raises(TypeError, match="Error instantiating ModelB"):
            build_from_cfg(cfg, sample_registry)
