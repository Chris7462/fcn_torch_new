"""
Unit tests for utils/config.py
"""

import os
import tempfile
import pytest
from utils.config import Config, ConfigDict


# ConfigDict tests
class TestConfigDict:
    """Tests for ConfigDict class"""

    def test_initialization_empty(self):
        """Test empty ConfigDict creation"""
        cfg = ConfigDict()
        assert len(cfg) == 0

    def test_initialization_with_data(self):
        """Test ConfigDict creation with initial data"""
        cfg = ConfigDict({'a': 1, 'b': 2})
        assert cfg['a'] == 1
        assert cfg['b'] == 2

    def test_nested_dict_conversion(self):
        """Test nested dicts are converted to ConfigDict"""
        cfg = ConfigDict({'outer': {'inner': 1}})
        assert isinstance(cfg['outer'], ConfigDict)
        assert cfg['outer']['inner'] == 1

    def test_attribute_access_get(self):
        """Test attribute-style access for getting values"""
        cfg = ConfigDict({'key': 'value'})
        assert cfg.key == 'value'

    def test_attribute_access_set(self):
        """Test attribute-style access for setting values"""
        cfg = ConfigDict()
        cfg.key = 'value'
        assert cfg['key'] == 'value'

    def test_attribute_access_nested(self):
        """Test nested attribute access"""
        cfg = ConfigDict({'a': {'b': {'c': 3}}})
        assert cfg.a.b.c == 3

    def test_attribute_access_missing_raises_error(self):
        """Test accessing missing attribute raises AttributeError"""
        cfg = ConfigDict({'a': 1})
        with pytest.raises(AttributeError, match="has no attribute 'missing'"):
            _ = cfg.missing

    def test_delete_attribute(self):
        """Test attribute deletion"""
        cfg = ConfigDict({'a': 1, 'b': 2})
        del cfg.a
        assert 'a' not in cfg
        assert 'b' in cfg

    def test_delete_missing_attribute_raises_error(self):
        """Test deleting missing attribute raises AttributeError"""
        cfg = ConfigDict()
        with pytest.raises(AttributeError, match="has no attribute 'missing'"):
            del cfg.missing


# Config tests
class TestConfig:
    """Tests for Config class"""

    @pytest.fixture
    def sample_config_file(self):
        """Create a temporary config file"""
        content = '''
# Sample config
batch_size = 16
learning_rate = 0.001

optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9
)

dataset = dict(
    train=dict(type='ImageNet', split='train'),
    val=dict(type='ImageNet', split='val')
)
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            f.flush()
            yield f.name
        os.unlink(f.name)

    def test_from_file(self, sample_config_file):
        """Test loading config from file"""
        cfg = Config.from_file(sample_config_file)
        assert cfg.batch_size == 16
        assert cfg.learning_rate == 0.001

    def test_from_file_not_found(self):
        """Test FileNotFoundError for missing file"""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            Config.from_file('nonexistent.py')

    def test_from_file_wrong_extension(self):
        """Test ValueError for non-.py files"""
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            f.write(b'key: value')
            f.flush()
            try:
                with pytest.raises(ValueError, match="Only .py config files"):
                    Config.from_file(f.name)
            finally:
                os.unlink(f.name)

    def test_attribute_access(self, sample_config_file):
        """Test attribute-style access"""
        cfg = Config.from_file(sample_config_file)
        assert cfg.batch_size == 16
        assert cfg.optimizer.type == 'SGD'
        assert cfg.optimizer.lr == 0.01

    def test_nested_attribute_access(self, sample_config_file):
        """Test deeply nested attribute access"""
        cfg = Config.from_file(sample_config_file)
        assert cfg.dataset.train.type == 'ImageNet'
        assert cfg.dataset.train.split == 'train'

    def test_dict_access(self, sample_config_file):
        """Test dict-style access"""
        cfg = Config.from_file(sample_config_file)
        assert cfg['batch_size'] == 16
        assert cfg['optimizer']['type'] == 'SGD'

    def test_set_attribute(self, sample_config_file):
        """Test setting attributes"""
        cfg = Config.from_file(sample_config_file)
        cfg.batch_size = 32
        assert cfg.batch_size == 32

    def test_set_item(self, sample_config_file):
        """Test setting items dict-style"""
        cfg = Config.from_file(sample_config_file)
        cfg['batch_size'] = 64
        assert cfg['batch_size'] == 64

    def test_contains(self, sample_config_file):
        """Test 'in' operator"""
        cfg = Config.from_file(sample_config_file)
        assert 'batch_size' in cfg
        assert 'nonexistent' not in cfg

    def test_missing_attribute_raises_error(self, sample_config_file):
        """Test accessing missing attribute raises AttributeError"""
        cfg = Config.from_file(sample_config_file)
        with pytest.raises(AttributeError, match="has no attribute 'missing'"):
            _ = cfg.missing

    def test_filename_property(self, sample_config_file):
        """Test filename property"""
        cfg = Config.from_file(sample_config_file)
        assert cfg.filename == sample_config_file

    def test_repr(self, sample_config_file):
        """Test __repr__ output"""
        cfg = Config.from_file(sample_config_file)
        repr_str = repr(cfg)
        assert 'Config' in repr_str
        assert 'batch_size' in repr_str

    def test_str(self, sample_config_file):
        """Test __str__ output"""
        cfg = Config.from_file(sample_config_file)
        str_output = str(cfg)
        assert 'batch_size' in str_output
        assert '16' in str_output

    def test_init_requires_dict(self):
        """Test Config.__init__ requires dict"""
        with pytest.raises(TypeError, match="cfg_dict must be a dict"):
            Config("not a dict")

    def test_config_with_imports(self):
        """Test config file with imports works"""
        content = '''
import os
data_root = os.path.join('data', 'images')
batch_size = 8
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            f.flush()
            try:
                cfg = Config.from_file(f.name)
                assert cfg.batch_size == 8
                assert 'images' in cfg.data_root
            finally:
                os.unlink(f.name)
