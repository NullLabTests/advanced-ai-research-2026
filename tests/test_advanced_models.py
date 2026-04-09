"""
Comprehensive test suite for advanced AI models
Tests with 90%+ coverage requirement
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.advanced_disinformation_analyzer import (
    AdvancedDisinformationAnalyzer,
    AnalysisResult,
    create_analyzer
)
from src.models.manifold_diffusion_model import (
    ManifoldDiffusionModel,
    DiffusionConfig,
    create_manifold_diffusion
)

class TestAdvancedDisinformationAnalyzer:
    """Test suite for AdvancedDisinformationAnalyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing"""
        with patch('src.models.advanced_disinformation_analyzer.AutoModel'), \
             patch('src.models.advanced_disinformation_analyzer.AutoTokenizer'):
            return AdvancedDisinformationAnalyzer(
                model_name="test-model",
                hidden_dim=256,
                dropout=0.1,
                human_weight=0.7
            )
    
    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing"""
        return [
            "This is a normal news article about local events.",
            "BREAKING: ALIENS HAVE LANDED! SHOCKING EVIDENCE REVEALED!!!",
            "Recent study shows correlation between exercise and health.",
            "AMAZING CURE FOR ALL DISEASES DISCOVERED! DOCTORS HATE THIS!"
        ]
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.model_name == "test-model"
        assert analyzer.hidden_dim == 256
        assert analyzer.human_weight == 0.7
        assert analyzer.enable_explanations is True
        assert len(analyzer.risk_factor_detectors) == 5
    
    def test_analyzer_forward_pass(self, analyzer):
        """Test forward pass of analyzer"""
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        with patch.object(analyzer.transformer, 'forward') as mock_transformer:
            mock_transformer.return_value = Mock(
                last_hidden_state=torch.randn(batch_size, seq_length, 256)
            )
            
            outputs = analyzer.forward(input_ids, attention_mask)
            
            assert 'risk_logits' in outputs
            assert 'emotion_logits' in outputs
            assert 'coherence_score' in outputs
            assert 'credibility_score' in outputs
            assert 'risk_factors' in outputs
            assert 'explanation_embedding' in outputs
            assert 'embeddings' in outputs
    
    def test_analyze_text(self, analyzer, sample_texts):
        """Test text analysis functionality"""
        text = sample_texts[0]
        
        with patch.object(analyzer, 'tokenizer') as mock_tokenizer, \
             patch.object(analyzer, 'forward') as mock_forward:
            
            # Mock tokenizer
            mock_tokenizer.return_value = {
                "input_ids": torch.randint(0, 1000, (1, 10)),
                "attention_mask": torch.ones(1, 10)
            }
            
            # Mock forward pass
            mock_forward.return_value = {
                'risk_logits': torch.randn(1, 2),
                'emotion_logits': torch.randn(1, 5),
                'coherence_score': torch.sigmoid(torch.randn(1, 1)),
                'credibility_score': torch.sigmoid(torch.randn(1, 1)),
                'risk_factors': {
                    'emotional_language': torch.sigmoid(torch.randn(1, 1)),
                    'logical_fallacies': torch.sigmoid(torch.randn(1, 1)),
                    'source_questions': torch.sigmoid(torch.randn(1, 1)),
                    'urgency_tactics': torch.sigmoid(torch.randn(1, 1)),
                    'conspiracy_indicators': torch.sigmoid(torch.randn(1, 1))
                },
                'explanation_embedding': torch.randn(1, 100),
                'embeddings': torch.randn(1, 256)
            }
            
            result = analyzer.analyze_text(text)
            
            assert isinstance(result, AnalysisResult)
            assert result.text == text
            assert 0 <= result.final_risk_score <= 1
            assert 0 <= result.llm_judge_score <= 1
            assert 0 <= result.human_judge_score <= 1
            assert 0 <= result.confidence <= 1
            assert isinstance(result.risk_factors, list)
            assert isinstance(result.explanation, str)
    
    def test_analyze_text_with_human_score(self, analyzer, sample_texts):
        """Test text analysis with provided human score"""
        text = sample_texts[0]
        human_score = 0.8
        
        with patch.object(analyzer, 'tokenizer') as mock_tokenizer, \
             patch.object(analyzer, 'forward') as mock_forward:
            
            mock_tokenizer.return_value = {
                "input_ids": torch.randint(0, 1000, (1, 10)),
                "attention_mask": torch.ones(1, 10)
            }
            
            mock_forward.return_value = {
                'risk_logits': torch.randn(1, 2),
                'emotion_logits': torch.randn(1, 5),
                'coherence_score': torch.sigmoid(torch.randn(1, 1)),
                'credibility_score': torch.sigmoid(torch.randn(1, 1)),
                'risk_factors': {
                    'emotional_language': torch.sigmoid(torch.randn(1, 1)),
                    'logical_fallacies': torch.sigmoid(torch.randn(1, 1)),
                    'source_questions': torch.sigmoid(torch.randn(1, 1)),
                    'urgency_tactics': torch.sigmoid(torch.randn(1, 1)),
                    'conspiracy_indicators': torch.sigmoid(torch.randn(1, 1))
                },
                'explanation_embedding': torch.randn(1, 100),
                'embeddings': torch.randn(1, 256)
            }
            
            result = analyzer.analyze_text(text, human_score=human_score)
            
            assert result.human_judge_score == human_score
    
    def test_batch_analyze(self, analyzer, sample_texts):
        """Test batch analysis functionality"""
        with patch.object(analyzer, 'analyze_text') as mock_analyze:
            mock_analyze.return_value = Mock(
                final_risk_score=0.5,
                llm_judge_score=0.4,
                human_judge_score=0.6,
                confidence=0.8,
                explanation="Test explanation",
                risk_factors=["test_factor"],
                emotional_intensity=0.3,
                logical_coherence=0.7,
                source_credibility=0.8,
                timestamp="2024-01-01T00:00:00"
            )
            
            results = analyzer.batch_analyze(sample_texts, batch_size=2)
            
            assert len(results) == len(sample_texts)
            assert mock_analyze.call_count == len(sample_texts)
    
    def test_calculate_emotional_intensity(self, analyzer):
        """Test emotional intensity calculation"""
        # Test with different text types
        normal_text = "This is a normal sentence."
        emotional_text = "AMAZING! SHOCKING! INCREDIBLE!!!"
        caps_text = "THIS IS ALL CAPS TEXT"
        
        with patch.object(analyzer, 'forward') as mock_forward:
            mock_forward.return_value = {
                'emotion_logits': torch.randn(1, 5),
                'coherence_score': torch.sigmoid(torch.randn(1, 1)),
                'credibility_score': torch.sigmoid(torch.randn(1, 1)),
                'risk_factors': {},
                'explanation_embedding': torch.randn(1, 100),
                'embeddings': torch.randn(1, 256),
                'risk_logits': torch.randn(1, 2)
            }
            
            intensity_normal = analyzer._calculate_emotional_intensity(normal_text, mock_forward.return_value)
            intensity_emotional = analyzer._calculate_emotional_intensity(emotional_text, mock_forward.return_value)
            intensity_caps = analyzer._calculate_emotional_intensity(caps_text, mock_forward.return_value)
            
            assert 0 <= intensity_normal <= 1
            assert 0 <= intensity_emotional <= 1
            assert 0 <= intensity_caps <= 1
            assert intensity_emotional > intensity_normal  # More emotional text should have higher intensity
    
    def test_simulate_human_judge(self, analyzer):
        """Test human judge simulation"""
        with patch.object(analyzer, 'forward') as mock_forward:
            mock_forward.return_value = {
                'emotion_logits': torch.randn(1, 5),
                'coherence_score': torch.sigmoid(torch.randn(1, 1)),
                'credibility_score': torch.sigmoid(torch.randn(1, 1)),
                'risk_factors': {},
                'explanation_embedding': torch.randn(1, 100),
                'embeddings': torch.randn(1, 256),
                'risk_logits': torch.randn(1, 2)
            }
            
            # Test with emotional text
            emotional_text = "AMAZING SHOCKING BREAKING NEWS!"
            human_score = analyzer._simulate_human_judge(emotional_text, mock_forward.return_value)
            
            assert 0 <= human_score <= 1
    
    def test_generate_explanation(self, analyzer):
        """Test explanation generation"""
        outputs = {
            'coherence_score': torch.tensor([[0.3]]),
            'credibility_score': torch.tensor([[0.6]]),
            'risk_factors': {
                'emotional_language': torch.tensor([[0.8]]),
                'logical_fallacies': torch.tensor([[0.2]])
            }
        }
        
        risk_factors = ['Emotional Language']
        
        explanation = analyzer._generate_explanation("test text", outputs, risk_factors)
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
    
    def test_visualize_analysis(self, analyzer, sample_texts):
        """Test visualization functionality"""
        # Create mock results
        results = []
        for text in sample_texts:
            result = Mock(
                final_risk_score=np.random.random(),
                llm_judge_score=np.random.random(),
                human_judge_score=np.random.random(),
                emotional_intensity=np.random.random(),
                logical_coherence=np.random.random(),
                source_credibility=np.random.random(),
                confidence=np.random.random(),
                risk_factors=['test_factor'] if np.random.random() > 0.5 else []
            )
            results.append(result)
        
        with patch('matplotlib.pyplot.show'):
            analyzer.visualize_analysis(results)
        
        # Should not raise any exceptions
        assert True
    
    def test_generate_report(self, analyzer, sample_texts):
        """Test report generation"""
        # Create mock results
        results = []
        for text in sample_texts:
            result = Mock(
                final_risk_score=np.random.random(),
                llm_judge_score=np.random.random(),
                human_judge_score=np.random.random(),
                risk_factors=['test_factor'] if np.random.random() > 0.5 else []
            )
            results.append(result)
        
        report = analyzer.generate_report(results)
        
        assert isinstance(report, str)
        assert "Disinformation Analysis Report" in report
        assert "Summary" in report
        assert "Recommendations" in report
    
    def test_create_analyzer_factory(self):
        """Test analyzer factory function"""
        with patch('src.models.advanced_disinformation_analyzer.AdvancedDisinformationAnalyzer'):
            analyzer = create_analyzer(
                model_name="test-model",
                human_weight=0.8,
                enable_explanations=False
            )
            assert analyzer is not None

class TestManifoldDiffusionModel:
    """Test suite for ManifoldDiffusionModel"""
    
    @pytest.fixture
    def config(self):
        """Create configuration for testing"""
        return DiffusionConfig(
            data_dim=2,
            hidden_dim=256,
            num_layers=2,
            diffusion_steps=50,
            manifold_neighbors=5
        )
    
    @pytest.fixture
    def manifold_model(self, config):
        """Create manifold model for testing"""
        return ManifoldDiffusionModel(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample manifold data"""
        # Generate simple 2D data
        t = np.linspace(0, 4 * np.pi, 100)
        x = t * np.cos(t)
        y = t * np.sin(t)
        data = np.column_stack([x, y])
        return torch.tensor(data, dtype=torch.float32)
    
    def test_manifold_model_initialization(self, manifold_model, config):
        """Test manifold model initialization"""
        assert manifold_model.config.data_dim == config.data_dim
        assert manifold_model.config.hidden_dim == config.hidden_dim
        assert manifold_model.config.diffusion_steps == config.diffusion_steps
        assert manifold_model.config.manifold_neighbors == config.manifold_neighbors
        assert manifold_model.manifold_learner is not None
        assert manifold_model.diffusion_network is not None
    
    def test_diffusion_schedule_setup(self, manifold_model):
        """Test diffusion schedule setup"""
        assert len(manifold_model.betas) == manifold_model.config.diffusion_steps
        assert len(manifold_model.alphas) == manifold_model.config.diffusion_steps
        assert len(manifold_model.alphas_cumprod) == manifold_model.config.diffusion_steps
        
        # Check that betas are in valid range
        assert torch.all(manifold_model.betas >= 0)
        assert torch.all(manifold_model.betas <= 1)
    
    def test_learn_manifold_structure(self, manifold_model, sample_data):
        """Test manifold structure learning"""
        manifold_info = manifold_model.learn_manifold_structure(sample_data)
        
        assert 'adjacency' in manifold_info
        assert 'indices' in manifold_info
        assert 'distances' in manifold_info
        assert 'manifold_coords' in manifold_info
        assert 'tangent_basis' in manifold_info
        
        assert manifold_info['adjacency'].shape[0] == len(sample_data)
        assert manifold_info['manifold_coords'].shape[0] == len(sample_data)
    
    def test_forward_diffusion(self, manifold_model, sample_data):
        """Test forward diffusion process"""
        # Learn manifold first
        manifold_model.learn_manifold_structure(sample_data)
        
        # Test different timesteps
        timesteps = [0, 10, 25, 49]
        
        for t in timesteps:
            t_tensor = torch.tensor([t])
            noisy_data = manifold_model.q_sample(sample_data, t_tensor)
            
            assert noisy_data.shape == sample_data.shape
            assert torch.isfinite(noisy_data).all()
    
    def test_reverse_diffusion_step(self, manifold_model, sample_data):
        """Test single reverse diffusion step"""
        # Learn manifold first
        manifold_model.learn_manifold_structure(sample_data)
        
        # Start with noisy data
        t = torch.tensor([25])
        noisy_data = manifold_model.q_sample(sample_data, t)
        
        # Reverse step
        denoised = manifold_model.p_sample(noisy_data, t)
        
        assert denoised.shape == sample_data.shape
        assert torch.isfinite(denoised).all()
    
    def test_sample_generation(self, manifold_model):
        """Test sample generation"""
        shape = (50, 2)
        samples = manifold_model.sample(shape)
        
        assert samples.shape == shape
        assert torch.isfinite(samples).all()
    
    def test_sample_generation_with_intermediate(self, manifold_model):
        """Test sample generation with intermediate steps"""
        shape = (20, 2)
        intermediates = manifold_model.sample(shape, return_intermediate=True)
        
        assert isinstance(intermediates, list)
        assert len(intermediates) > 1
        assert all(isinstance(step, torch.Tensor) for step in intermediates)
        assert all(step.shape == shape for step in intermediates)
    
    def test_manifold_constraint_application(self, manifold_model, sample_data):
        """Test manifold constraint application"""
        # Learn manifold first
        manifold_model.learn_manifold_structure(sample_data)
        
        # Test constraint application
        x = torch.randn_like(sample_data)
        t = torch.tensor([25])
        
        constrained_x = manifold_model._apply_manifold_constraint(x, t)
        
        assert constrained_x.shape == x.shape
        assert torch.isfinite(constrained_x).all()
    
    def test_manifold_metrics_computation(self, manifold_model, sample_data):
        """Test manifold quality metrics"""
        # Learn manifold first
        manifold_model.learn_manifold_structure(sample_data)
        
        metrics = manifold_model.compute_manifold_metrics(sample_data)
        
        assert 'intrinsic_dimensionality' in metrics
        assert 'correlation_length' in metrics
        assert 'manifold_preservation' in metrics
        assert 'reconstruction_error' in metrics
        
        # Check that metrics are reasonable
        assert metrics['intrinsic_dimensionality'] > 0
        assert 0 <= metrics['manifold_preservation'] <= 1
        assert metrics['reconstruction_error'] >= 0
    
    def test_intrinsic_dimensionality_estimation(self, manifold_model):
        """Test intrinsic dimensionality estimation"""
        # Generate test data
        n_samples = 100
        n_dims = 3
        data = np.random.randn(n_samples, n_dims)
        
        # Estimate dimensionality
        estimated_dim = manifold_model._estimate_intrinsic_dimensionality(data)
        
        assert 1 <= estimated_dim <= n_dims
    
    def test_visualization_2d(self, manifold_model, sample_data):
        """Test 2D visualization"""
        # Learn manifold first
        manifold_model.learn_manifold_structure(sample_data)
        
        # Generate samples
        generated_samples = manifold_model.sample((50, 2))
        
        with patch('matplotlib.pyplot.show'):
            manifold_model.visualize_manifold(sample_data, generated_samples)
        
        # Should not raise any exceptions
        assert True
    
    def test_visualization_3d(self):
        """Test 3D visualization"""
        config = DiffusionConfig(data_dim=3)
        model = ManifoldDiffusionModel(config)
        
        # Generate 3D data
        data = torch.randn(50, 3)
        model.learn_manifold_structure(data)
        
        generated_samples = model.sample((30, 3))
        
        with patch('matplotlib.pyplot.show'):
            model.visualize_manifold(data, generated_samples)
        
        assert True
    
    def test_visualization_highd(self):
        """Test high-dimensional visualization"""
        config = DiffusionConfig(data_dim=10)
        model = ManifoldDiffusionModel(config)
        
        # Generate high-dimensional data
        data = torch.randn(50, 10)
        model.learn_manifold_structure(data)
        
        generated_samples = model.sample((30, 10))
        
        with patch('matplotlib.pyplot.show'):
            model.visualize_manifold(data, generated_samples)
        
        assert True
    
    def test_model_save_load(self, manifold_model, sample_data):
        """Test model saving and loading"""
        # Learn manifold structure
        manifold_model.learn_manifold_structure(sample_data)
        
        # Save model
        import tempfile
        with tempfile.NamedTemporaryFile() as tmp:
            manifold_model.save_model(tmp.name)
            
            # Create new model and load
            config = manifold_model.config
            new_model = ManifoldDiffusionModel(config)
            new_model.load_model(tmp.name)
            
            # Check that learned structure is preserved
            assert new_model.learned_manifold is not None
    
    def test_create_manifold_diffusion_factory(self):
        """Test manifold diffusion factory function"""
        with patch('src.models.manifold_diffusion_model.ManifoldDiffusionModel'):
            model = create_manifold_diffusion(
                data_dim=3,
                hidden_dim=512,
                diffusion_steps=200,
                manifold_neighbors=10
            )
            assert model is not None

class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_analyzer_manifold_integration(self):
        """Test integration between analyzer and manifold model"""
        # This would test real integration scenarios
        # For now, just ensure they can be imported and instantiated together
        with patch('src.models.advanced_disinformation_analyzer.AutoModel'), \
             patch('src.models.advanced_disinformation_analyzer.AutoTokenizer'):
            analyzer = create_analyzer()
            manifold_model = create_manifold_diffusion()
            
            assert analyzer is not None
            assert manifold_model is not None

# Performance tests
class TestPerformance:
    """Performance tests for the models"""
    
    def test_analyzer_performance(self):
        """Test analyzer performance with batch processing"""
        with patch('src.models.advanced_disinformation_analyzer.AutoModel'), \
             patch('src.models.advanced_disinformation_analyzer.AutoTokenizer'):
            analyzer = AdvancedDisinformationAnalyzer(
                model_name="test-model",
                hidden_dim=128,  # Smaller for performance testing
                dropout=0.1
            )
            
            # Generate test texts
            texts = ["Test text " + str(i) for i in range(100)]
            
            with patch.object(analyzer, 'analyze_text') as mock_analyze:
                mock_analyze.return_value = Mock(
                    final_risk_score=0.5,
                    llm_judge_score=0.4,
                    human_judge_score=0.6,
                    confidence=0.8,
                    explanation="Test",
                    risk_factors=[],
                    emotional_intensity=0.3,
                    logical_coherence=0.7,
                    source_credibility=0.8,
                    timestamp="2024-01-01T00:00:00"
                )
                
                import time
                start_time = time.time()
                results = analyzer.batch_analyze(texts, batch_size=32)
                end_time = time.time()
                
                assert len(results) == 100
                assert (end_time - start_time) < 10.0  # Should complete within 10 seconds
    
    def test_manifold_diffusion_performance(self):
        """Test manifold diffusion performance"""
        config = DiffusionConfig(
            data_dim=2,
            hidden_dim=128,
            diffusion_steps=50  # Fewer steps for performance testing
        )
        model = ManifoldDiffusionModel(config)
        
        # Generate sample data
        data = torch.randn(100, 2)
        model.learn_manifold_structure(data)
        
        import time
        start_time = time.time()
        samples = model.sample((200, 2))
        end_time = time.time()
        
        assert samples.shape == (200, 2)
        assert (end_time - start_time) < 15.0  # Should complete within 15 seconds

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src/models", "--cov-report=html", "--cov-fail-under=90"])
