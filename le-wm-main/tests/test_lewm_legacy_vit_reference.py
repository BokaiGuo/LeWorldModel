import pytest
from experiments.spectral_reachability.lewm_legacy_vit_reference import cls_reference,projected_reference


class CPUInput:
    class device:
        type='cpu'


@pytest.mark.parametrize('function',[cls_reference,projected_reference])
def test_no_cpu_neural_reference(function):
    with pytest.raises(ValueError,match='host CUDA'):function(CPUInput(),{})
