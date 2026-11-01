import pytest
from experiments.spectral_reachability import agent_e3_feedback_transport_budget_20261029_v1 as b


@pytest.mark.parametrize('k',[1,3])
def test_partition_integer_bound(k):
    x=b.budget(k)
    assert x['total']==sum(x['parts'].values())<=x['cap']
    assert x['encoding_fixed_arrays_fit'] and x['encoding_gzip_bound']>=x['encoding_npy']
    assert x['physical_and_metadata_fit_actual_writer_unproven']
    assert not x['full_production_transport_proven']


def test_two_full_rgb_arrays_and_header():
    sizes=b.encoding_array_sizes()
    assert len(sizes)==7 and sizes[0]>224*224*3
    assert sizes[1]>4*224*224*3 and sizes[3]>4*257*192
    assert b.budget(3)['parts']['physical_PNG']==32*256*1024


@pytest.mark.parametrize('n',[-1,True,1.1])
def test_bad_compression_sizes(n):
    with pytest.raises(ValueError):b.compress_bound(n)
