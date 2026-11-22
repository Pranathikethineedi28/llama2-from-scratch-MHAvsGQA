import pytest
import torch

from llama.model import KVCache, LlamaModel, ModelArgs, count_parameters, generate


def tiny(n_kv_heads: int) -> LlamaModel:
    torch.manual_seed(0)
    return LlamaModel(ModelArgs(vocab_size=20, dim=32, n_layers=2, n_heads=4, n_kv_heads=n_kv_heads,
                                hidden_dim=64, max_seq_len=64))


@pytest.mark.parametrize("kv", [4, 2, 1])
def test_output_shape(kv):
    tokens = torch.randint(0, 20, (2, 10))
    assert tiny(kv)(tokens).shape == (2, 10, 20)


def test_gqa_shrinks_only_key_value_projections():
    mha, gqa = tiny(4), tiny(2)
    head_dim = 32 // 4
    # Each layer's K and V projections lose (4 - 2) heads * head_dim * dim parameters.
    expected_saving = 2 * 2 * (4 - 2) * head_dim * 32  # layers * (K and V) * removed heads * head_dim * dim
    assert count_parameters(mha) - count_parameters(gqa) == expected_saving


def test_causal_mask_future_tokens_do_not_change_past_outputs():
    model = tiny(2).eval()
    a = torch.randint(0, 20, (1, 12))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 1) % 20  # change only the future
    with torch.no_grad():
        assert torch.allclose(model(a)[0, :8], model(b)[0, :8], atol=1e-5)


@pytest.mark.parametrize("kv", [4, 2, 1])
def test_kv_cache_gives_same_logits_as_full_forward(kv):
    model = tiny(kv).eval()
    tokens = torch.randint(0, 20, (1, 16))
    with torch.no_grad():
        full = model(tokens)
        cache = KVCache(model.args.n_layers)
        prefix = model(tokens[:, :10], 0, cache)
        steps = [model(tokens[:, i:i + 1], i, cache) for i in range(10, 16)]
    incremental = torch.cat([prefix] + steps, dim=1)
    assert torch.allclose(full, incremental, atol=1e-5)


def test_kv_cache_size_matches_the_formula():
    for kv in (4, 2, 1):
        model = tiny(kv)
        _, nbytes = generate(model, torch.zeros((1, 5), dtype=torch.long), max_new_tokens=11)
        # 5 prompt tokens + 10 fed back through the cache (the last sampled token is never fed)
        assert nbytes == model.args.kv_cache_bytes_per_token(4) * 15


def test_cache_ratio_is_heads_over_kv_heads():
    mha = ModelArgs(n_heads=8, n_kv_heads=8).kv_cache_bytes_per_token()
    gqa = ModelArgs(n_heads=8, n_kv_heads=2).kv_cache_bytes_per_token()
    assert mha / gqa == 4


def test_invalid_head_split_is_rejected():
    with pytest.raises(ValueError):
        LlamaModel(ModelArgs(n_heads=8, n_kv_heads=3))
