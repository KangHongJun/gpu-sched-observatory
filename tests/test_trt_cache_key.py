from gso.bench.workloads import trt_infer


def test_cache_key_changes_with_each_component():
    base = trt_infer.cache_key("resnet50", 8, "fp16", "11.5.0.63")
    assert base == trt_infer.cache_key("resnet50", 8, "fp16", "11.5.0.63")
    assert base != trt_infer.cache_key("resnet50", 16, "fp16", "11.5.0.63")
    assert base != trt_infer.cache_key("resnet50", 8, "fp32", "11.5.0.63")
    assert base != trt_infer.cache_key("resnet50", 8, "fp16", "11.6.0.0")
    assert base != trt_infer.cache_key("resnet18", 8, "fp16", "11.5.0.63")


def test_cache_key_is_a_safe_filename():
    key = trt_infer.cache_key("resnet50", 8, "fp16", "11.5.0.63")
    assert "/" not in key and " " not in key
